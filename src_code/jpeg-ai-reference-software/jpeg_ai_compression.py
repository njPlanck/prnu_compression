"""
Encode image directories with the JPEG-AI reference software.

Camera RAW inputs are demosaiced to RGB and all input formats are staged as
PNG because the reference encoder accepts PNG/YUV input. The output includes
The JPEG-AI bitstream and per-camera metric reports. Reconstructed PNGs are
created temporarily for quality metrics and removed after each image.
"""

import argparse
import csv
import gc
import glob
import os
import sys
import tempfile
import faulthandler
from multiprocessing import cpu_count
from pathlib import Path

faulthandler.enable(all_threads=True)

# Edit these defaults to run the script directly from an IDE. Command-line
# positional paths and --target-size-kb override them when provided.
DEFAULT_INPUT_DIR = Path(
    "/home/chinasa/python_projects/fivek_prnu_split/query_set"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/chinasa/python_projects/fivek_prnu_split/"
    "query_set_compressed_jpegai"
)
DEFAULT_TARGET_SIZE_KIB = 500

import torch
import torchvision.transforms as transforms
from PIL import Image
from piq import brisque, psnr, ssim
from tqdm import tqdm

JPEG_AI_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(JPEG_AI_ROOT))

from src.codec import get_downloader
from src.codec.common import Image as JPEGImage
from src.codec.coders import (
    CodecEncoder,
    def_encoder_base_parser,
    def_encoder_parser_decorator,
)


RAW_EXTENSIONS = {
    ".3fr", ".arw", ".cr2", ".cr3", ".dng", ".fff", ".iiq", ".kdc",
    ".mef", ".mos", ".mrw", ".nef", ".nrw", ".orf", ".pef", ".raf",
    ".rw2", ".sr2", ".srf", ".srw",
}
IMAGE_EXTENSIONS = {
    ".png", ".jpeg", ".jpg", ".tiff", ".tif", ".dng", ".webp", ".bmp", ".heic", ".heif",
    *RAW_EXTENSIONS,
}
METRICS_TRANSFORM = transforms.ToTensor()


def load_input_image_as_pil(input_path):
    """Load RGB input, demosaicing camera RAW files to 8-bit pixels."""
    if Path(input_path).suffix.lower() in RAW_EXTENSIONS:
        try:
            import rawpy
        except ImportError as error:
            raise RuntimeError(
                "Reading camera RAW images requires rawpy; install it with "
                "'python -m pip install rawpy' in the JPEG-AI environment"
            ) from error
        with rawpy.imread(input_path) as raw:
            rgb_array = raw.postprocess(
                use_camera_wb=True,
                half_size=False,
                no_auto_bright=True,
                output_bps=8,
            )
        return Image.fromarray(rgb_array)

    with Image.open(input_path) as image:
        return image.convert("RGB")


def compute_bpp(file_path, width, height):
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions: {width}x{height}")
    return (os.path.getsize(file_path) * 8) / (width * height)


def target_bpp_x100(target_size_kb, width, height):
    """Convert target KiB to JPEG-AI's hundredths-of-bpp target units."""
    if target_size_kb <= 0:
        raise ValueError("Target size must be greater than zero KiB")
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions: {width}x{height}")

    target_bytes = target_size_kb * 1024
    target_rate = round(target_bytes * 8 * 100 / (width * height))
    if target_rate < 1:
        raise ValueError(
            f"Target size {target_size_kb:g} KiB is below JPEG-AI's "
            "0.01-bpp rate precision for this image"
        )
    if target_rate > 200:
        raise ValueError(
            f"Target size {target_size_kb:g} KiB exceeds JPEG-AI's "
            "supported maximum of 2.00 bpp for this image"
        )
    return target_rate


def evaluate_metrics(original, reconstructed):
    original = original.convert("RGB")
    reconstructed = reconstructed.convert("RGB")
    x = METRICS_TRANSFORM(original).unsqueeze(0)
    y = METRICS_TRANSFORM(reconstructed).unsqueeze(0)
    if x.shape != y.shape:
        raise ValueError(
            "Input and reconstructed images have different dimensions: "
            f"{tuple(x.shape)} vs {tuple(y.shape)}"
        )
    return {
        "PSNR": psnr(x, y, data_range=1.0).item(),
        "SSIM": ssim(x, y).item(),
        "BRISQUE": brisque(y).item(),
    }


class RecoEncoder(CodecEncoder):
    def __init__(self, base_parser, parser_decorator, name="reco"):
        super().__init__(name, base_parser, parser_decorator)

    def encode_and_decode_with_metrics(
        self,
        input_path,
        bitstream_path,
        target_size_kb=None,
        set_target_bpp=1,
    ):
        """Encode input and report rates/metrics without retaining a decoded PNG."""
        try:
            original = load_input_image_as_pil(input_path)
            width, height = original.size
            target_rate = (
                target_bpp_x100(target_size_kb, width, height)
                if target_size_kb is not None
                else set_target_bpp
            )

            # The reference Image loader accepts PNG/YUV, not camera RAW files.
            with tempfile.TemporaryDirectory(prefix="jpeg_ai_input_") as temp_dir:
                png_path = os.path.join(temp_dir, "input.png")
                original.save(png_path, format="PNG")
                jpeg_ai_image = JPEGImage.read_file(png_path)

                if self.ce.target_device == "cpu":
                    torch.set_num_threads(max(1, cpu_count() // 2))

                self.ce.target_bpps = [target_rate]
                self.set_target_bpp_idx(0)
                self.rec_image, decisions = self.ce.compress(jpeg_ai_image)

                self.create_bs(bitstream_path)
                self.init_ec_module()
                self.ce.encode(self.ec_module, decisions)
                self.close_bs()
                del decisions
                del jpeg_ai_image
                decoded_png_path = os.path.join(temp_dir, "reconstruction.png")
                self.rec_image.write_png(decoded_png_path)

                actual_bitstream_path = (
                    bitstream_path
                    if os.path.exists(bitstream_path)
                    else bitstream_path + ".bin"
                )
                if not os.path.exists(actual_bitstream_path):
                    raise FileNotFoundError(
                        f"JPEG-AI bitstream was not written: {bitstream_path}"
                    )

                with Image.open(decoded_png_path) as decoded_file:
                    reconstructed = decoded_file.convert("RGB")
                metrics = evaluate_metrics(original, reconstructed)
            metrics.update({
                "BPP": compute_bpp(actual_bitstream_path, width, height),
                "Target Size (KiB)": target_size_kb,
                "Actual Size (KiB)": os.path.getsize(actual_bitstream_path) / 1024,
                "Target BPP": target_rate / 100,
                "Status": "OK",
            })
            return metrics
        except Exception as error:
            return {"error": str(error)}
        finally:
            self.rec_image = None
            self.ec_module = None
            self.bs = None
            gc.collect()


def list_images(directory):
    image_files = set()
    for extension in IMAGE_EXTENSIONS:
        image_files.update(glob.glob(os.path.join(directory, f"*{extension}")))
        image_files.update(glob.glob(os.path.join(directory, f"*{extension.upper()}")))
    return sorted(image_files)


def process_dir_with_encoder(
    coder,
    input_dir,
    output_dir,
    target_size_kb=None,
    set_target_bpp=1,
    max_samples=None,
):
    input_root = Path(input_dir).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_root}")
    if output_root == input_root or input_root in output_root.parents:
        raise ValueError(
            f"Output directory must not be inside the input tree: {output_root}"
        )
    input_dir = str(input_root)
    output_dir = str(output_root)

    target_name = (
        f"target_size_{target_size_kb:g}KiB"
        if target_size_kb is not None
        else f"target_bpp_{set_target_bpp}"
    )
    target_output_dir = os.path.join(output_dir, target_name)
    os.makedirs(target_output_dir, exist_ok=True)
    metrics_by_camera = {}
    processed_count = 0

    for root, directories, _ in os.walk(input_dir):
        directories.sort()
        image_files = list_images(root)
        if not image_files:
            continue

        rel_path = os.path.relpath(root, input_dir)
        path_parts = [] if rel_path == "." else rel_path.split(os.sep)
        camera_name = path_parts[0] if path_parts else "default_camera"
        camera_dir = camera_name.replace(" ", "_")
        camera_subdirs = path_parts[1:]
        sub_save_dir = os.path.join(
            target_output_dir, camera_dir, *camera_subdirs
        )
        os.makedirs(sub_save_dir, exist_ok=True)
        camera_rows = metrics_by_camera.setdefault(camera_name, [])

        for image_path in tqdm(image_files, desc=f"Processing {camera_name}"):
            if max_samples is not None and processed_count >= max_samples:
                break

            filename = os.path.basename(image_path)
            extension = os.path.splitext(filename)[1]
            base_name = os.path.splitext(filename)[0]
            bitstream_path = os.path.join(sub_save_dir, base_name + ".bits")

            if os.path.exists(bitstream_path):
                print(f"Skipping (bitstream already exists): {bitstream_path}")
                processed_count += 1
                continue

            metrics = coder.encode_and_decode_with_metrics(
                image_path,
                bitstream_path,
                target_size_kb=target_size_kb,
                set_target_bpp=set_target_bpp,
            )
            processed_count += 1

            if "error" in metrics:
                print(f"ERROR {filename}: {metrics['error']}")
                camera_rows.append({
                    "Image": filename,
                    "Format": extension.upper(),
                    "Target Size (KiB)": target_size_kb,
                    "Actual Size (KiB)": None,
                    "Target BPP": None,
                    "PSNR": None,
                    "SSIM": None,
                    "BRISQUE": None,
                    "BPP": None,
                    "Status": metrics["error"],
                })
                continue

            print(
                f"{filename}: target={metrics['Target Size (KiB)']} KiB, "
                f"actual={metrics['Actual Size (KiB)']:.2f} KiB, "
                f"PSNR={metrics['PSNR']:.2f}, SSIM={metrics['SSIM']:.4f}, "
                f"BPP={metrics['BPP']:.4f}"
            )
            camera_rows.append({
                "Image": filename,
                "Format": extension.upper(),
                "Target Size (KiB)": target_size_kb,
                "Actual Size (KiB)": round(metrics["Actual Size (KiB)"], 3),
                "Target BPP": round(metrics["Target BPP"], 2),
                "PSNR": round(metrics["PSNR"], 2),
                "SSIM": round(metrics["SSIM"], 4),
                "BRISQUE": round(metrics["BRISQUE"], 2),
                "BPP": round(metrics["BPP"], 4),
                "Status": "OK",
            })

        if max_samples is not None and processed_count >= max_samples:
            break

    fieldnames = [
        "Image",
        "Format",
        "Target Size (KiB)",
        "Actual Size (KiB)",
        "Target BPP",
        "PSNR",
        "SSIM",
        "BRISQUE",
        "BPP",
        "Status",
    ]
    for camera_name, csv_rows in metrics_by_camera.items():
        if not csv_rows:
            continue
        csv_path = os.path.join(
            target_output_dir, f"{camera_name.replace(' ', '_')}_metrics.csv"
        )
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nMetrics saved for {camera_name}: {csv_path}")


def create_argument_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Demosaic camera RAW images to RGB PNG, then encode/decode them "
            "with the JPEG-AI reference software."
        )
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default=str(DEFAULT_INPUT_DIR),
        help=f"Input directory of images (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "output_path",
        nargs="?",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory for outputs (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--target-size-kb",
        type=float,
        nargs="+",
        default=None,
        help="One or more target bitstream sizes in KiB (1 KiB = 1024 bytes)",
    )
    parser.add_argument(
        "--set-target-bpp",
        type=int,
        default=1,
        help=(
            "Legacy fixed target in hundredths of a bit/pixel; used when "
            "--target-size-kb is omitted"
        ),
    )
    parser.add_argument(
        "--models-dir-name",
        default="models",
        help="JPEG-AI model directory, relative to the current working directory",
    )
    parser.add_argument("--gpu", type=int, help="GPU index to use")
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=1,
        help="PyTorch CPU threads per process (default: 1 to limit peak memory)",
    )
    parser.add_argument("--num-samples", type=int, help="Maximum images per target")
    return parser


def initialize_encoder(args, initial_target_bpp):
    encoder_parser = def_encoder_base_parser("Reconstruction")
    coder = RecoEncoder(
        encoder_parser, def_encoder_parser_decorator(encoder_parser)
    )
    seed_input = os.path.join(args.input_path, "jpeg_ai_seed.png")
    seed_bitstream = os.path.join(args.output_path, "jpeg_ai_seed.bits")
    codec_args = [
        seed_input,
        seed_bitstream,
        "--set_target_bpp",
        str(initial_target_bpp),
        "--models_dir_name",
        args.models_dir_name,
    ]

    coder.print_coder_info()
    kwargs, _, _ = coder.init_common_codec(
        build_model=True,
        ce=None,
        cmd_args=codec_args,
        overload_ce=True,
        cmd_args_add=False,
    )
    coder.load_models(
        get_downloader(
            kwargs.get("models_dir_name", "models"),
            critical_for_file_absence=not kwargs.get("skip_loading_error", False),
        )
    )
    return coder


def main(argv=None):
    args = create_argument_parser().parse_args(argv)
    target_sizes_kb = (
        args.target_size_kb
        if args.target_size_kb is not None
        else [DEFAULT_TARGET_SIZE_KIB]
    )
    if target_sizes_kb:
        if any(size <= 0 for size in target_sizes_kb):
            raise ValueError("Every target size must be greater than zero KiB")
        target_sizes = list(dict.fromkeys(target_sizes_kb))
        initial_target_bpp = 1
    else:
        if not 1 <= args.set_target_bpp <= 200:
            raise ValueError(
                "Target BPP must be between 1 (0.01 bpp) and 200 (2.00 bpp)"
            )
        target_sizes = [None]
        initial_target_bpp = args.set_target_bpp

    if args.num_samples is not None and args.num_samples < 1:
        raise ValueError("--num-samples must be greater than zero")
    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads must be greater than zero")
    input_path = Path(args.input_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_path}")
    if output_path == input_path or input_path in output_path.parents:
        raise ValueError(
            f"Output directory must not be inside the input tree: {output_path}"
        )
    os.makedirs(output_path, exist_ok=True)

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"Using GPU {args.gpu}")
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)

    args.input_path = str(input_path)
    args.output_path = str(output_path)
    coder = initialize_encoder(args, initial_target_bpp)
    for target_size_kb in target_sizes:
        process_dir_with_encoder(
            coder,
            args.input_path,
            args.output_path,
            target_size_kb=target_size_kb,
            set_target_bpp=args.set_target_bpp,
            max_samples=args.num_samples,
        )


if __name__ == "__main__":
    main()
