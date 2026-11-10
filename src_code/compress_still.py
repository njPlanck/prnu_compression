"""Compress camera image datasets with AVIF, HEIC, BPG, and WebP still-image codecs."""

from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
from PIL import Image, ImageOps, features


DEFAULT_INPUT_DIR = Path(
    "/home/chinasa/python_projects/fivek_prnu_split/query_set"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/chinasa/python_projects/fivek_prnu_split/query_set_still_compressed_500kb"
)
DEFAULT_TARGET_SIZE_KB = [500]
DEFAULT_TARGET_SIZE_TOLERANCE_KB = 12.0
SUPPORTED_CODECS = ("avif", "heic", "bpg", "webp")
METRIC_FIELDNAMES = [
    "Image",
    "Codec",
    "TargetKiB",
    "Quality",
    "EncoderMethod",
    "SizeBytes",
    "BPP",
    "PSNR",
    "SSIM",
    "BRISQUE",
    "MetricsStatus",
    "Status",
]
SUPPORTED_INPUT_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".dng",
}
IMAGEMAGICK = shutil.which("magick") or shutil.which("convert")
AVIF_SPEED = 8
HEIF_PLUGIN_DIRECTORY = Path.home() / ".local" / "lib" / "libheif" / "plugins"


def _find_tool(name: str) -> Optional[str]:
    found = shutil.which(name)
    if found:
        return found
    user_local_tool = Path.home() / ".local" / "bin" / name
    return str(user_local_tool) if user_local_tool.is_file() else None


def _require_tool(name: str) -> str:
    tool_path = _find_tool(name)
    if tool_path is None:
        raise RuntimeError(f"Required command not found: {name}")
    return tool_path


def _heif_plugin_args() -> List[str]:
    if HEIF_PLUGIN_DIRECTORY.is_dir():
        return ["--plugin-directory", str(HEIF_PLUGIN_DIRECTORY)]
    return []


def _run(command: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


@lru_cache(maxsize=4)
def _has_heic_encoder(encoder_path: str) -> bool:
    result = _run([encoder_path, *_heif_plugin_args(), "--list-encoders"])
    if result.returncode != 0:
        return False
    output = result.stdout.decode("utf-8", errors="replace")
    heic_section = output.partition("HEIC encoders:")[2].partition("AVIF encoders:")[0]
    return any(line.lstrip().startswith("- ") for line in heic_section.splitlines())


def _imagemagick_format_flags() -> Dict[str, str]:
    if IMAGEMAGICK is None:
        return {}
    result = _run([IMAGEMAGICK, "-list", "format"])
    if result.returncode != 0:
        return {}

    flags = {}
    for line in result.stdout.decode("utf-8", errors="replace").splitlines():
        parts = line.split()
        if not parts:
            continue
        format_name = parts[0].rstrip("*").upper()
        flag_index = 1
        if parts[0].endswith("*") and len(parts) >= 3:
            format_name = parts[1].upper()
            flag_index = 2
        elif format_name not in {"AVIF", "HEIC", "WEBP"} and len(parts) >= 3:
            format_name = parts[1].upper()
            flag_index = 2
        if format_name in {"AVIF", "HEIC", "WEBP"} and len(parts) > flag_index:
            flags[format_name] = parts[flag_index]
    return flags


def codec_unavailable_reason(codec: str, format_flags: Dict[str, str]) -> Optional[str]:
    codec = codec.lower()
    if codec not in SUPPORTED_CODECS:
        return f"Unsupported codec: {codec}"

    if codec == "avif":
        if IMAGEMAGICK is None:
            return "AVIF unavailable: ImageMagick (magick/convert) is required"
        flags = format_flags.get(codec.upper(), "")
        if "w" not in flags:
            return (
                "AVIF unavailable: ImageMagick was built without "
                "AVIF encoding support"
            )
        return None

    if codec == "webp":
        if not features.check("webp"):
            return "WebP unavailable: Pillow was built without WebP support"
        return None

    if codec == "heic":
        if IMAGEMAGICK is not None and "w" in format_flags.get("HEIC", ""):
            return None
        encoder_path = _find_tool("heif-enc")
        if encoder_path is None:
            return (
                "HEIC unavailable: encoding requires ImageMagick HEIC write "
                "support or the heif-enc command"
            )
        if not _has_heic_encoder(encoder_path):
            return (
                "HEIC unavailable: libheif has no HEVC encoder. Install the "
                "libheif x265 encoder plugin."
            )
        if (
            _find_tool("heif-convert") is None
            and (IMAGEMAGICK is None or "r" not in format_flags.get("HEIC", ""))
        ):
            return "HEIC unavailable: a HEIC decoder (heif-convert or ImageMagick) is required"
        return None

    if _find_tool("bpgenc") is None or _find_tool("bpgdec") is None:
        return "BPG unavailable: both bpgenc and bpgdec commands are required"
    return None


def load_input_image(input_path: Path) -> Image.Image:
    """Load an RGB image, developing DNG files with rawpy."""
    if input_path.suffix.lower() == ".dng":
        try:
            import rawpy
        except ImportError as exc:
            raise RuntimeError(
                "DNG input requires rawpy; install it with `python -m pip install rawpy`"
            ) from exc
        with rawpy.imread(str(input_path)) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                half_size=False,
                no_auto_bright=True,
            )
        return Image.fromarray(rgb)

    with Image.open(input_path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def _run_encoder(codec: str, command: List[str], output_path: Path) -> None:
    result = _run(command)
    if result.returncode != 0 or not output_path.is_file():
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            detail
            or f"{codec.upper()} encoder failed with exit code {result.returncode}"
        )


def _encode_avif_candidate(
    input_png: Path,
    output_path: Path,
    quality: int,
) -> None:
    if IMAGEMAGICK is None:
        raise RuntimeError("AVIF encoding requires ImageMagick")
    command = [
        IMAGEMAGICK,
        str(input_png),
        "-strip",
        "-quality",
        str(quality),
        "-define",
        f"heic:speed={AVIF_SPEED}",
        str(output_path),
    ]
    _run_encoder("avif", command, output_path)


def _encode_heic_candidate(
    input_png: Path,
    output_path: Path,
    quality: int,
    format_flags: Dict[str, str],
) -> None:
    if IMAGEMAGICK is not None and "w" in format_flags.get("HEIC", ""):
        command = [
            IMAGEMAGICK,
            str(input_png),
            "-strip",
            "-quality",
            str(quality),
            str(output_path),
        ]
    else:
        command = [
            _require_tool("heif-enc"),
            *_heif_plugin_args(),
            "-q",
            str(quality),
            str(input_png),
            "-o",
            str(output_path),
        ]
    _run_encoder("heic", command, output_path)


def _encode_bpg_candidate(
    input_png: Path,
    output_path: Path,
    quality: int,
) -> None:
    command = [
        _require_tool("bpgenc"),
        "-q",
        str(quality),
        "-o",
        str(output_path),
        str(input_png),
    ]
    _run_encoder("bpg", command, output_path)


def _encode_webp_candidate(
    input_png: Path,
    output_path: Path,
    quality: int,
    method: int,
) -> None:
    with Image.open(input_png) as image:
        image.save(
            output_path,
            format="WEBP",
            quality=quality,
            method=method,
            lossless=False,
        )
    if not output_path.is_file():
        raise RuntimeError("WEBP encoder did not produce an output file")


def _encode_candidate(
    codec: str,
    input_png: Path,
    output_path: Path,
    quality: int,
    format_flags: Dict[str, str],
    webp_method: int = 6,
) -> None:
    """Dispatch a quality candidate to its codec-specific encoder."""
    if codec == "avif":
        _encode_avif_candidate(input_png, output_path, quality)
    elif codec == "heic":
        _encode_heic_candidate(input_png, output_path, quality, format_flags)
    elif codec == "bpg":
        _encode_bpg_candidate(input_png, output_path, quality)
    elif codec == "webp":
        _encode_webp_candidate(input_png, output_path, quality, webp_method)
    else:
        raise ValueError(f"Unsupported codec: {codec}")


def _decode_candidate(codec: str, encoded_path: Path, png_path: Path) -> None:
    codec = codec.lower()
    if codec == "bpg":
        command = [_require_tool("bpgdec"), "-o", str(png_path), str(encoded_path)]
    elif codec == "webp":
        with Image.open(encoded_path) as image:
            image.save(png_path, format="PNG")
        return
    elif codec == "heic" and _find_tool("heif-convert") is not None:
        command = [_require_tool("heif-convert"), str(encoded_path), str(png_path)]
    else:
        if IMAGEMAGICK is None:
            raise RuntimeError("Decoding this output requires ImageMagick")
        command = [IMAGEMAGICK, str(encoded_path), str(png_path)]

    result = _run(command)
    if result.returncode != 0 or not png_path.is_file():
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(detail or f"{codec.upper()} verification decode failed")


def _evaluate_metrics(original: Image.Image, decoded: Image.Image) -> Dict[str, float]:
    if original.size != decoded.size:
        raise ValueError(
            f"Decoded dimensions changed from {original.size} to {decoded.size}"
        )
    try:
        import torch
        from piq import brisque, psnr, ssim
    except ImportError as exc:
        raise RuntimeError(
            "Image metrics require torch and piq; install the project's requirements"
        ) from exc

    original_array = np.asarray(original.convert("RGB"), dtype=np.float32) / 255.0
    decoded_array = np.asarray(decoded.convert("RGB"), dtype=np.float32) / 255.0
    reference = torch.from_numpy(original_array).permute(2, 0, 1).unsqueeze(0)
    reconstruction = torch.from_numpy(decoded_array).permute(2, 0, 1).unsqueeze(0)
    return {
        "PSNR": float(psnr(reference, reconstruction, data_range=1.0).item()),
        "SSIM": float(ssim(reference, reconstruction, data_range=1.0).item()),
        "BRISQUE": float(brisque(reconstruction, data_range=1.0).item()),
    }


def _quality_limits(codec: str) -> Tuple[int, int]:
    return (0, 51) if codec.lower() == "bpg" else (0, 100)


def compress_to_target_size(
    input_path: Path,
    output_path: Path,
    target_size_kb: float,
    codec: str,
    format_flags: Optional[Dict[str, str]] = None,
) -> Dict[str, object]:
    """Save the tested quality whose output is closest to the target size."""
    codec = codec.lower()
    if not math.isfinite(target_size_kb) or target_size_kb <= 0:
        return {"error": "Target size must be a finite positive number"}
    format_flags = format_flags if format_flags is not None else _imagemagick_format_flags()
    unavailable = codec_unavailable_reason(codec, format_flags)
    if unavailable:
        return {"error": unavailable}

    try:
        original = load_input_image(input_path)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        return {"error": f"Image load failed: {exc}"}

    target_bytes = int(target_size_kb * 1024)
    min_quality, max_quality = _quality_limits(codec)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"{codec}_") as temp_dir:
        temp_root = Path(temp_dir)
        input_png = temp_root / "input.png"
        candidate = temp_root / f"candidate.{codec}"
        best_path = temp_root / f"best.{codec}"
        original.save(input_png, format="PNG")

        bpg_quantizer = codec == "bpg"
        best_quality = None
        best_webp_method = None
        best_size = None
        try:
            webp_methods = range(7) if codec == "webp" else (6,)
            for webp_method in webp_methods:
                low, high = min_quality, max_quality
                while low <= high:
                    quality = (low + high) // 2
                    candidate.unlink(missing_ok=True)
                    _encode_candidate(
                        codec,
                        input_png,
                        candidate,
                        quality,
                        format_flags,
                        webp_method,
                    )
                    candidate_size = candidate.stat().st_size
                    candidate_difference = abs(candidate_size - target_bytes)
                    best_difference = (
                        abs(best_size - target_bytes)
                        if best_size is not None
                        else None
                    )
                    is_better = (
                        best_difference is None
                        or candidate_difference < best_difference
                    )
                    if candidate_difference == best_difference:
                        if codec == "webp":
                            is_better = (
                                best_webp_method is None
                                or webp_method > best_webp_method
                                or (
                                    webp_method == best_webp_method
                                    and (
                                        best_quality is None
                                        or quality > best_quality
                                    )
                                )
                            )
                        elif best_quality is None:
                            is_better = True
                        else:
                            is_better = (
                                quality < best_quality
                                if bpg_quantizer
                                else quality > best_quality
                            )
                    if is_better:
                        best_quality = quality
                        best_webp_method = (
                            webp_method if codec == "webp" else None
                        )
                        best_size = candidate_size
                        shutil.copyfile(candidate, best_path)

                    if (candidate_size < target_bytes) != bpg_quantizer:
                        low = quality + 1
                    else:
                        high = quality - 1

            if best_quality is None or best_size is None:
                return {"error": f"{codec.upper()} quality search found no candidates"}

            decoded_png = temp_root / "decoded.png"
            _decode_candidate(codec, best_path, decoded_png)
            with Image.open(decoded_png) as decoded_source:
                decoded = decoded_source.convert("RGB")
            if original.size != decoded.size:
                raise ValueError(
                    f"Decoded dimensions changed from {original.size} to {decoded.size}"
                )
            metrics_error = None
            try:
                metrics = _evaluate_metrics(original, decoded)
            except RuntimeError as exc:
                metrics = {"PSNR": None, "SSIM": None, "BRISQUE": None}
                metrics_error = str(exc)
            width, height = original.size
            metrics.update(
                {
                    "Quality": best_quality,
                    "EncoderMethod": best_webp_method,
                    "SizeBytes": best_size,
                    "TargetDifferenceKiB": (best_size - target_bytes) / 1024,
                    "BPP": (best_size * 8.0) / (width * height),
                    "Width": width,
                    "Height": height,
                    "MetricsError": metrics_error,
                }
            )
            temporary_output = output_path.with_name(output_path.name + ".partial")
            try:
                shutil.copyfile(best_path, temporary_output)
                os.replace(temporary_output, output_path)
            finally:
                temporary_output.unlink(missing_ok=True)
            return metrics
        except (OSError, RuntimeError, ValueError) as exc:
            return {"error": f"{codec.upper()} compression failed: {exc}"}


def compress_avif_to_target_size(
    input_path: Path,
    output_path: Path,
    target_size_kb: float,
    format_flags: Optional[Dict[str, str]] = None,
) -> Dict[str, object]:
    """Encode candidates as AVIF with ImageMagick, then select the closest size."""
    return compress_to_target_size(
        input_path, output_path, target_size_kb, "avif", format_flags
    )


def compress_heic_to_target_size(
    input_path: Path,
    output_path: Path,
    target_size_kb: float,
    format_flags: Optional[Dict[str, str]] = None,
) -> Dict[str, object]:
    """Encode candidates as HEIC with ImageMagick or the libheif encoder."""
    return compress_to_target_size(
        input_path, output_path, target_size_kb, "heic", format_flags
    )


def compress_bpg_to_target_size(
    input_path: Path,
    output_path: Path,
    target_size_kb: float,
    format_flags: Optional[Dict[str, str]] = None,
) -> Dict[str, object]:
    """Encode candidates with bpgenc and search its quantizer range."""
    return compress_to_target_size(
        input_path, output_path, target_size_kb, "bpg", format_flags
    )


def compress_webp_to_target_size(
    input_path: Path,
    output_path: Path,
    target_size_kb: float,
    format_flags: Optional[Dict[str, str]] = None,
) -> Dict[str, object]:
    """Encode WebP candidates with Pillow while searching quality and method."""
    return compress_to_target_size(
        input_path, output_path, target_size_kb, "webp", format_flags
    )


CODEC_COMPRESSORS = {
    "avif": compress_avif_to_target_size,
    "heic": compress_heic_to_target_size,
    "bpg": compress_bpg_to_target_size,
    "webp": compress_webp_to_target_size,
}


def _supported_images(input_dir: Path) -> Iterable[Path]:
    for path in sorted(input_dir.rglob("*")):
        if (
            path.is_file()
            and path.suffix.lower() in SUPPORTED_INPUT_EXTENSIONS
            and not path.name.lower().startswith("merge")
        ):
            yield path


def _target_directory_name(target_size_kb: float) -> str:
    label = f"{target_size_kb:g}".replace(".", "p")
    return f"target_{label}kb"


def _write_camera_metrics(
    target_root: Path,
    camera_folder: str,
    rows: List[Dict[str, object]],
) -> None:
    target_root.mkdir(parents=True, exist_ok=True)
    csv_path = target_root / f"{camera_folder}_metrics.csv"
    temporary_path = csv_path.with_name(csv_path.name + ".partial")
    try:
        with temporary_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=METRIC_FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_path, csv_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _append_camera_metric(
    target_root: Path,
    camera_folder: str,
    row: Dict[str, object],
) -> None:
    target_root.mkdir(parents=True, exist_ok=True)
    csv_path = target_root / f"{camera_folder}_metrics.csv"
    if csv_path.is_file() and csv_path.stat().st_size > 0:
        with csv_path.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            if reader.fieldnames != METRIC_FIELDNAMES:
                rows = list(reader)
                rows.append(row)
                _write_camera_metrics(target_root, camera_folder, rows)
                return

    write_header = not csv_path.is_file() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=METRIC_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        csv_file.flush()


def _read_camera_metrics(path: Path) -> List[Dict[str, object]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _recover_existing_output(
    input_path: Path,
    output_path: Path,
    codec: str,
) -> Dict[str, object]:
    original = load_input_image(input_path)
    with tempfile.TemporaryDirectory(prefix=f"{codec}_resume_") as temp_dir:
        decoded_path = Path(temp_dir) / "decoded.png"
        _decode_candidate(codec, output_path, decoded_path)
        with Image.open(decoded_path) as decoded_source:
            decoded = decoded_source.convert("RGB")
    if original.size != decoded.size:
        raise ValueError(
            f"Decoded dimensions changed from {original.size} to {decoded.size}"
        )

    metrics_error = "Recovered existing output; encoder quality is unknown"
    metrics: Dict[str, object] = {
        "PSNR": None,
        "SSIM": None,
        "BRISQUE": None,
    }
    try:
        metrics.update(_evaluate_metrics(original, decoded))
        metrics_error = ""
    except RuntimeError as exc:
        metrics_error = f"Metrics unavailable: {exc}"

    size_bytes = output_path.stat().st_size
    width, height = original.size
    metrics.update(
        {
            "Quality": None,
            "EncoderMethod": None,
            "SizeBytes": size_bytes,
            "BPP": (size_bytes * 8.0) / (width * height),
            "Width": width,
            "Height": height,
            "MetricsError": metrics_error,
        }
    )
    return metrics


def process_dataset(
    input_dir: Path,
    output_dir: Path,
    target_sizes_kb: Iterable[float],
    codecs: Iterable[str] = SUPPORTED_CODECS,
    resume: bool = False,
    target_tolerance_kb: float = DEFAULT_TARGET_SIZE_TOLERANCE_KB,
) -> None:
    """Compress supported images and persist per-camera progress incrementally."""
    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if output_dir == input_dir or input_dir in output_dir.parents:
        raise ValueError("Output directory must be outside the input directory tree")

    codec_list = tuple(dict.fromkeys(codec.lower() for codec in codecs))
    invalid_codecs = [codec for codec in codec_list if codec not in SUPPORTED_CODECS]
    if invalid_codecs:
        raise ValueError(f"Unsupported codecs: {', '.join(invalid_codecs)}")
    targets = tuple(target_sizes_kb)
    if not targets or any(not math.isfinite(size) or size <= 0 for size in targets):
        raise ValueError("At least one finite, positive target size is required")
    if not math.isfinite(target_tolerance_kb) or target_tolerance_kb < 0:
        raise ValueError("Target size tolerance must be finite and non-negative")

    images = list(_supported_images(input_dir))
    if not images:
        raise FileNotFoundError(
            f"No supported images found under {input_dir} "
            f"({', '.join(sorted(SUPPORTED_INPUT_EXTENSIONS))})"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    format_flags = _imagemagick_format_flags()
    for codec in codec_list:
        unavailable = codec_unavailable_reason(codec, format_flags)
        if unavailable:
            print(f"WARNING: {unavailable}")

    for target_size_kb in targets:
        target_root = output_dir / _target_directory_name(target_size_kb)
        rows_by_camera: Dict[str, List[Dict[str, object]]] = {}
        initialized_cameras = set()
        if resume:
            for metrics_path in target_root.glob("*_metrics.csv"):
                rows_by_camera[metrics_path.name[: -len("_metrics.csv")]] = (
                    _read_camera_metrics(metrics_path)
                )
        try:
            for input_path in images:
                relative_path = input_path.relative_to(input_dir)
                if len(relative_path.parts) < 2:
                    print(f"Skipping image outside a camera subfolder: {relative_path}")
                    continue
                camera_name = relative_path.parts[0]
                camera_folder = camera_name.replace(" ", "_")
                print(f"\nImage: {relative_path} | camera: {camera_name}")
                camera_rows = rows_by_camera.setdefault(camera_folder, [])
                if not resume and camera_folder not in initialized_cameras:
                    old_metrics = target_root / f"{camera_folder}_metrics.csv"
                    old_metrics.unlink(missing_ok=True)
                    initialized_cameras.add(camera_folder)

                for codec in codec_list:
                    extension = "heic" if codec == "heic" else codec
                    output_path = (
                        target_root
                        / codec
                        / relative_path.parent
                        / f"{relative_path.name}.{extension}"
                    )
                    previous_row = next(
                        (
                            existing
                            for existing in camera_rows
                            if existing.get("Image") == str(relative_path)
                            and existing.get("Codec") == codec.upper()
                            and existing.get("Status") == "OK"
                        ),
                        None,
                    )
                    if (
                        resume
                        and previous_row is not None
                        and output_path.is_file()
                        and abs(
                            output_path.stat().st_size - target_size_kb * 1024
                        )
                        <= target_tolerance_kb * 1024
                    ):
                        print(f"  {codec.upper()}: skipping completed output")
                        continue
                    if (
                        resume
                        and previous_row is None
                        and output_path.is_file()
                        and abs(
                            output_path.stat().st_size - target_size_kb * 1024
                        )
                        <= target_tolerance_kb * 1024
                    ):
                        try:
                            result = _recover_existing_output(
                                input_path, output_path, codec
                            )
                        except (OSError, RuntimeError, ValueError) as exc:
                            print(
                                f"  {codec.upper()}: existing output could not be "
                                f"verified ({exc}); re-encoding"
                            )
                        else:
                            row = {
                                "Image": str(relative_path),
                                "Codec": codec.upper(),
                                "TargetKiB": target_size_kb,
                                "Quality": result.get("Quality"),
                                "EncoderMethod": result.get("EncoderMethod"),
                                "SizeBytes": result.get("SizeBytes"),
                                "BPP": result.get("BPP"),
                                "PSNR": result.get("PSNR"),
                                "SSIM": result.get("SSIM"),
                                "BRISQUE": result.get("BRISQUE"),
                                "MetricsStatus": result.get("MetricsError") or "OK",
                                "Status": "OK",
                            }
                            camera_rows.append(row)
                            _append_camera_metric(target_root, camera_folder, row)
                            print(
                                f"  {codec.upper()}: recovered existing output "
                                f"({result['SizeBytes'] / 1024:.2f} KiB)"
                            )
                            continue

                    try:
                        result = CODEC_COMPRESSORS[codec](
                            input_path,
                            output_path,
                            target_size_kb,
                            format_flags,
                        )
                    except KeyboardInterrupt:
                        result = {"error": "Interrupted by user"}
                        row = {
                            "Image": str(relative_path),
                            "Codec": codec.upper(),
                            "TargetKiB": target_size_kb,
                            "Quality": None,
                            "EncoderMethod": None,
                            "SizeBytes": None,
                            "BPP": None,
                            "PSNR": None,
                            "SSIM": None,
                            "BRISQUE": None,
                            "MetricsStatus": "",
                            "Status": result["error"],
                        }
                        replacing_previous = any(
                            existing.get("Image") == str(relative_path)
                            and existing.get("Codec") == codec.upper()
                            for existing in camera_rows
                        )
                        camera_rows[:] = [
                            existing
                            for existing in camera_rows
                            if not (
                                existing.get("Image") == str(relative_path)
                                and existing.get("Codec") == codec.upper()
                            )
                        ]
                        camera_rows.append(row)
                        if replacing_previous:
                            _write_camera_metrics(
                                target_root, camera_folder, camera_rows
                            )
                        else:
                            _append_camera_metric(target_root, camera_folder, row)
                        raise

                    row: Dict[str, object] = {
                        "Image": str(relative_path),
                        "Codec": codec.upper(),
                        "TargetKiB": target_size_kb,
                        "Quality": result.get("Quality"),
                        "EncoderMethod": result.get("EncoderMethod"),
                        "SizeBytes": result.get("SizeBytes"),
                        "BPP": result.get("BPP"),
                        "PSNR": result.get("PSNR"),
                        "SSIM": result.get("SSIM"),
                        "BRISQUE": result.get("BRISQUE"),
                        "MetricsStatus": result.get("MetricsError") or "OK",
                        "Status": "OK" if "error" not in result else result["error"],
                    }
                    replacing_previous = any(
                        existing.get("Image") == str(relative_path)
                        and existing.get("Codec") == codec.upper()
                        for existing in camera_rows
                    )
                    camera_rows[:] = [
                        existing
                        for existing in camera_rows
                        if not (
                            existing.get("Image") == str(relative_path)
                            and existing.get("Codec") == codec.upper()
                        )
                    ]
                    camera_rows.append(row)

                    if "error" in result:
                        if (
                            output_path.is_file()
                            and abs(
                                output_path.stat().st_size - target_size_kb * 1024
                            )
                            > target_tolerance_kb * 1024
                        ):
                            output_path.unlink()
                        print(f"  {codec.upper()}: {result['error']}")
                    else:
                        encoding_parameters = f"q={result['Quality']}"
                        if result.get("EncoderMethod") is not None:
                            encoding_parameters += (
                                f", method={result['EncoderMethod']}"
                            )
                        print(
                            f"  {codec.upper()}: {result['SizeBytes'] / 1024:.2f} KiB "
                            f"(target delta={result['TargetDifferenceKiB']:+.2f} KiB, "
                            f"{encoding_parameters}, BPP={result['BPP']:.4f}, "
                            + (
                                f"PSNR={result['PSNR']:.2f}, SSIM={result['SSIM']:.4f}"
                                if result["PSNR"] is not None
                                else f"metrics unavailable: {result['MetricsError']}"
                            )
                            + ")"
                        )
                    if replacing_previous:
                        _write_camera_metrics(target_root, camera_folder, camera_rows)
                    else:
                        _append_camera_metric(target_root, camera_folder, row)
        except KeyboardInterrupt:
            print(
                "\nInterrupted by user. Completed outputs and per-camera CSV "
                "progress have been preserved."
            )
            return

        for camera_folder in rows_by_camera:
            print(f"Metrics saved: {target_root / f'{camera_folder}_metrics.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compress camera image folders with AVIF, HEIC, BPG, and WebP "
            "to one or more per-image target sizes."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Image root with camera subfolders (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Compression output root (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--target-size-kb",
        type=float,
        nargs="+",
        default=DEFAULT_TARGET_SIZE_KB,
        help="Target output size per image, in KiB; pass multiple targets if needed.",
    )
    parser.add_argument(
        "--target-size-tolerance-kb",
        type=float,
        default=DEFAULT_TARGET_SIZE_TOLERANCE_KB,
        help=(
            "Maximum allowed difference above or below each target, in KiB "
            f"(default: {DEFAULT_TARGET_SIZE_TOLERANCE_KB:g})."
        ),
    )
    parser.add_argument(
        "--codecs",
        nargs="+",
        choices=SUPPORTED_CODECS,
        default=list(SUPPORTED_CODECS),
        help="Codecs to run (default: all four).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip successful outputs already within the target-size tolerance.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    process_dataset(
        arguments.input_dir,
        arguments.output_dir,
        arguments.target_size_kb,
        arguments.codecs,
        arguments.resume,
        arguments.target_size_tolerance_kb,
    )