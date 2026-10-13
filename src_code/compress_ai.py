import argparse
import gc
import math
import pickle
import struct

import torch
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image

from compressai.zoo import bmshj2018_hyperprior


# ============================================================
# Configuration
# ============================================================

torch.set_num_threads(1)

MAGIC = b"CAI1"
VERSION = 1


# ============================================================
# Model
# ============================================================

def load_model(quality, device="cpu"):
    """
    Load the pretrained CompressAI BMShJ2018 hyperprior.

    quality:
        1 = lowest bitrate
        8 = highest bitrate
    """

    model = bmshj2018_hyperprior(
        quality=quality,
        metric="mse",
        pretrained=True,
    )

    model = model.to(device)
    model.eval()

    # Build/update entropy coding tables.
    model.update(force=True)

    return model


# ============================================================
# Image utilities
# ============================================================

def pad_for_network(x):
    """
    Internally pad the image to a multiple of 64.

    IMPORTANT:
        This does NOT resize the image.

    Example:

        4386 x 2920
            ->
        4416 x 2944

    The extra pixels are removed after decoding.
    """

    _, _, h, w = x.shape

    padded_h = math.ceil(h / 64) * 64
    padded_w = math.ceil(w / 64) * 64

    pad_h = padded_h - h
    pad_w = padded_w - w

    if pad_h == 0 and pad_w == 0:
        return x, padded_h, padded_w

    x = F.pad(
        x,
        (0, pad_w, 0, pad_h),
        mode="constant",
        value=0.0,
    )

    return x, padded_h, padded_w


def tensor_to_pil(x):
    """
    Convert [1,3,H,W] tensor to RGB PIL image.
    """

    x = x.clamp(0, 1)

    x = (
        x * 255.0 + 0.5
    ).byte()

    x = x.squeeze(0)

    x = x.permute(
        1,
        2,
        0,
    )

    return Image.fromarray(
        x.cpu().numpy(),
        "RGB",
    )


# ============================================================
# Binary container
# ============================================================

def create_container(
    quality,
    orig_h,
    orig_w,
    padded_h,
    padded_w,
    latent_shape,
    strings,
):
    """
    Create a self-contained compressed file.

    Stored:

        model quality
        original image dimensions
        internal padded dimensions
        latent shape
        entropy-coded y stream
        entropy-coded z stream

    The neural-network weights are NOT stored in every image.
    The decoder loads the matching pretrained model.
    """

    y_string = strings[0][0]
    z_string = strings[1][0]

    metadata = {
        "version": VERSION,

        "quality": int(quality),

        "metric": "mse",

        # Actual image dimensions.
        "orig_h": int(orig_h),
        "orig_w": int(orig_w),

        # Internal network dimensions.
        "padded_h": int(padded_h),
        "padded_w": int(padded_w),

        # Required by CompressAI decompress().
        "latent_shape": (
            int(latent_shape[0]),
            int(latent_shape[1]),
        ),

        "y_size": len(y_string),
        "z_size": len(z_string),
    }

    metadata_bytes = pickle.dumps(
        metadata,
        protocol=pickle.HIGHEST_PROTOCOL,
    )

    header = (
        MAGIC
        + struct.pack(
            "<B",
            VERSION,
        )
        + struct.pack(
            "<Q",
            len(metadata_bytes),
        )
    )

    payload = (
        header
        + metadata_bytes
        + y_string
        + z_string
    )

    return payload


def parse_container(path):
    """
    Read a CAI1 compressed file.
    """

    with open(path, "rb") as f:
        data = f.read()

    offset = 0

    # --------------------------------------------------------
    # Magic
    # --------------------------------------------------------

    magic = data[
        offset:
        offset + 4
    ]

    offset += 4

    if magic != MAGIC:
        raise ValueError(
            "Invalid CAI1 file."
        )

    # --------------------------------------------------------
    # Version
    # --------------------------------------------------------

    version = struct.unpack_from(
        "<B",
        data,
        offset,
    )[0]

    offset += 1

    if version != VERSION:
        raise ValueError(
            f"Unsupported version: {version}"
        )

    # --------------------------------------------------------
    # Metadata length
    # --------------------------------------------------------

    metadata_size = struct.unpack_from(
        "<Q",
        data,
        offset,
    )[0]

    offset += 8

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata_bytes = data[
        offset:
        offset + metadata_size
    ]

    offset += metadata_size

    metadata = pickle.loads(
        metadata_bytes
    )

    # --------------------------------------------------------
    # Entropy streams
    # --------------------------------------------------------

    y_size = metadata["y_size"]
    z_size = metadata["z_size"]

    y_string = data[
        offset:
        offset + y_size
    ]

    offset += y_size

    z_string = data[
        offset:
        offset + z_size
    ]

    strings = [
        [y_string],
        [z_string],
    ]

    return metadata, strings


# ============================================================
# Compression
# ============================================================

def compress_to_target_size(
    image_path,
    output_path,
    target_kb=200,
):
    """
    Compress an image using its ORIGINAL dimensions.

    There is NO resizing.

    The only spatial modification is temporary internal
    padding required by the neural network.

    The reconstructed image is cropped back to the exact
    original dimensions.

    target_kb:
        Maximum allowed compressed container size.
    """

    target_bytes = int(
        target_kb * 1024
    )

    print()
    print("=" * 70)
    print("CompressAI ORIGINAL-DIMENSION compression")
    print("=" * 70)

    print(
        f"Input : {image_path}"
    )

    print(
        f"Output: {output_path}"
    )

    print(
        f"Target: {target_bytes:,} bytes "
        f"({target_kb:.2f} KB)"
    )

    print()

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    img = Image.open(
        image_path
    ).convert("RGB")

    orig_w, orig_h = img.size

    print(
        f"Original dimensions: "
        f"{orig_w} x {orig_h}"
    )

    # --------------------------------------------------------
    # Convert to tensor
    # --------------------------------------------------------

    x = T.ToTensor()(
        img
    ).unsqueeze(0)

    # --------------------------------------------------------
    # Internal padding
    # --------------------------------------------------------

    x_padded, padded_h, padded_w = (
        pad_for_network(x)
    )

    print(
        f"Internal dimensions: "
        f"{padded_w} x {padded_h}"
    )

    if (
        padded_w != orig_w
        or padded_h != orig_h
    ):
        print(
            "Internal padding applied "
            "(removed after decoding)."
        )
    else:
        print(
            "No internal padding required."
        )

    # --------------------------------------------------------
    # Search quality levels
    # --------------------------------------------------------

    best = None

    for quality in range(1, 9):

        print()
        print(
            f"Testing quality {quality}..."
        )

        model = None

        try:

            model = load_model(
                quality=quality,
                device="cpu",
            )

            # ------------------------------------------------
            # Encode
            # ------------------------------------------------

            with torch.inference_mode():

                encoded = model.compress(
                    x_padded
                )

            strings = encoded["strings"]

            latent_shape = encoded["shape"]

            # ------------------------------------------------
            # Create complete file
            # ------------------------------------------------

            payload = create_container(
                quality=quality,

                orig_h=orig_h,
                orig_w=orig_w,

                padded_h=padded_h,
                padded_w=padded_w,

                latent_shape=latent_shape,

                strings=strings,
            )

            size_bytes = len(payload)

            # ------------------------------------------------
            # Bits per original pixel
            # ------------------------------------------------

            bpp = (
                size_bytes * 8.0
                / (orig_w * orig_h)
            )

            print(
                f"  Size: {size_bytes:,} bytes "
                f"({size_bytes / 1024:.2f} KB)"
            )

            print(
                f"  BPP : {bpp:.6f}"
            )

            # ------------------------------------------------
            # Target test
            # ------------------------------------------------

            if size_bytes <= target_bytes:

                best = {
                    "quality": quality,
                    "payload": payload,
                    "size": size_bytes,
                    "bpp": bpp,
                }

                print(
                    "  -> FITS TARGET"
                )

            else:

                print(
                    "  -> TOO LARGE"
                )

                # Higher quality normally means higher rate.
                break

        finally:

            if model is not None:
                del model

            gc.collect()

    # --------------------------------------------------------
    # No quality level fits
    # --------------------------------------------------------

    if best is None:

        print()
        print("=" * 70)
        print("COMPRESSION FAILED")
        print("=" * 70)

        print(
            f"No available quality level "
            f"can encode the original "
            f"{orig_w} x {orig_h} image "
            f"within {target_kb} KB."
        )

        print()
        print(
            "IMPORTANT:"
        )

        print(
            "The image was NOT resized."
        )

        print(
            "The image dimensions were preserved."
        )

        print(
            "The limitation is the available "
            "CompressAI quality levels."
        )

        raise RuntimeError(
            "Target size cannot be achieved "
            "at the original resolution "
            "with quality levels 1-8."
        )

    # --------------------------------------------------------
    # Save compressed file
    # --------------------------------------------------------

    with open(
        output_path,
        "wb",
    ) as f:

        f.write(
            best["payload"]
        )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("COMPRESSION SUCCESS")
    print("=" * 70)

    print(
        f"Original dimensions : "
        f"{orig_w} x {orig_h}"
    )

    print(
        f"Internal dimensions : "
        f"{padded_w} x {padded_h}"
    )

    print(
        f"Quality             : "
        f"{best['quality']}"
    )

    print(
        f"Compressed size     : "
        f"{best['size']:,} bytes"
    )

    print(
        f"Compressed size     : "
        f"{best['size'] / 1024:.2f} KB"
    )

    print(
        f"Target              : "
        f"{target_bytes:,} bytes"
    )

    print(
        f"BPP                 : "
        f"{best['bpp']:.6f}"
    )

    print(
        f"Saved               : "
        f"{output_path}"
    )

    print("=" * 70)


# ============================================================
# Decompression
# ============================================================

def decompress_file(
    input_path,
    output_path,
):
    """
    Decompress a CAI1 file.

    The resulting image has exactly the original dimensions
    stored in the compressed file.
    """

    print()
    print("=" * 70)
    print("CompressAI decompression")
    print("=" * 70)

    print(
        f"Input : {input_path}"
    )

    print(
        f"Output: {output_path}"
    )

    print()

    # --------------------------------------------------------
    # Read container
    # --------------------------------------------------------

    metadata, strings = parse_container(
        input_path
    )

    quality = metadata["quality"]

    orig_h = metadata["orig_h"]
    orig_w = metadata["orig_w"]

    padded_h = metadata["padded_h"]
    padded_w = metadata["padded_w"]

    latent_shape = tuple(
        metadata["latent_shape"]
    )

    print(
        f"Quality             : {quality}"
    )

    print(
        f"Original dimensions : "
        f"{orig_w} x {orig_h}"
    )

    print(
        f"Internal dimensions : "
        f"{padded_w} x {padded_h}"
    )

    print(
        f"Latent shape        : "
        f"{latent_shape}"
    )

    # --------------------------------------------------------
    # Load matching model
    # --------------------------------------------------------

    model = load_model(
        quality=quality,
        device="cpu",
    )

    try:

        # ----------------------------------------------------
        # Decode
        # ----------------------------------------------------

        with torch.inference_mode():

            decoded = model.decompress(
                strings,
                latent_shape,
            )

        x_hat = decoded["x_hat"]

        decoded_h = x_hat.shape[-2]
        decoded_w = x_hat.shape[-1]

        print(
            f"Decoded dimensions  : "
            f"{decoded_w} x {decoded_h}"
        )

        # ----------------------------------------------------
        # Remove internal padding
        # ----------------------------------------------------

        x_hat = x_hat[
            :,
            :,
            :orig_h,
            :orig_w,
        ]

        # ----------------------------------------------------
        # Verify dimensions
        # ----------------------------------------------------

        final_h = x_hat.shape[-2]
        final_w = x_hat.shape[-1]

        if (
            final_h != orig_h
            or final_w != orig_w
        ):
            raise RuntimeError(
                "Decoded image dimensions do not "
                "match original dimensions."
            )

        print(
            f"Final dimensions    : "
            f"{final_w} x {final_h}"
        )

        # ----------------------------------------------------
        # Convert to PIL
        # ----------------------------------------------------

        output_img = tensor_to_pil(
            x_hat
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        output_img.save(
            output_path
        )

        print()
        print("=" * 70)
        print("DECOMPRESSION SUCCESS")
        print("=" * 70)

        print(
            f"Output              : "
            f"{output_path}"
        )

        print(
            f"Dimensions           : "
            f"{output_img.width} x "
            f"{output_img.height}"
        )

        print("=" * 70)

    finally:

        del model
        gc.collect()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Your input
    # --------------------------------------------------------

    input_file = (
        "/home/chinasa/python_projects/"
        "fivek_prnu_split/"
        "query_set_compressed_3mb/jpeg/"
        "Canon_Canon_EOS_5D_SN_620302732/"
        "Canon_Canon_EOS_5D_SN_620302732_"
        "a0004-jmac_MG_1384.jpeg"
    )

    compressed_file = (
        "compressed_target.bin"
    )

    reconstructed_file = (
        "reconstructed.png"
    )

    # --------------------------------------------------------
    # COMPRESS
    # --------------------------------------------------------

    compress_to_target_size(
        image_path=input_file,
        output_path=compressed_file,
        target_kb=200,
    )

    # --------------------------------------------------------
    # DECOMPRESS
    # --------------------------------------------------------

    # Uncomment to test immediately.

    # decompress_file(
    #     input_path=compressed_file,
    #     output_path=reconstructed_file,
    # )
