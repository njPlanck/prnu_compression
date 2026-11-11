
# How does Compression degrade PRNU-based Source Camera Identification?
With the continued growth and adoption of digital media, the need to attribute digital images to
their sources after they have been acquired remains a fundamental problem in multimedia forensics.
So far, many studies have identified that Source Camera Identification (SCI) using 
Photo Response Non-Uniformity (PRNU) degrades under compression. These studies mostly assess baseline 
JPEG and its impact at different Quality Factors (QFs). However, for a given QF, different images 
compress to different sizes. This is problematic for evaluating the effect of compression in the real-world setting,
as in most networks and storage systems the target size is often the fixed parameter while the QF varies per image. 
In this paper we  evaluate the SCI performace under compression at different target bitrates,
simulating real-world strict bandwidth and storage constraints. 
We also extend this evaluation to the more recent compression schemes like JPEG 2000, JPEG XR and JPEG XL.
This is to provides a codec recommendation for severe image degradation at very low bitrates that mostly 
preserve the integrity of the forensic signal required for SCI. For the experiment, the Adobe Fivek dataset 
was used with 17 cameras selected between three different camera brands (NIKON, Canon and Sony).

The comprehensive report is available in the [paper PDF](paper/IEEE-conference-template-062824.pdf).

## How to run

The scripts use the Adobe FiveK DNG images and expect them to be split into camera-named
`reference_set` and `query_set` folders. The dataset is not included in this repository.

1. From the repository root, create and activate a Python environment, then install the
   listed and additional dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   python -m pip install -r requirements.txt
   python -m pip install rawpy imagecodecs piq jxlpy
   ```

   JPEG XL compression also requires the `cjxl` and `djxl` command-line tools to be
   installed and available on `PATH`.

2. Update the dataset and output paths in the `__main__` sections of
   `src_code/reference.py`, `src_code/compression.py`, and `src_code/query.py` to match
   your local setup. These scripts currently contain machine-specific paths. In
   `reference.py`, use `filter_type='wavelet'` and an output directory matching
   `query.py`'s `FINGERPRINT_DIR` to run the default evaluation.

3. Build the camera fingerprints, compress the query images to the target size, and
   evaluate source-camera identification from the repository root:

   ```bash
   python src_code/reference.py
   python src_code/compression.py
   python src_code/query.py
   ```

   The evaluation script defaults to the JPEG 2000 (`jp2`) compressed images and the
   Wavelet filter. It writes evaluation spreadsheets under `results/`. To evaluate a
   different codec or filter, update the corresponding settings in `src_code/query.py`.

### JPEG-AI reference compression

The optional JPEG-AI pipeline is in the included
[reference software](src_code/jpeg-ai-reference-software/README.md). Follow its README
to configure the `jpeg_ai_vm` environment, build the required libraries, and obtain
the model files before running the wrapper. From the reference software directory:

```bash
conda activate jpeg_ai_vm
python -m pip install rawpy piq
python jpeg_ai_compression.py
```

The defaults in `jpeg_ai_compression.py` are set to the FiveK `query_set`
directories used in this workspace; edit `DEFAULT_INPUT_DIR`,
`DEFAULT_OUTPUT_DIR`, and `DEFAULT_TARGET_SIZE_KIB` near the top of that file to
run directly from an IDE or terminal without arguments. Positional paths and
`--target-size-kb` override those defaults, for example:

```bash
python jpeg_ai_compression.py /path/to/camera_images /path/to/jpeg_ai_outputs \
  --target-size-kb 50 100 200
```

`--target-size-kb` accepts one or more per-image target sizes in KiB (1024 bytes).
The script demosaics camera RAW inputs with `rawpy`, stages every source image as
RGB PNG for JPEG-AI (whose reference encoder accepts PNG/YUV). Reconstructed PNGs
are created temporarily for quality metrics and removed after each image. The
persistent per-image outputs are JPEG-AI bitstreams; per-camera metrics CSVs are
also written. CSV reports include requested size, actual bitstream size,
target/actual BPP, and image quality metrics. JPEG-AI rate control targets BPP
derived from each image's pixel count, so actual bitstream sizes can differ from
the requested sizes due to codec rate-control precision and stream overhead.
Input subfolders are traversed recursively and retained in the output; the first
subfolder level is treated as the camera name and receives its own metrics CSV.
The output directory must be outside the input tree to avoid reprocessing
generated files.

For a fixed rate rather than a per-image file-size goal, omit `--target-size-kb` and
use `--set-target-bpp N`, where `N` is the target rate in hundredths of a bit per
pixel (for example, `12` means `0.12` bpp).

### Still-image codec compression

`src_code/compress_still.py` follows the folder traversal, per-camera reports, and
per-image target-size workflow of `src_code/compression.py`, using the still-image
codecs AVIF, HEIC, BPG, and WebP. It develops DNG files to RGB with `rawpy`,
keeps image dimensions unchanged, and binary-searches codec quality for the
closest tested file size to each requested target. It saves that closest result
even if codec quality steps mean the target cannot be matched exactly; the
reported size delta shows how close it got. Successful outputs are stored by
target size, codec, and camera folder, with per-camera CSV reports alongside
them.
CSV progress is saved after every completed image/codec pair. Ctrl+C stops the
current encode and preserves completed outputs and metrics. Use `--resume` when
restarting to skip successful outputs within the configured tolerance (12 KiB
by default).

Run it from the repository root:

```bash
python src_code/compress_still.py \
  --input-dir /path/to/query_set \
  --output-dir /path/to/query_set_still_compressed \
  --target-size-kb 100 250 500 \
  --target-size-tolerance-kb 12 \
  --resume
```

By default, the script tries all four codecs and uses editable input/output
defaults near the top of the file. AVIF requires an ImageMagick build with AVIF
write support. WebP uses Pillow's WebP encoder and requires Pillow to be built
with WebP support; quality search uses Pillow's lossy quality setting rather
than ImageMagick's WebP quality option. Its search checks encoder effort
methods 0 through 6 as well as quality and records the selected method in the
per-camera CSV.

On Ubuntu, HEIC encoding requires the command-line utilities and HEVC encoder
plugin:

```bash
sudo apt install libheif-examples libheif-plugin-x265
```

The script checks that `heif-enc` actually lists an HEVC encoder before starting
HEIC encoding, so a decoder-only libheif installation is not mistaken for full
HEIC support.

BPG is not packaged in Ubuntu's repositories. Build its tools from the official
[BPG source archive](https://bellard.org/bpg/libbpg-0.9.8.tar.gz):

```bash
curl -fLO https://bellard.org/bpg/libbpg-0.9.8.tar.gz
tar -xzf libbpg-0.9.8.tar.gz
cd libbpg-0.9.8
make -j2 USE_X265= USE_JCTVC=y USE_BPGVIEW=
install -d ~/.local/bin
install -m 755 bpgenc bpgdec ~/.local/bin/
```

The JCTVC BPG backend is slower than x265, but does not require CMake or the
bundled legacy x265 build. `~/.local/bin` is detected by the script even when it
is not on `PATH`; the script also checks `~/.local/lib/libheif/plugins` for a
user-local HEIC encoder plugin. HEIC and BPG outputs are decoded to verify them.
Unavailable codecs are reported in the CSV rather than silently substituted.
Image-quality metrics use the project's `torch` and `piq` dependencies.
AVIF uses a faster encoder speed preset. BPG's quantizer runs in the opposite
direction from the quality controls used by the other codecs, and the search
accounts for that.
