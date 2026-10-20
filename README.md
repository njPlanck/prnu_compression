
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
