import os
import numpy as np
import rawpy
from tqdm import tqdm
import pywt
from scipy.ndimage import uniform_filter
import numpy as np
from numba import njit, prange

# =====================================================================
# 1. LOAD FULL-RESOLUTION RGB GREEN CHANNEL
# =====================================================================

def load_rgb_green(dng_path):
    """
    Develops DNG into RGB, extracts the full-resolution Green channel, 
    and normalizes to [0, 1].
    """
    with rawpy.imread(dng_path) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,
            half_size=False,
            no_auto_bright=True,
            output_bps=16,
            user_flip=0  # <--- CRITICAL FIX: Ignores EXIF. Forces raw sensor alignment!
        )

    # RGB -> Green channel normalized to float32 [0.0, 1.0]
    green = (rgb[:, :, 1] / 65535.0).astype(np.float32)
    return np.clip(green, 0.0, 1.0)


# =====================================================================
# 2. DENOISING FILTERS
# =====================================================================

@njit(parallel=True, fastmath=True)
def sigma_filter(img, window_size=3, sigma=0.03):
    """
    Highly optimized Lee-style Sigma filter for full-res PRNU extraction.
    Uses multi-threading and dynamic boundary checks to avoid padding overhead.
    """
    pad = window_size // 2
    h, w = img.shape
    out = np.zeros_like(img)
    
    # prange enables multi-threading across CPU cores
    for y in prange(h):
        for x in range(w):
            sum_val = 0.0
            count = 0.0
            center = img[y, x]
            
            # Dynamic boundaries to handle image edges without a black border
            y_min = max(0, y - pad)
            y_max = min(h, y + pad + 1)
            x_min = max(0, x - pad)
            x_max = min(w, x + pad + 1)
            
            for ny in range(y_min, y_max):
                for nx in range(x_min, x_max):
                    val = img[ny, nx]
                    if abs(val - center) <= sigma:
                        sum_val += val
                        count += 1.0
            
            if count > 0.0:
                out[y, x] = sum_val / count
            else:
                out[y, x] = center
                
    return out

def wavelet_filter(img, level=4, wavelet='db4', window_size=3):
    """Mihcak-style Wiener-like wavelet denoising."""
    coeffs = pywt.wavedec2(img, wavelet, level=level)
    denoised_coeffs = [coeffs[0]]

    cD1 = coeffs[-1][2]
    sigma_0 = np.median(np.abs(cD1)) / 0.6745
    noise_var = sigma_0 ** 2

    for i in range(1, len(coeffs)):
        cH, cV, cD = coeffs[i]
        cleaned_subbands = []

        for subband in (cH, cV, cD):
            local_var = uniform_filter(subband ** 2, size=window_size)
            shrinkage = np.maximum(0, local_var - noise_var) / np.maximum(local_var, 1e-8)
            cleaned_subbands.append(subband * shrinkage)

        denoised_coeffs.append(tuple(cleaned_subbands))

    clean_img = pywt.waverec2(denoised_coeffs, wavelet)
    return clean_img[:img.shape[0], :img.shape[1]].astype(np.float32)


def denoise_channel(img, filter_type='wavelet'):
    """Dispatch to requested denoising algorithm."""
    filter_type = filter_type.lower()
    if filter_type == 'sigma':
        return sigma_filter(img)
    elif filter_type == 'wavelet':
        return wavelet_filter(img)
    else:
        raise ValueError(f"Unsupported filter_type: {filter_type}. Use 'wavelet' or 'sigma'.")


# =====================================================================
# 3. ZERO-MEAN & LINEAR NOISE REMOVAL
# =====================================================================

def remove_linear_patterns(W_c, valid_mask, passes=2):
    """Removes global, row, and column means."""
    for _ in range(passes):
        valid_pixel_count = np.sum(valid_mask)
        if valid_pixel_count > 0:
            mean_valid = np.sum(W_c) / valid_pixel_count
            W_c = np.where(valid_mask, W_c - mean_valid, 0.0)

        row_counts = np.sum(valid_mask, axis=1, keepdims=True)
        row_counts[row_counts == 0] = 1
        row_means = np.sum(W_c, axis=1, keepdims=True) / row_counts
        W_c = np.where(valid_mask, W_c - row_means, 0.0)

        col_counts = np.sum(valid_mask, axis=0, keepdims=True)
        col_counts[col_counts == 0] = 1
        col_means = np.sum(W_c, axis=0, keepdims=True) / col_counts
        W_c = np.where(valid_mask, W_c - col_means, 0.0)

    return W_c


# =====================================================================
# 4. MASTER FINGERPRINT BUILDER
# =====================================================================

def build_master_fingerprints(
    reference_dir,
    output_fingerprint_dir,
    filter_type='wavelet'
):
    if not os.path.exists(reference_dir):
        print(f"Reference directory not found: {reference_dir}")
        return

    os.makedirs(output_fingerprint_dir, exist_ok=True)

    body_folders = [
        f for f in sorted(os.listdir(reference_dir))
        if os.path.isdir(os.path.join(reference_dir, f))
    ]

    print(f"Building FULL-RESOLUTION RGB GREEN fingerprints using {filter_type.upper()}")
    print("=" * 80)

    for body_name in body_folders:
        body_path = os.path.join(reference_dir, body_name)
        dng_files = [
            os.path.join(body_path, f)
            for f in sorted(os.listdir(body_path))
            if f.lower().endswith('.dng')
        ]

        if not dng_files:
            continue

        print(f"\nCamera: {body_name} ({len(dng_files)} images)")

        num_k = None
        den_k = None
        cam_target_shape = None
        valid_images = 0
        skipped_images = 0
        shape_counts = {}

        for dng_path in tqdm(dng_files, desc=f" Processing {body_name}", leave=False):
            filename = os.path.basename(dng_path)

            try:
                green_img = load_rgb_green(dng_path)

                # Initialize camera-specific native shape on the first valid image
                if num_k is None:
                    cam_target_shape = green_img.shape
                    num_k = np.zeros(cam_target_shape, dtype=np.float64)
                    den_k = np.zeros(cam_target_shape, dtype=np.float64)

                # Enforce shape consistency across all images for this camera body
                if green_img.shape != cam_target_shape:
                    raise ValueError(
                        f"Shape mismatch: got {green_img.shape}, expected camera lock shape {cam_target_shape}"
                    )

                shape_counts[green_img.shape] = shape_counts.get(green_img.shape, 0) + 1

                clean_green = denoise_channel(green_img, filter_type=filter_type)
                W_green = green_img - clean_green

                valid_mask = (green_img > 0.02) & (green_img < 0.98)
                W_c = np.where(valid_mask, W_green, 0.0)
                W_c = remove_linear_patterns(W_c, valid_mask, passes=2)

                I_c = np.where(valid_mask, green_img, 0.0)

                num_k += W_c * I_c
                den_k += I_c ** 2
                valid_images += 1

            except Exception as e:
                skipped_images += 1
                tqdm.write(f"  [!] Error processing {filename}: {e}")

        print(f"    Shapes used: {shape_counts}")
        print(f"    Compatible images: {valid_images}/{len(dng_files)}")
        print(f"    Skipped images: {skipped_images}")

        if valid_images == 0:
            print(f" [!] No compatible images for {body_name}; fingerprint was not created.")
            continue

        K_hat = (num_k / (den_k + 1e-8)).astype(np.float32)

        out_file = os.path.join(
            output_fingerprint_dir,
            f"{body_name}_{filter_type}_rgb_green_fingerprint.npz"
        )

        np.savez_compressed(out_file, fingerprint=K_hat)
        print(f" [✓] Saved -> {out_file}")
        print(f"     Fingerprint shape: {K_hat.shape}")


# =====================================================================
# MAIN ENTRYPOINT
# =====================================================================

if __name__ == "__main__":
    ref_dir = "/home/chinasa/python_projects/fivek_prnu_split/reference_set"
    out_dir = "/home/chinasa/python_projects/prnu_compression/fivek_rgb_green_sigma_fingerprints"

    build_master_fingerprints(
        reference_dir=ref_dir,
        output_fingerprint_dir=out_dir,
        filter_type='sigma'
    )