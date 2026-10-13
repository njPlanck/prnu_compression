import os
import numpy as np
import rawpy
import pywt
import bm3d
import time
from scipy.fftpack import fft2, ifft2, fftshift
from scipy.ndimage import uniform_filter

# =====================================================================
# 1. DENOISING FILTERS (WAVELET OR BM3D)
# =====================================================================

def wavelet_filter(img, level=4, wavelet='db4', window_size=3):
    """Exact PRNU Wavelet Filter based on Lukas/Fridrich/Mihcak."""
    coeffs = pywt.wavedec2(img, wavelet, level=level)
    denoised_coeffs = [coeffs[0]]
    
    cH1, cV1, cD1 = coeffs[-1]
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
    return clean_img[:img.shape[0], :img.shape[1]]

def bm3d_filter(img, sigma_psd=0.03):
    """
    Optimized BM3D Filter wrapper using the official C-accelerated package.
    ALL_STAGES ensures maximum collaborative filtering performance for PRNU.
    """
    clean_img = bm3d.bm3d(img, sigma_psd=sigma_psd, stage_arg=bm3d.BM3DStages.ALL_STAGES)
    return clean_img

# =====================================================================
# 2. PCE CALCULATOR
# =====================================================================

def compute_pce(X, Y, search_radius=60):
    X = X - np.mean(X)
    Y = Y - np.mean(Y)

    F_X = fft2(X)
    F_Y = fft2(Y)
    
    R = F_X * np.conj(F_Y)
    r = np.real(ifft2(R))
    r = fftshift(r)

    h, w = r.shape
    center_y, center_x = h // 2, w // 2

    y_min = max(0, center_y - search_radius)
    y_max = min(h, center_y + search_radius + 1)
    x_min = max(0, center_x - search_radius)
    x_max = min(w, center_x + search_radius + 1)

    search_area = r[y_min:y_max, x_min:x_max]

    peak_y_local, peak_x_local = np.unravel_index(np.argmax(search_area), search_area.shape)
    peak_value = search_area[peak_y_local, peak_x_local]

    peak_y = y_min + peak_y_local
    peak_x = x_min + peak_x_local

    r_copy = r.copy()
    p_rad = 5
    py_min = max(0, peak_y - p_rad)
    py_max = min(h, peak_y + p_rad + 1)
    px_min = max(0, peak_x - p_rad)
    px_max = min(w, peak_x + p_rad + 1)

    r_copy[py_min:py_max, px_min:px_max] = 0

    valid_area_size = r.size - ((py_max - py_min) * (px_max - px_min))
    energy = np.sum(r_copy ** 2) / valid_area_size

    pce = (peak_value ** 2) / (energy + 1e-8)
    if peak_value < 0:
        pce = -pce

    shift = (peak_y - center_y, peak_x - center_x)
    return pce, shift

# =====================================================================
# 3. FAST DATABASE SEARCH PIPELINE
# =====================================================================

def extract_test_noise(dng_path, filter_type='bm3d'):
    """Extracts contiguous G1 green channel noise using the chosen filter."""
    with rawpy.imread(dng_path) as raw:
        bayer = raw.raw_image_visible.astype(np.float32)
        colors = raw.raw_colors
        
        # Crop alignment
        if colors.shape != bayer.shape:
            top, left = raw.sizes.top_margin, raw.sizes.left_margin
            h, w = bayer.shape
            colors = colors[top:top+h, left:left+w] if top+h <= colors.shape[0] else colors[:h, :w]

        # Black/White level normalization
        white = float(raw.white_level)
        black_levels = raw.black_level_per_channel
        black_g = (float(black_levels[1]) + float(black_levels[3])) / 2.0
        bayer = np.clip((bayer - black_g) / (white - black_g), 0, 1).astype(np.float32)

        # Find where G1 sits in the 2x2 Bayer block
        g1_coords = np.argwhere(colors[:2, :2] == 1)
        if len(g1_coords) == 0:
            g1_coords = np.argwhere(colors[:2, :2] == 3)
            
        r_off, c_off = g1_coords[0]

        # Extract contiguous G1 sub-matrix
        g1_sub = bayer[r_off::2, c_off::2]

        # Denoise using selected filter type
        if filter_type.lower() == 'bm3d':
            clean_g1 = bm3d_filter(g1_sub, sigma_psd=0.03)
        else:
            clean_g1 = wavelet_filter(g1_sub)
            
        W_g1 = g1_sub - clean_g1
        
        # Mask out extreme highlights/shadows
        valid_mask = (g1_sub > 0.02) & (g1_sub < 0.98)
        W_g1 = np.where(valid_mask, W_g1, 0.0)

        # Eliminate Linear Pattern Noise (Row/Column zero-meaning)
        valid_pixel_count = np.sum(valid_mask)
        if valid_pixel_count > 0:
            mean_valid = np.sum(W_g1) / valid_pixel_count
            W_g1 = np.where(valid_mask, W_g1 - mean_valid, 0.0)
            
            row_counts = np.sum(valid_mask, axis=1, keepdims=True)
            row_counts[row_counts == 0] = 1
            row_means = np.sum(W_g1, axis=1, keepdims=True) / row_counts
            W_g1 = np.where(valid_mask, W_g1 - row_means, 0.0)

            col_counts = np.sum(valid_mask, axis=0, keepdims=True)
            col_counts[col_counts == 0] = 1
            col_means = np.sum(W_g1, axis=0, keepdims=True) / col_counts
            W_g1 = np.where(valid_mask, W_g1 - col_means, 0.0)

        return W_g1

def find_best_match(test_image_path, fingerprint_dir, filter_type='bm3d', search_radius=60):
    print(f"\nTarget Image: {os.path.basename(test_image_path)}")
    print(f"Extracting noise residual using {filter_type.upper()} filter...")
    
    start_time = time.time()
    W_test = extract_test_noise(test_image_path, filter_type=filter_type)
    print(f"Noise extracted in {time.time() - start_time:.2f} seconds.\n")

    best_pce = -float('inf')
    best_match = None
    best_shift = (0, 0)
    
    print(f"{'Fingerprint':<50} | {'PCE Score':<10}")
    print("-" * 65)

    for fp_file in sorted(os.listdir(fingerprint_dir)):
        if not fp_file.endswith('.npz'):
            continue
            
        fp_path = os.path.join(fingerprint_dir, fp_file)
        data = np.load(fp_path)
        K_hat = data['fingerprint']

        min_h = min(K_hat.shape[0], W_test.shape[0])
        min_w = min(K_hat.shape[1], W_test.shape[1])
        
        K_hat_crop = K_hat[:min_h, :min_w]
        W_crop = W_test[:min_h, :min_w]

        pce, shift = compute_pce(W_crop, K_hat_crop, search_radius=search_radius)
        
        display_name = fp_file[:47] + "..." if len(fp_file) > 50 else fp_file
        print(f"{display_name:<50} | {pce:>8.2f}")
        
        if pce > best_pce:
            best_pce = pce
            best_match = fp_file
            best_shift = shift

    print("\n" + "=" * 65)
    print("FINAL VERDICT")
    print("=" * 65)
    
    if best_match is None:
        print("No fingerprints found in the directory.")
        return

    print(f"Top Match:    {best_match}")
    print(f"PCE Score:    {best_pce:.2f}")
    print(f"ISP Shift:    y={best_shift[0]}, x={best_shift[1]} (pixels)")
    
    if best_pce > 60:
        print("Confidence:   [✓] HIGH (Definite Match)")
    elif best_pce > 45:
        print("Confidence:   [?] MEDIUM (Possible Match)")
    else:
        print("Confidence:   [x] LOW (No match found in database)")
    print("=" * 65)

# =====================================================================
# EXECUTION
# =====================================================================

if __name__ == "__main__":
    FINGERPRINT_DIR = "/home/chinasa/python_projects/prnu_compression/fivek_pure_bayer_wavelet_fingerprints"
    TEST_IMAGE = "/home/chinasa/python_projects/fivek_prnu_split/query_set/Canon_Canon_EOS_20D_SN_620312693/a0010-jmac_MG_4807.dng" 
    
    # Toggle between 'bm3d' and 'wavelet' here
    find_best_match(TEST_IMAGE, FINGERPRINT_DIR, filter_type='wavelet', search_radius=60)