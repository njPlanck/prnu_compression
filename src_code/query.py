import os
import time
import gc
import multiprocessing
import concurrent.futures
import numpy as np
import rawpy
import pandas as pd
import imagecodecs
import pywt
from scipy.fftpack import fft2, ifft2, fftshift
import cv2
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    roc_curve,
    roc_auc_score,
)
from reference import sigma_filter, wavelet_filter, denoise_channel

# Global worker storage for fingerprints (loaded once per process to save RAM & overhead)
_WORKER_FINGERPRINTS = {}

def init_worker(fingerprint_dir):
    """Initializes each worker process by loading fingerprints into memory once."""
    global _WORKER_FINGERPRINTS
    fp_files = sorted([f for f in os.listdir(fingerprint_dir) if f.endswith('.npz')])
    for fp_file in fp_files:
        fp_name = os.path.splitext(fp_file)[0]
        fp_path = os.path.join(fingerprint_dir, fp_file)
        data = np.load(fp_path)
        key = 'fingerprint' if 'fingerprint' in data else list(data.keys())[0]
        _WORKER_FINGERPRINTS[fp_name] = data[key]

# =====================================================================
# 2. IMAGE LOADING & PATCH NOISE EXTRACTION
# =====================================================================

def load_image(img_path):
    ext = os.path.splitext(img_path)[1].lower()
    with open(img_path, 'rb') as f:
        data = f.read()

    if ext == '.jxl':
        img = imagecodecs.jpegxl_decode(data)
    elif ext in ['.jxr', '.hdp', '.wdp']:
        img = imagecodecs.jpegxr_decode(data)
    elif ext in ['.jp2', '.j2k']:
        img = imagecodecs.jpeg2k_decode(data)
    else:
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if img is None:
        raise ValueError(f"Could not load image: {img_path}")
    return np.asarray(img)

def extract_test_patch_noise(img_path, patch_size=(512, 512), filter_type='wavelet'):
    ext = os.path.splitext(img_path)[1].lower()

    if ext == '.dng':
        with rawpy.imread(img_path) as raw:
            img = raw.postprocess(
                use_camera_wb=True, half_size=False, no_auto_bright=True,
                output_bps=16, user_flip=0
            )
    else:
        img = load_image(img_path)

    if img.dtype == np.uint16:
        img_float = img.astype(np.float32) / 65535.0
    elif img.dtype == np.uint8:
        img_float = img.astype(np.float32) / 255.0
    else:
        img_float = img.astype(np.float32)
        if np.max(img_float) > 1.0:
            img_float /= (65535.0 if np.max(img_float) > 255.0 else 255.0)

    if img_float.ndim == 3 and img_float.shape[2] >= 3:
        processed_channel = img_float[:, :, 1]
    elif img_float.ndim == 2:
        processed_channel = img_float
    else:
        del img, img_float
        raise ValueError(f"Unexpected image shape")

    del img, img_float
    processed_channel = np.clip(processed_channel, 0.0, 1.0).astype(np.float32)

    th, tw = processed_channel.shape
    ph, pw = patch_size
    ph = min(ph, th)
    pw = min(pw, tw)
    
    y1 = max(0, (th - ph) // 2)
    x1 = max(0, (tw - pw) // 2)
    patch_channel = processed_channel[y1:y1+ph, x1:x1+pw]
    del processed_channel

    clean_patch = denoise_channel(patch_channel, filter_type=filter_type)
    W_patch = patch_channel - clean_patch
    del clean_patch

    valid_mask = (patch_channel > 0.02) & (patch_channel < 0.98)
    W_patch = np.where(valid_mask, W_patch, 0.0)

    for _ in range(2):
        valid_pixel_count = np.sum(valid_mask)
        if valid_pixel_count > 0:
            mean_valid = np.sum(W_patch) / valid_pixel_count
            W_patch = np.where(valid_mask, W_patch - mean_valid, 0.0)

            row_counts = np.sum(valid_mask, axis=1, keepdims=True)
            row_counts[row_counts == 0] = 1
            row_means = np.sum(W_patch, axis=1, keepdims=True) / row_counts
            W_patch = np.where(valid_mask, W_patch - row_means, 0.0)

            col_counts = np.sum(valid_mask, axis=0, keepdims=True)
            col_counts[col_counts == 0] = 1
            col_means = np.sum(W_patch, axis=0, keepdims=True) / col_counts
            W_patch = np.where(valid_mask, W_patch - col_means, 0.0)

    return W_patch

# =====================================================================
# 3. PCE CALCULATOR
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

    r_copy[py_min:py_max, px_min:px_max] = 0.0
    valid_area_size = r.size - ((py_max - py_min) * (px_max - px_min))
    energy = np.sum(r_copy ** 2) / max(1, valid_area_size)

    pce = (peak_value ** 2) / (energy + 1e-12)
    if peak_value < 0:
        pce = -pce

    return float(pce), (peak_y - center_y, peak_x - center_x)

def compute_patch_pce(W_patch, K_hat, search_radius=60):
    ph, pw = W_patch.shape
    kh, kw = K_hat.shape
    
    if kh >= ph and kw >= pw:
        y1 = (kh - ph) // 2
        x1 = (kw - pw) // 2
        K_patch = K_hat[y1:y1+ph, x1:x1+pw]
    else:
        y1 = max(0, (kh - ph) // 2)
        x1 = max(0, (kw - pw) // 2)
        y2 = min(kh, y1 + ph)
        x2 = min(kw, x1 + pw)
        K_patch = K_hat[y1:y2, x1:x2]
        if K_patch.shape != W_patch.shape:
            K_patch = cv2.resize(K_patch, (pw, ph), interpolation=cv2.INTER_CUBIC)

    return compute_pce(W_patch, K_patch, search_radius=search_radius)

# =====================================================================
# 4. EVALUATION WORKER
# =====================================================================

def process_single_image(args):
    img_path, true_label, filter_type, search_radius, patch_size = args
    img_name = os.path.basename(img_path)
    
    start_time = time.time()
    best_pce = -float('inf')
    best_match_label = None
    best_shift = (0, 0)
    image_pces = {}
    roc_pairs = []
    
    matching_fp_key = next((k for k in _WORKER_FINGERPRINTS.keys() if k.startswith(true_label)), None)
    
    try:
        W_patch = extract_test_patch_noise(img_path, patch_size=patch_size, filter_type=filter_type)
    except Exception as e:
        return None, f"Error extracting noise from {img_name}: {e}"

    for fp_name, K_hat in _WORKER_FINGERPRINTS.items():
        pce, shift = compute_patch_pce(W_patch, K_hat, search_radius=search_radius)
        image_pces[fp_name] = pce

        is_genuine = fp_name.startswith(true_label)
        roc_pairs.append((1 if is_genuine else 0, pce))

        if pce > best_pce:
            best_pce = pce
            best_match_label = fp_name
            best_shift = shift

    del W_patch
    gc.collect()

    if best_match_label is None:
        return None, f"No matching fingerprint found for {img_name}"

    matching_camera_pce = image_pces.get(matching_fp_key, np.nan)
    
    pred_camera = best_match_label
    for suffix in ['_wavelet_rgb_green_fingerprint', '_sigma_rgb_green_fingerprint', '_fingerprint']:
        if pred_camera.endswith(suffix):
            pred_camera = pred_camera[:-len(suffix)]
            break

    is_correct = best_match_label.startswith(true_label)
    elapsed = time.time() - start_time

    res = {
        'True_Camera': true_label,
        'Image_Name': img_name,
        'Predicted_Camera': pred_camera,
        'Best_PCE': best_pce,
        'Matching_Camera_PCE': matching_camera_pce,
        'Shift_Y': best_shift[0],
        'Shift_X': best_shift[1],
        'Correct': is_correct,
        'Elapsed_s': elapsed,
        'ROC_Pairs': roc_pairs
    }

    log_msg = (
        f" -> {img_name} | Best Match: {pred_camera} | "
        f"Best PCE: {best_pce:.2f} | True PCE: {matching_camera_pce:.2f} | "
        f"Correct: {is_correct} ({elapsed:.1f}s)"
    )

    return res, log_msg

# =====================================================================
# 5. EVALUATION PIPELINE WITH CHECKPOINTING
# =====================================================================

def evaluate_dataset(
    query_set_dir, fingerprint_dir, filter_type='wavelet', search_radius=60, max_workers=4, patch_size=(512, 512)
):
    print(f"\nScanning fingerprints from: {fingerprint_dir}")
    fp_files = [f for f in os.listdir(fingerprint_dir) if f.endswith('.npz')]
    print(f"Found {len(fp_files)} fingerprints.\n")

    camera_folders = [
        d for d in sorted(os.listdir(query_set_dir))
        if os.path.isdir(os.path.join(query_set_dir, d))
    ]
    valid_extensions = ('.dng', '.jpg', '.jpeg', '.jp2', '.j2k', '.jxr', '.jxl')

    tasks = []
    for cam_folder in camera_folders:
        cam_path = os.path.join(query_set_dir, cam_folder)
        img_files = [
            os.path.join(cam_path, f)
            for f in os.listdir(cam_path)
            if f.lower().endswith(valid_extensions)
        ]
        for img_path in img_files:
            tasks.append((img_path, cam_folder, filter_type, search_radius, patch_size))

    if not tasks:
        print("[!] No valid test images found.")
        return

    print(f"Starting evaluation on {len(tasks)} images using patch size {patch_size[1]}x{patch_size[0]}...")

    results = []
    y_true, y_pred = [], []
    roc_labels, roc_scores = [], []
    workers = min(os.cpu_count() or 4, max_workers)
    checkpoint_interval = 200

    # Using ProcessPoolExecutor with initializer to avoid copying fingerprints per task
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, 
        initializer=init_worker, 
        initargs=(fingerprint_dir,)
    ) as executor:
        futures = [executor.submit(process_single_image, task) for task in tasks]

        for i, future in enumerate(tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Evaluating")):
            res, log_msg = future.result()
            if log_msg:
                tqdm.write(log_msg)

            if res is not None:
                results.append(res)
                y_true.append(res['True_Camera'])
                y_pred.append(res['Predicted_Camera'])

                for label, score in res['ROC_Pairs']:
                    roc_labels.append(label)
                    roc_scores.append(score)

            # Periodic Checkpoint Saving to prevent data loss
            if (i + 1) % checkpoint_interval == 0:
                temp_df = pd.DataFrame([r for r in results]).drop(columns=['ROC_Pairs'], errors='ignore')
                temp_df.to_excel(f"results/jxl_500kb_pce_evaluation_checkpoint_{filter_type}.xlsx", index=False)

    if not results:
        print("\n[!] No valid results were computed.")
        return

    df_results = pd.DataFrame(results).drop(columns=['ROC_Pairs'])
    pce_excel_path = f"results/jxl_500kb_pce_evaluation_results_{filter_type}.xlsx"
    df_results.to_excel(pce_excel_path, index=False)
    print(f"\n[✓] PCE scores saved to {pce_excel_path}")

    labels = sorted(list(set(y_true).union(set(y_pred))))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    metrics_report = classification_report(
        y_true, y_pred, labels=labels, zero_division=0, output_dict=True
    )
    df_metrics = pd.DataFrame(metrics_report).transpose()

    roc_auc = np.nan
    tpr_at_1_fpr = np.nan
    tpr_at_5_fpr = np.nan

    if roc_labels:
        roc_labels_np = np.asarray(roc_labels, dtype=np.int32)
        roc_scores_np = np.asarray(roc_scores, dtype=np.float64)

        fpr, tpr, thresholds = roc_curve(roc_labels_np, roc_scores_np)
        roc_auc = roc_auc_score(roc_labels_np, roc_scores_np)

        valid_1 = np.where(fpr <= 0.01)[0]
        if len(valid_1) > 0:
            tpr_at_1_fpr = np.max(tpr[valid_1])

        valid_5 = np.where(fpr <= 0.05)[0]
        if len(valid_5) > 0:
            tpr_at_5_fpr = np.max(tpr[valid_5])

        df_roc = pd.DataFrame({'FPR': fpr, 'TPR': tpr, 'Threshold': thresholds})
        roc_excel_path = f"results/jxl_500kb_roc_curve_{filter_type}.xlsx"
        df_roc.to_excel(roc_excel_path, index=False)
        print(f"[✓] ROC curve saved to {roc_excel_path}")

    summary_excel_path = f"results/jxl_500kb_evaluation_metrics_summary_{filter_type}.xlsx"
    df_prnu_metrics = pd.DataFrame({
        'Metric': [
            'ROC-AUC',
            'TPR @ FPR <= 1%',
            'TPR @ FPR <= 5%',
            'Number of Genuine Pairs',
            'Number of Impostor Pairs',
        ],
        'Value': [
            roc_auc,
            tpr_at_1_fpr,
            tpr_at_5_fpr,
            int(np.sum(np.asarray(roc_labels) == 1)),
            int(np.sum(np.asarray(roc_labels) == 0)),
        ],
    })

    df_overall = pd.DataFrame({
        'Metric': ['Accuracy', 'Weighted Precision', 'Weighted Recall', 'Weighted F1'],
        'Value': [accuracy, precision, recall, f1],
    })

    with pd.ExcelWriter(summary_excel_path) as writer:
        cm_df.to_excel(writer, sheet_name='Confusion_Matrix')
        df_metrics.to_excel(writer, sheet_name='Classification_Report')
        df_overall.to_excel(writer, sheet_name='Overall_Metrics', index=False)
        df_prnu_metrics.to_excel(writer, sheet_name='PRNU_ROC_Metrics', index=False)

    print(f"[✓] Evaluation metrics saved to {summary_excel_path}")
    print("\n" + "=" * 70)
    print("FINAL PRNU / CAMERA IDENTIFICATION METRICS")
    print("=" * 70)
    print(f"Accuracy:           {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall:    {recall:.4f}")
    print(f"Weighted F1:        {f1:.4f}")
    if not np.isnan(roc_auc):
        print(f"\nROC-AUC:            {roc_auc:.4f}")
        print(f"TPR @ 1% FPR:       {tpr_at_1_fpr:.4f}")
        print(f"TPR @ 5% FPR:       {tpr_at_5_fpr:.4f}")
    print("=" * 70)

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    QUERY_SET_DIR = "/home/chinasa/python_projects/fivek_prnu_split/query_set_compressed_500kb/jxl"
    FINGERPRINT_DIR = "/home/chinasa/python_projects/prnu_compression/fivek_rgb_green_wavelet_fingerprints"

    evaluate_dataset(
        query_set_dir=QUERY_SET_DIR,
        fingerprint_dir=FINGERPRINT_DIR,
        filter_type='wavelet',
        search_radius=60,
        max_workers=4,
        patch_size=(512, 512)
    )