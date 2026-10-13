import os
import io 
import cv2
import subprocess
import torch
import numpy as np
from PIL import Image
import imagecodecs
from piq import psnr, ssim, brisque
import torchvision.transforms as transforms
import csv
import rawpy
import jxlpy
import tempfile
import shutil

# Global transform
transform = transforms.ToTensor()

def load_input_image_as_pil(input_path):
    """Safely opens standard images OR develops RAW DNG images into a PIL Image."""
    if input_path.lower().endswith(".dng"):
        with rawpy.imread(input_path) as raw:
            rgb_array = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True)
        return Image.fromarray(rgb_array)
    else:
        return Image.open(input_path).convert("RGB")

def load_input_image_as_cv2(input_path):
    """Safely opens standard images OR develops RAW DNG images into a BGR OpenCV matrix."""
    if input_path.lower().endswith(".dng"):
        with rawpy.imread(input_path) as raw:
            rgb_array = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True)
        return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    else:
        return cv2.imread(input_path, cv2.IMREAD_COLOR)

def compute_bpp(file_path, width, height):
    size_bytes = os.path.getsize(file_path)
    return (size_bytes * 8) / (width * height)

def evaluate_metrics(img1, img2):
    if img1.mode != "RGB": img1 = img1.convert("RGB")
    if img2.mode != "RGB": img2 = img2.convert("RGB")
    x = transform(img1).unsqueeze(0)
    y = transform(img2).unsqueeze(0)
    if x.shape[1] != y.shape[1]:
        if x.shape[1] == 1: x = x.repeat(1, 3, 1, 1)
        if y.shape[1] == 1: y = y.repeat(1, 3, 1, 1)
    return {
        "PSNR": psnr(x, y, data_range=1.0).item(),
        "SSIM": ssim(x, y).item(),
        "BRISQUE": brisque(y).item()
    }

def compress_jpeg(input_path, output_path, target_size_kb, min_quality=1, max_quality=100):
    img1 = load_input_image_as_pil(input_path)
    target_bytes = target_size_kb * 1024
    low, high = min_quality, max_quality
    
    best_data = None
    best_diff = float("inf")
    
    while low <= high:
        quality = (low + high) // 2
        buffer = io.BytesIO()
        img1.save(buffer, "JPEG", quality=quality, subsampling=0, optimize=True, progressive=False)
        
        data = buffer.getvalue()
        size = len(data)
        diff = abs(size - target_bytes)
        
        # Always save the data that gets closest to the absolute target size
        if diff < best_diff:
            best_diff = diff
            best_data = data
            
        # Break early if we hit the target within a tiny 1% tolerance
        if diff < target_bytes * 0.01:
            break
            
        if size > target_bytes:
            # File is too big -> decrease quality
            high = quality - 1
        else:
            # File is too small -> increase quality
            low = quality + 1

    if best_data is not None:
        with open(output_path, "wb") as f:
            f.write(best_data)
            width, height = img1.size
            bpp = compute_bpp(output_path, width, height)
            metrics = evaluate_metrics(img1, Image.open(output_path).convert("RGB"))
            metrics["BPP"] = bpp
        return metrics
    return {"error": "Target size unreachable"}

def compress_jp2(input_path, output_path, target_size_kb, min_q=1, max_q=1000):
    target_bytes = target_size_kb * 1024
    img_cv = load_input_image_as_cv2(input_path)
    if img_cv is None: return {"error": "Read failed"}
    img1_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    low, high = min_q, max_q
    best_buf = None
    while low <= high:
        mid_q = (low + high) // 2
        success, encoded_img = cv2.imencode(".jp2", img_cv, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, mid_q])
        if not success: continue
        if len(encoded_img) <= target_bytes:
            best_buf = encoded_img
            low = mid_q + 1
        else:
            high = mid_q - 1
    if best_buf is not None:
        with open(output_path, "wb") as f:
            f.write(best_buf.tobytes())
            width, height = img1_pil.size
            bpp = compute_bpp(output_path, width, height)
            decoded = cv2.imdecode(best_buf, cv2.IMREAD_COLOR)
            metrics = evaluate_metrics(img1_pil, Image.fromarray(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)))
            metrics["BPP"] = bpp
        return metrics
    return {"error": "Target size unreachable"}

def compress_jxr(input_path, output_path, target_size_kb, tolerance=0.01, max_iterations=10):
    img_bgr = load_input_image_as_cv2(input_path)
    if img_bgr is None:
        return {"error": "Read failed"}

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if not img_rgb.flags["C_CONTIGUOUS"]:
        img_rgb = np.ascontiguousarray(img_rgb)

    img1_pil = Image.fromarray(img_rgb)
    target_bytes = target_size_kb * 1024

    try:
        imagecodecs.jpegxr_encode(img_rgb[:10, :10], level=0.5)
    except Exception as e:
        return {"error": f"JPEG-XR encoder unavailable: {e}"}

    try:
        size_at_0 = len(imagecodecs.jpegxr_encode(img_rgb, level=0.0))
        size_at_1 = len(imagecodecs.jpegxr_encode(img_rgb, level=1.0))
    except Exception as e:
        return {"error": f"JPEG-XR initial evaluation failed: {e}"}

    is_inverse = size_at_0 > size_at_1

    low, high = 0.0, 1.0
    best_data = None
    best_diff = float("inf")

    for _ in range(max_iterations):
        level = (low + high) / 2.0
        try:
            jxr_data = imagecodecs.jpegxr_encode(img_rgb, level=level)
        except Exception as e:
            return {"error": f"JPEG-XR encode failed: {e}"}

        size = len(jxr_data)
        diff = abs(size - target_bytes)

        if diff < best_diff:
            best_diff = diff
            best_data = jxr_data

        if diff < target_bytes * tolerance:
            break

        if size < target_bytes:
            if is_inverse:
                high = level
            else:
                low = level
        else:
            if is_inverse:
                low = level
            else:
                high = level

    if best_data is None:
        return {"error": "No valid JPEG-XR output generated"}

    try:
        with open(output_path, "wb") as f:
            f.write(best_data)
    except Exception as e:
        return {"error": f"Failed writing output: {e}"}

    try:
        decoded_arr = imagecodecs.jpegxr_decode(best_data)
        if decoded_arr.ndim == 2:
            img2_pil = Image.fromarray(decoded_arr).convert("RGB")
        elif decoded_arr.ndim == 3 and decoded_arr.shape[2] == 4:
            img2_pil = Image.fromarray(decoded_arr[:, :, :3])
        else:
            img2_pil = Image.fromarray(decoded_arr)
        
        width, height = img1_pil.size
        metrics = evaluate_metrics(img1_pil, img2_pil)
        metrics["BPP"] = (os.path.getsize(output_path) * 8) / (width * height)
        return metrics
    except Exception as e:
        return {"error": f"JPEG-XR decode failed: {e}"}

def compress_jxl(input_path, output_path, target_size_kb, tolerance=0.01, max_iterations=20):
    """
    Optimized JXL compression loop with 20 iterations and a higher distance ceiling 
    for maximum precision against the target file size.
    """
    if shutil.which("cjxl") is None or shutil.which("djxl") is None:
        return {"error": "cjxl or djxl toolkit not found in system PATH"}

    img = load_input_image_as_pil(input_path)
    target_bytes = target_size_kb * 1024

    temp_png = output_path + ".tmp.png"
    temp_jxl = output_path + ".tmp.jxl"
    best_jxl = output_path + ".best.jxl"
    temp_decode_png = output_path + ".decoded.png"
    
    img.save(temp_png, "PNG")

    # Expanded search space to account for highly complex images
    low, high = 0.0, 50.0
    best_diff = float("inf")

    for _ in range(max_iterations):
        d = (low + high) / 2.0
        # Increased decimal formatting for tighter granularity
        cmd = ["cjxl", temp_png, temp_jxl, "-d", f"{d:.4f}", "--effort", "7"]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.returncode != 0:
            # If the distance pushes past what this cjxl version allows, shrink the upper bound
            high = d
            continue

        size = os.path.getsize(temp_jxl)
        diff = abs(size - target_bytes)

        # Always save the absolute closest iteration
        if diff < best_diff:
            best_diff = diff
            shutil.copyfile(temp_jxl, best_jxl)

        # Break early if we nail it within the tiny tolerance margin
        if diff < target_bytes * tolerance:
            break

        if size > target_bytes:
            # File is too big -> increase distance for more compression
            low = d
        else:
            # File is too small -> decrease distance for better quality
            high = d

    # Cleanup temporary working files
    for path in (temp_png, temp_jxl):
        if os.path.exists(path):
            os.remove(path)

    if not os.path.exists(best_jxl):
        return {"error": "Binary search failed to produce a valid JPEG XL file"}

    # Finalize the best match
    shutil.move(best_jxl, output_path)
    width, height = img.size
    bpp = compute_bpp(output_path, width, height)
    
    try:
        subprocess.run(
            ["djxl", output_path, temp_decode_png],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        img2 = Image.open(temp_decode_png).convert("RGB")
        img2.load() 
    except Exception as e:
        return {"error": f"JPEG XL verification decode failed: {e}"}
    finally:
        if os.path.exists(temp_decode_png):
            os.remove(temp_decode_png)

    metrics = evaluate_metrics(img, img2)
    metrics["BPP"] = bpp
    return metrics

def process_and_isolate_dataset(flat_input_dir, main_output_dir, target_size_kb):
    formats = ["jpeg", "jp2", "jxr", "jxl"]
    compressors = {"jpeg": compress_jpeg, "jp2": compress_jp2, "jxr": compress_jxr, "jxl": compress_jxl}
    metrics_by_camera = {}

    print("Initializing Format-First Directory Architecture...")

    for root, _, files in os.walk(flat_input_dir):
        for filename in files:
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".tiff", ".dng")) and not filename.startswith("merge"):
                input_path = os.path.join(root, filename)
                
                # 1. Identify camera directly by the parent folder name
                rel_path = os.path.relpath(root, flat_input_dir)
                
                # Skip loose images without a camera folder
                if rel_path == ".":
                    print(f"  X Skipped (No parent folder identified): {filename}")
                    continue
                
                # The primary subfolder dictates the camera model
                camera_name = rel_path.split(os.sep)[0]
                camera_safe_name = camera_name.replace(" ", "_")

                # Create safe naming based on directory structure to prevent overwriting
                image_base_name = os.path.splitext(filename)[0]
                folder_prefix = rel_path.replace(os.sep, "_")
                unique_name = f"{folder_prefix}_{image_base_name}"

                print(f"\n Image: {unique_name} ➔ Identified Camera: {camera_name}")

                # 2. Build designated output infrastructure (Format -> Camera)
                for fmt in formats:
                    os.makedirs(os.path.join(main_output_dir, fmt, camera_safe_name), exist_ok=True)

                if camera_name not in metrics_by_camera:
                    metrics_by_camera[camera_name] = []

                # 3. Compress across pipelines
                for fmt in formats:
                    output_path = os.path.join(main_output_dir, fmt, camera_safe_name, f"{unique_name}.{fmt}")
                    metrics = compressors[fmt](input_path, output_path, target_size_kb)

                    if "error" in metrics:
                        print(f"    X {fmt.upper()}: {metrics['error']}")
                        metrics_by_camera[camera_name].append({
                            "Image": f"{unique_name}.{fmt}", "Format": fmt.upper(), 
                            "PSNR": None, "SSIM": None, "BRISQUE": None, "BPP": None, "Status": metrics["error"]
                        })
                    else:
                        print(f"    ✓ {fmt.upper()}: PSNR={metrics['PSNR']:.2f}, SSIM={metrics['SSIM']:.4f}, BPP={metrics['BPP']:.4f}")
                        metrics_by_camera[camera_name].append({
                            "Image": f"{unique_name}.{fmt}", "Format": fmt.upper(),
                            "PSNR": round(metrics["PSNR"], 2), "SSIM": round(metrics["SSIM"], 4), 
                            "BRISQUE": round(metrics["BRISQUE"], 2), "BPP": round(metrics["BPP"], 4), "Status": "OK"
                        })

    # 4. Finalize and write metrics CSV to the root of the output directory
    for camera_name, csv_rows in metrics_by_camera.items():
        camera_folder_name = camera_name.replace(" ", "_")
        csv_path = os.path.join(main_output_dir, f"{camera_folder_name}_metrics.csv")
        
        with open(csv_path, "w", newline="") as csvfile:
            fieldnames = ["Image", "Format", "PSNR", "SSIM", "BRISQUE", "BPP", "Status"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\n📊 Performance Matrix saved for {camera_name} at: {csv_path}")

if __name__ == "__main__":
    main_input_folder = "/home/chinasa/python_projects/fivek_prnu_split/query_set" 
    main_output_folder = "/home/chinasa/python_projects/fivek_prnu_split/query_set_compressed_500kb"
    target_kb = 500

    process_and_isolate_dataset(main_input_folder, main_output_folder, target_kb)