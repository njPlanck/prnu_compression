import os
import numpy as np
from skimage import io, restoration
from scipy import signal
from scipy.signal import fftconvolve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import imagecodecs
import cv2
import pyjxl
#import pillow_jxl
from PIL import Image
from numba import njit
from reference import sigma_filter, gaussian_denoise, wavelet_wiener_denoise


# sigma filter to remove nnoise from the scene
@njit
def sigma_filter(img, window_size=3, sigma=0.03):
    pad = window_size // 2
    h, w = img.shape
    out = np.zeros_like(img)
    for y in range(pad, h - pad):
        for x in range(pad, w - pad):
            sum_val = 0.0
            count = 0
            center = img[y, x]
            for dy in range(-pad, pad + 1):
                for dx in range(-pad, pad + 1):
                    ny, nx = y + dy, x + dx
                    val = img[ny, nx]
                    if abs(val - center) <= sigma:
                        sum_val += val
                        count += 1
            if count > 0:
                out[y, x] = sum_val / count
            else:
                out[y, x] = center
    return out

#normalise the residual

def normalize_image(img):
    img = img.astype(np.float64)
    img_min, img_max = np.min(img), np.max(img)
    return (img - img_min) / (img_max - img_min + 1e-8)


# Extract PRNU noise with gaussioab
def extract_noise(img):
    denoised = gaussian_filter(img,sigma=1)
    residual = img - denoised
    return residual

'''
# Extract PRNU noise
def extract_noise(img):
    denoised = sigma_filter(img)
    residual = img - denoised
    residual = normalize_image(residual)
    return residual
'''

# Compute average fingerprint for a camera
def get_camera_fingerprint(images):
    prnu = np.zeros_like(images[0])
    for img in images:
        prnu += extract_noise(img)
    return prnu / len(images)



# Load grayscale images from a folder 
def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png','.jp2')):
            img = Image.open(os.path.join(folder_path, filename)).convert("L")  # 'L' mode = grayscale
            img = np.array(img)  # Convert to NumPy array for further use
            img = img.astype(np.float32) / 255.0  # Normalize if needed

        elif filename.lower().endswith(('.jxl')):
            with open(os.path.join(folder_path, filename), 'rb') as f:
                img_bytes = f.read()
                img = imagecodecs.jpegxl_decode(img_bytes)
                img = np.array(img).astype(np.float32) / 255.0  # normalize
                if img.ndim == 3 and img.shape[2] in [3,4]:
                    # Convert RGB to grayscale (luminance)
                    # # Using standard weights: 0.299 R + 0.587 G + 0.114 B
                    img = np.dot(img[...,:3], [0.299, 0.587, 0.114]).astype(np.float32)
                else:
                    img = img  # Already single channel
        else:
            with open(os.path.join(folder_path, filename), 'rb') as f:
                img = f.read()
                img = imagecodecs.jpegxr_decode(img)
                img = np.array(img)  # Convert to NumPy array for further use
                img = img.astype(np.float32) / 255.0  # Normalize if needed
                if img.ndim == 3 and img.shape[2] in [3,4]:
                    # Convert RGB to grayscale (luminance)
                    # # Using standard weights: 0.299 R + 0.587 G + 0.114 B
                    img = np.dot(img[...,:3], [0.299, 0.587, 0.114]).astype(np.float32)
                else:
                    img = img  # Already single channel
        images.append(img)
    return images

    

# Create fingerprints for each camera folders


def build_camera_fingerprints(root_folder):
    fingerprints = {}
    for camera_name in os.listdir(root_folder):
        cam_folder = os.path.join(root_folder, camera_name)
        if os.path.isdir(cam_folder):
            images = load_images_from_folder(cam_folder)
            if images:
                fingerprints[camera_name] = get_camera_fingerprint(images)
                print(f"[INFO] Fingerprint created for {camera_name} using {len(images)} images.")
            else:
                print(f"[WARN] No images found for {camera_name}. Skipping.")
    return fingerprints


def match_camera(test_img, fingerprints):
    test_noise = extract_noise(test_img)
    test_norm = (test_noise - np.mean(test_noise)) / np.std(test_noise)

    best_match = None
    best_score = -np.inf

    for name, fingerprint in fingerprints.items():
        ref_norm = (fingerprint - np.mean(fingerprint)) / np.std(fingerprint)

        # Use FFT-based normalized cross-correlation
        corr = fftconvolve(test_norm, ref_norm[::-1, ::-1], mode='same')
        score = np.max(corr) / (test_norm.shape[0] * test_norm.shape[1])  # Optional: normalize

        if score > best_score:
            best_score = score
            best_match = name

    return best_match, best_score


def validate_on_test_set(test_folder, fingerprints):
    y_true = []
    y_pred = []

    for root, dirs, files in os.walk(test_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png','.jp2')):
                # Get full path
                file_path = os.path.join(root, file)

                # Load and normalize test image
                #test_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                #test_image = pyjxl.decode(file_path)
                test_image = Image.open(file_path).convert("L") 
                test_image = np.array(test_image)  # Convert to NumPy array for further use
                test_image = test_image.astype(np.float32) / 255.0  # Normalize to [0, 1]
            elif file.lower().endswith(('.jxl')):
                file_path = os.path.join(root, file)            
                with open(file_path, 'rb') as f:
                    test_image = f.read()
                    test_image = imagecodecs.jpegxl_decode(test_image)
                    test_image = np.array(test_image).astype(np.float32) / 255.0  # normalize
                    if test_image.ndim == 3 and test_image.shape[2] in [3,4]:
                        # Convert RGB to grayscale (luminance)
                        # # Using standard weights: 0.299 R + 0.587 G + 0.114 B
                        test_image = np.dot(test_image[...,:3], [0.299, 0.587, 0.114]).astype(np.float32)
                    else:
                        test_image = test_image  # Already single channel
            else:
                file_path = os.path.join(root, file)
                with open(file_path, 'rb') as f:
                    test_image = f.read()
                    test_image = imagecodecs.jpegxr_decode(test_image)
                    test_image = np.array(test_image)  # Convert to NumPy array for further use
                    test_image = test_image.astype(np.float32) / 255.0  # Normalize to [0, 1]
                    if test_image.ndim == 3 and test_image.shape[2] in [3,4]:
                       # Convert RGB to grayscale (luminance)
                       # Using standard weights: 0.299 R + 0.587 G + 0.114 B
                       test_image = np.dot(test_image[...,:3], [0.299, 0.587, 0.114]).astype(np.float32)
                    else:
                        test_image = test_image  # Already single channel
            

            # Match to camera
            predicted_camera, score = match_camera(test_image, fingerprints)

            # Assume ground truth camera name is the parent folder name (adjust if different)
            ground_truth_camera = os.path.basename(os.path.dirname(file_path))

            y_true.append(ground_truth_camera)
            y_pred.append(predicted_camera)

            print(f"[TEST] {file}: True={ground_truth_camera}, Predicted={predicted_camera} (Score: {score:.2f})")

    return y_true, y_pred

# Run validation

if __name__ == "__main__":
    # Path to train and test folders
    train_root = "train_images"
    test_root =  "testjpeg100kb" 

    # Build fingerprints
    camera_fingerprints = build_camera_fingerprints(train_root)

    # Validate
    y_true, y_pred = validate_on_test_set(test_root, camera_fingerprints)

    # Confusion Matrix
    labels = sorted(list(set(y_true) | set(y_pred)))  # All labels seen
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    plt.figure(figsize=(10, 8))
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix: Camera Identification via PRNU")
    plt.show()