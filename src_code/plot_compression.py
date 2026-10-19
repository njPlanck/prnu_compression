import matplotlib
# Force non-interactive backend to prevent warnings on servers
matplotlib.use('Agg')

import os
import cv2
import imagecodecs
import numpy as np
import matplotlib.pyplot as plt
import rawpy

def load_image(img_path):
    ext = os.path.splitext(img_path)[1].lower()
    
    # Handle DNG RAW files using rawpy
    if ext == '.dng':
        with rawpy.imread(img_path) as raw:
            img = raw.postprocess(
                use_camera_wb=True, 
                half_size=False, 
                no_auto_bright=True,
                output_bps=16, 
                user_flip=0
            )
        img = (img / 256.0).astype(np.uint8)
        
    elif ext == '.jxl':
        with open(img_path, 'rb') as f:
            data = f.read()
        img = imagecodecs.jpegxl_decode(data)
        
    elif ext in ['.jxr', '.hdp', '.wdp']:
        with open(img_path, 'rb') as f:
            data = f.read()
        img = imagecodecs.jpegxr_decode(data)
        
    elif ext in ['.jp2', '.j2k']:
        with open(img_path, 'rb') as f:
            data = f.read()
        img = imagecodecs.jpeg2k_decode(data)
        
    else:
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if img is None:
        raise ValueError(f"Could not load image: {img_path}")

    img = np.asarray(img)

    # --- NORMALIZE PLANAR & DIMENSION FORMATS TO (H, W, 3) RGB ---
    if img.ndim == 3 and img.shape[0] == 3 and img.shape[2] != 3:
        img = np.transpose(img, (1, 2, 0))
    elif img.ndim == 2:
        img = np.stack((img,) * 3, axis=-1)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    if img.dtype != np.uint8:
        if img.max() > 1.0:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        else:
            img = (img * 255.0)
        img = np.clip(img, 0, 255).astype(np.uint8)

    return img

def compare_image_formats(format_paths, output_filename="codec_residuals_dashboard.png", step=2, crop_size=1024, zoom_window=(350, 650)):
    """
    Extracts 1024x1024 crops and plots Error Residuals (Codec - RAW) 
    within a zoomed pixel window to explicitly visualize data loss per channel.
    """
    
    format_styles = {
        'JPEG':       {'color': "#AF282A", 'linestyle': '-'},
        'JPEG XL':    {'color': "#020202", 'linestyle': '-'},
        'JPEG 2000':  {'color': "#61ce5b", 'linestyle': '-.'},
        'JPEG XR':    {'color': "#FB0509", 'linestyle': ':'}
    }

    processed_data = {}

    for fmt_name, path in format_paths.items():
        if not os.path.exists(path):
            print(f"Skipping {fmt_name}: File not found at {path}")
            continue
            
        try:
            img = load_image(path)
            h, w = img.shape[:2]
            
            if h < crop_size or w < crop_size:
                print(f"Skipping {fmt_name}: Image size ({w}x{h}) is smaller than required {crop_size}x{crop_size}.")
                continue
                
            # 1. Take a 1024x1024 center crop
            start_y = (h - crop_size) // 2
            start_x = (w - crop_size) // 2
            crop = img[start_y:start_y+crop_size, start_x:start_x+crop_size]
            
            # 2. Extract center horizontal strip using step size
            center_row_idx = crop.shape[0] // 2
            strip = crop[center_row_idx, ::step, :].astype(np.float32) 
            
            processed_data[fmt_name] = {
                'crop': crop,
                'strip': strip,
                'center_row': center_row_idx
            }
        except Exception as e:
            print(f"Error processing {fmt_name} from {path}: {e}")

    if 'RAW (.dng)' not in processed_data:
        print("Error: 'RAW (.dng)' baseline is required to calculate error residuals.")
        return

    num_formats = len(processed_data)
    
    # Create the dashboard layout
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, num_formats)

    # --- Plot Full-Color Visual Crops (Row 0) ---
    for col_idx, (fmt_name, data) in enumerate(processed_data.items()):
        ax_img = fig.add_subplot(gs[0, col_idx])
        ax_img.imshow(data['crop'])
        ax_img.set_title(f"{fmt_name}\n({crop_size}x{crop_size} Center Crop)", fontsize=10, fontweight='bold')
        ax_img.axis('off')
        ax_img.axhline(y=data['center_row'], color='yellow', linestyle='--', linewidth=1.2, alpha=0.9)

    # --- Plot Error Residual Charts (Rows 1, 2, 3 for R, G, B) ---
    channel_rows = [1, 2, 3]
    channel_names = ['Red Channel Loss (Codec - RAW)', 'Green Channel Loss (Codec - RAW)', 'Blue Channel Loss (Codec - RAW)']
    
    raw_strip = processed_data['RAW (.dng)']['strip']

    axes_lines = []
    for i, r_idx in enumerate(channel_rows):
        ax = fig.add_subplot(gs[r_idx, :], sharex=axes_lines[0] if axes_lines else None)
        axes_lines.append(ax)
        
        # Clean zero-error baseline reference line
        ax.axhline(0, color='black', linestyle='-', linewidth=1.2, alpha=0.7)
        
        for fmt_name, data in processed_data.items():
            if fmt_name == 'RAW (.dng)':
                continue  # Skip RAW baseline itself
                
            strip = data['strip']
            style = format_styles.get(fmt_name, {'color': '#555555', 'linestyle': '-'})
            x_indices = np.arange(0, crop_size, step)
            
            # CALCULATE ERROR RESIDUAL (Compressed Pixel minus RAW Baseline Pixel)
            residual = strip[:, i] - raw_strip[:len(strip), i]
            
            ax.plot(
                x_indices[:len(residual)], 
                residual, 
                label=fmt_name, 
                color=style['color'], 
                linestyle=style['linestyle'], 
                alpha=0.85, 
                linewidth=1.2
            )
            
        ax.set_ylabel(channel_names[i])
        
        # Zoom into the specified window
        ax.set_xlim(zoom_window[0], zoom_window[1])
        # Optional fixed y-limit range to easily compare error magnitudes (e.g., -15 to +15 intensity levels)
        ax.set_ylim(-20, 20)
        
        ax.grid(True, linestyle='--', alpha=0.8)
        ax.legend(loc='upper right', frameon=True)

    axes_lines[-1].set_xlabel(f"Pixel Position Across Width (Zoomed Window: {zoom_window[0]} to {zoom_window[1]}, Step={step})")
    plt.suptitle(f"Codec Error Residual Dashboard: Data Loss Per Channel vs RAW", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save output securely
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Residual dashboard successfully saved to: {os.path.abspath(output_filename)}")

# --- Example Usage ---
if __name__ == "__main__":
    paths = {
        'RAW (.dng)': '/home/chinasa/python_projects/fivek_prnu_split/query_set/Canon_Canon_EOS_5D_SN_620302732/a1369-jmac_MG_5781.dng',
        'JPEG': '/home/chinasa/python_projects/fivek_prnu_split/query_set_compressed_500kb/jpeg/Canon_Canon_EOS_5D_SN_620302732/Canon_Canon_EOS_5D_SN_620302732_a1369-jmac_MG_5781.jpeg',
        'JPEG XL': '/home/chinasa/python_projects/fivek_prnu_split/query_set_compressed_500kb/jxl/Canon_Canon_EOS_5D_SN_620302732/Canon_Canon_EOS_5D_SN_620302732_a1369-jmac_MG_5781.jxl',
        'JPEG 2000': '/home/chinasa/python_projects/fivek_prnu_split/query_set_compressed_500kb/jp2/Canon_Canon_EOS_5D_SN_620302732/Canon_Canon_EOS_5D_SN_620302732_a1369-jmac_MG_5781.jp2',
        'JPEG XR': '/home/chinasa/python_projects/fivek_prnu_split/query_set_compressed_500kb/jxr/Canon_Canon_EOS_5D_SN_620302732/Canon_Canon_EOS_5D_SN_620302732_a1369-jmac_MG_5781.jxr'
    }
    
    compare_image_formats(paths, crop_size=1024, step=1, zoom_window=(350, 650))