import torch
import torchvision.transforms as T
import pickle, gc
from compressai.zoo import bmshj2018_hyperprior

# Keep memory overhead low
torch.set_num_threads(1)

def convert_bin_to_png(bin_path, output_png_path, quality=8):
    # 1. Load bitstreams + metadata directly from the .bin payload
    with open(bin_path, 'rb') as f:
        strings, meta = pickle.load(f)

    # Compute latent dimensions
    latent_h = meta['pad_h'] // 16 * 4
    latent_w = meta['pad_w'] // 16 * 4

    # 2. Load model for Quality 8
    model = bmshj2018_hyperprior(quality=quality, pretrained=True).eval()

    # 3. Decode
    with torch.inference_mode():
        out_dec = model.decompress(strings, (latent_h, latent_w))

    # 4. Crop padding and save
    rec_tensor = out_dec['x_hat'].clamp(0, 1).squeeze(0)[:, :meta['orig_h'], :meta['orig_w']]
    rec_img = T.ToPILImage()(rec_tensor)

    # Free memory
    del out_dec, rec_tensor, model
    gc.collect()

    rec_img.save(output_png_path)
    print(f"Success! Saved decoded image to {output_png_path}")

if __name__ == "__main__":
    convert_bin_to_png("compressed_target.bin", "decompressed_result.png", quality=8)