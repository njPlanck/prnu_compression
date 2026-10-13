import os
import shutil


def create_prnu_subset(source_dir, target_dir, min_images=70):
    """
    Creates a PRNU-ready subset by copying camera folders that contain
    at least `min_images` DNG files.

    Folder structure:
        source_dir/
            Nikon_D700_SN_2008608/
            Nikon_D700_SN_2008611/
            Canon_5D_SN_123456/
            ...

    Output:
        target_dir/
            Nikon_D700_SN_2008608/
            Canon_5D_SN_123456/
            ...
    """

    if not os.path.isdir(source_dir):
        print(f"Source directory not found:\n{source_dir}")
        return

    os.makedirs(target_dir, exist_ok=True)

    copied_folders = 0
    copied_images = 0
    skipped_folders = 0

    print(f"\nSelecting folders with at least {min_images} DNG images...\n")

    for folder in sorted(os.listdir(source_dir)):

        source_folder = os.path.join(source_dir, folder)

        if not os.path.isdir(source_folder):
            continue

        dng_files = [
            f for f in os.listdir(source_folder)
            if f.lower().endswith(".dng")
        ]

        image_count = len(dng_files)

        if image_count >= min_images:

            destination = os.path.join(target_dir, folder)

            if os.path.exists(destination):
                print(f"[!] Already exists: {folder}")
                continue

            shutil.copytree(source_folder, destination)

            copied_folders += 1
            copied_images += image_count

            print(f"[✓] {folder:<60} {image_count:4d} images")

        else:

            skipped_folders += 1

            print(f"[-] {folder:<60} {image_count:4d} images")

    print("\n" + "=" * 70)
    print("PRNU subset created")
    print("=" * 70)
    print(f"Folders copied : {copied_folders}")
    print(f"Folders skipped: {skipped_folders}")
    print(f"Images copied  : {copied_images}")
    print(f"Output         : {target_dir}")


if __name__ == "__main__":

    SOURCE_DIR = "/home/chinasa/python_projects/fivek_dataset/fivek_dataset/raw_photos"

    TARGET_DIR = "/home/chinasa/python_projects/fivek_prnu_ready"

    create_prnu_subset(
        SOURCE_DIR,
        TARGET_DIR,
        min_images=70
    )