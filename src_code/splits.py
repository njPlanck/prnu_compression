import os
import shutil
import random


def create_prnu_reference_query_split(source_dir, output_dir, ref_count=40):
    """
    Splits each camera body's images into:

        output_dir/
            reference_set/
            query_set/

    Images are MOVED (not copied) to save disk space.
    """

    random.seed(42)

    if not os.path.isdir(source_dir):
        print(f"Source directory not found:\n{source_dir}")
        return

    ref_dir = os.path.join(output_dir, "reference_set")
    query_dir = os.path.join(output_dir, "query_set")

    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(query_dir, exist_ok=True)

    print(f"\nCreating PRNU split ({ref_count} reference images per camera)...\n")

    for body_name in sorted(os.listdir(source_dir)):

        body_path = os.path.join(source_dir, body_name)

        if not os.path.isdir(body_path):
            continue

        dng_files = [
            f for f in os.listdir(body_path)
            if f.lower().endswith(".dng")
        ]

        total_images = len(dng_files)

        if total_images < ref_count:
            print(f"[!] Skipping {body_name} ({total_images} images)")
            continue

        random.shuffle(dng_files)

        ref_files = dng_files[:ref_count]
        query_files = dng_files[ref_count:]

        ref_body_dir = os.path.join(ref_dir, body_name)
        query_body_dir = os.path.join(query_dir, body_name)

        os.makedirs(ref_body_dir, exist_ok=True)
        os.makedirs(query_body_dir, exist_ok=True)

        # Move reference images
        for image in ref_files:
            shutil.move(
                os.path.join(body_path, image),
                os.path.join(ref_body_dir, image)
            )

        # Move query images
        for image in query_files:
            shutil.move(
                os.path.join(body_path, image),
                os.path.join(query_body_dir, image)
            )

        # Remove the empty source folder
        try:
            os.rmdir(body_path)
        except OSError:
            pass

        print(
            f"[✓] {body_name:<45} "
            f"Reference: {len(ref_files):3d} | "
            f"Query: {len(query_files):3d}"
        )

    print("\n" + "=" * 60)
    print("PRNU dataset split complete.")
    print(f"Reference set : {ref_dir}")
    print(f"Query set     : {query_dir}")
    print("=" * 60)


if __name__ == "__main__":

    SOURCE_DIR = "/home/chinasa/python_projects/fivek_prnu_ready"

    OUTPUT_DIR = "/home/chinasa/python_projects/fivek_prnu_split"

    create_prnu_reference_query_split(
        SOURCE_DIR,
        OUTPUT_DIR,
        ref_count=50
    )