#!/usr/bin/env python3

import os
import json
import shutil
import subprocess
from collections import defaultdict


def clean(text):
    """Make strings safe for folder names."""
    if text is None:
        return "Unknown"

    text = str(text).strip()

    if text == "":
        return "Unknown"

    bad = '<>:"/\\|?*'

    for c in bad:
        text = text.replace(c, "_")

    return text.replace(" ", "_")


def get_serial(item):
    """
    Returns the best available BODY serial number.

    Priority is important because different manufacturers
    store the camera body serial in different locations.
    """

    serial_fields = [

        # Generic EXIF
        "CameraSerialNumber",
        "BodySerialNumber",
        "InternalSerialNumber",
        "SerialNumber",

        # Manufacturer specific
        "CanonSerialNumber",
        "SonySerialNumber",
        "CameraSerialNumber2",

        # Misc
        "CameraID",

    ]

    for field in serial_fields:

        value = item.get(field)

        if value is None:
            continue

        value = str(value).strip()

        if value != "":
            return clean(value)

    return None


def get_fallback_identifier(item):
    """
    Used ONLY if no body serial exists.
    """

    owner = clean(item.get("OwnerName"))
    artist = clean(item.get("Artist"))
    lens = clean(item.get("LensModel"))
    copyright = clean(item.get("Copyright"))

    if owner != "Unknown":
        return f"OWNER_{owner}"

    if artist != "Unknown":
        return f"ARTIST_{artist}"

    if copyright != "Unknown":
        return f"COPYRIGHT_{copyright}"

    if lens != "Unknown":
        return f"LENS_{lens}"

    return "UNKNOWN_CAMERA"


# Group camera data

def group_by_camera(dataset_dir):

    if shutil.which("exiftool") is None:
        print("ERROR: ExifTool is not installed.")
        return

    print("\nReading metadata...\n")

    cmd = [

        "exiftool",

        "-json",

        "-r",

        "-ext", "dng",

        "-Make",
        "-Model",

        # Serial fields
        "-CameraSerialNumber",
        "-BodySerialNumber",
        "-InternalSerialNumber",
        "-SerialNumber",
        "-CanonSerialNumber",
        "-SonySerialNumber",
        "-CameraSerialNumber2",
        "-CameraID",

        # Fallback metadata
        "-OwnerName",
        "-Artist",
        "-Copyright",
        "-LensModel",

        dataset_dir,
    ]

    try:

        result = subprocess.check_output(
            cmd,
            stderr=subprocess.DEVNULL
        )

        metadata = json.loads(
            result.decode("utf-8", errors="ignore")
        )

    except Exception as e:

        print(e)
        return

    groups = defaultdict(list)

    serial_count = 0
    fallback_count = 0

    for item in metadata:

        file = item.get("SourceFile")

        if file is None:
            continue

        if not os.path.isfile(file):
            continue

        make = clean(item.get("Make", "UnknownMake"))
        model = clean(item.get("Model", "UnknownModel"))

        serial = get_serial(item)

        if serial is not None:

            key = f"{make}_{model}_SN_{serial}"
            serial_count += 1

        else:

            fallback = get_fallback_identifier(item)
            key = f"{make}_{model}_{fallback}"
            fallback_count += 1

        groups[key].append(file)

    print(f"Images found            : {len(metadata)}")
    print(f"Camera groups detected  : {len(groups)}")
    print(f"Images with serial      : {serial_count}")
    print(f"Fallback grouping used  : {fallback_count}")
    print()

    moved = 0

    for folder_name, files in sorted(groups.items()):

        folder = os.path.join(dataset_dir, folder_name)

        os.makedirs(folder, exist_ok=True)

        print("=" * 70)
        print(folder_name)
        print(f"Images: {len(files)}")

        for src in files:

            dst = os.path.join(
                folder,
                os.path.basename(src)
            )

            if os.path.abspath(src) == os.path.abspath(dst):
                continue

            if os.path.exists(dst):
                continue

            shutil.move(src, dst)
            moved += 1

    # Remove empty folders
    
    for root, dirs, files in os.walk(dataset_dir, topdown=False):

        if root == dataset_dir:
            continue

        try:
            if len(os.listdir(root)) == 0:
                os.rmdir(root)
        except:
            pass

    print("\n" + "=" * 70)
    print("Finished")
    print("=" * 70)
    print(f"Folders created : {len(groups)}")
    print(f"Files moved     : {moved}")


if __name__ == "__main__":
    DATASET_DIR = "/home/chinasa/python_projects/fivek_dataset/fivek_dataset/raw_photos"
    group_by_camera(DATASET_DIR)