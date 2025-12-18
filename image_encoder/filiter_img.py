import cv2
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm

# --- Configuration ---
INPUT_ROOT = "./docling_output"  # Root folder containing paper folders
# Thresholds
WHITE_PIXEL_THRESH = 230  # Intensity (0-255) to be considered "white"
WHITE_AREA_RATIO = 0.40  # Reject if >40% of pixels are white

SATURATION_THRESH = 20  # HSV Saturation (0-255) to be considered "colored"
COLOR_AREA_RATIO = 0.20  # Reject if >25% of pixels have high saturation


# ---------------------

def is_sem_image(img_bgr):
    """
    Returns True if image looks like an SEM (kept).
    Returns False, Reason if it looks like a plot (rejected).
    """
    total_pixels = img_bgr.shape[0] * img_bgr.shape[1]

    # 1. White Background Check
    # Convert to grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Count pixels that are almost pure white
    white_count = np.sum(gray > WHITE_PIXEL_THRESH)

    if (white_count / total_pixels) > WHITE_AREA_RATIO:
        return False, f"Too White ({white_count / total_pixels:.2%} > {WHITE_AREA_RATIO:.0%})"

    # 2. Color Check
    # Convert to HSV (Hue, Saturation, Value)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Saturation channel is index 1
    s = hsv[:, :, 1]

    # Count pixels with significant saturation (color)
    # We use a threshold (e.g., 20) to ignore JPEG noise/slight greyscale drift
    color_pixel_count = np.sum(s > SATURATION_THRESH)

    if (color_pixel_count / total_pixels) > COLOR_AREA_RATIO:
        return False, f"Too Colored ({color_pixel_count / total_pixels:.2%} > {COLOR_AREA_RATIO:.0%})"

    return True, "Passed"


def filter_images():
    root_path = Path(INPUT_ROOT)

    # Find all 'images_processed' folders
    processed_folders = list(root_path.rglob("images_processed"))

    if not processed_folders:
        print("No 'images_processed' folders found.")
        return

    print(f"Filtering images in {len(processed_folders)} folders...")

    removed_count = 0
    kept_count = 0

    for folder in processed_folders:
        # Create a "filtered_out" folder sibling to images_processed
        # e.g. dataset/Paper1/filtered_out
        reject_folder = folder.parent / "filtered_out"
        reject_folder.mkdir(exist_ok=True)

        images = list(folder.glob("*.png")) + list(folder.glob("*.jpg"))

        for img_path in tqdm(images, desc=f"Scanning {folder.parent.name}", leave=False):
            try:
                img = cv2.imread(str(img_path))
                if img is None: continue

                is_sem, reason = is_sem_image(img)

                if not is_sem:
                    # Move to rejected folder
                    # We prepend the reason to the filename for easy debugging later
                    # e.g. "Too_White_image_01.png"
                    new_name = f"{reason.split(' ')[0]}_{img_path.name}"
                    dest_path = reject_folder / new_name

                    shutil.move(str(img_path), str(dest_path))
                    removed_count += 1
                else:
                    kept_count += 1

            except Exception as e:
                print(f"Error checking {img_path.name}: {e}")

    print(f"\nDone.")
    print(f"Kept: {kept_count} images (SEMs)")
    print(f"Removed: {removed_count} images (Plots/Diagrams)")
    print(f"Rejected images are moved to 'filtered_out' folders for review.")


if __name__ == "__main__":
    filter_images()