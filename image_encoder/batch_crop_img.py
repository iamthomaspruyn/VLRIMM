import cv2
import shutil
from pathlib import Path
from tqdm import tqdm

# Import the cropping function from your other file
try:
    from crop_test_twopass import extract_grid_subplots
except ImportError:
    print("Error: 'crop_test_twopass.py' not found. Please place it in this folder.")
    exit()

# --- Configuration ---
INPUT_ROOT = "./docling_output"  # Top-level folder


# ---------------------

def batch_process_structure():
    root_path = Path(INPUT_ROOT)

    if not root_path.exists():
        print(f"Error: Input root '{INPUT_ROOT}' does not exist.")
        return

    # 1. Find all folders named "images"
    print(f"Scanning '{INPUT_ROOT}' for 'images' folders...")
    source_folders = list(root_path.rglob("images"))

    if not source_folders:
        print("No folders named 'images' found inside the root directory.")
        return

    print(f"Found {len(source_folders)} 'images' folders. Starting processing...")

    for source_folder in source_folders:
        # 2. Determine Output Path
        # Structure: .../PaperName/images -> .../PaperName/images_processed
        paper_folder = source_folder.parent
        output_folder = paper_folder / "images_processed"

        # Create output directory
        output_folder.mkdir(parents=True, exist_ok=True)

        # Get list of images
        image_files = list(source_folder.glob("*.png")) + \
                      list(source_folder.glob("*.jpg")) + \
                      list(source_folder.glob("*.jpeg"))

        if not image_files:
            continue

        print(f"\nProcessing: {paper_folder.name}")

        for img_path in tqdm(image_files, desc=f"  Inside {source_folder.name}"):
            try:
                # 3. Check INPUT Dimensions (Must be >= 200px)
                # We load the image to check dimensions
                img = cv2.imread(str(img_path))

                if img is None:
                    continue

                h, w = img.shape[:2]

                # STRICT INPUT FILTER:
                # If either side is smaller than 200, skip the file entirely.
                if h < 200 or w < 200:
                    # print(f"    Skipping {img_path.name}: Input too small ({w}x{h})")
                    continue

                # 4. Process (Output size is handled inside the imported function)
                extract_grid_subplots(
                    str(img_path),
                    output_dir=str(output_folder),
                    crop_top_ratio=0.0,
                    crop_bottom_ratio=0.0
                )

            except Exception as e:
                print(f"    Error processing {img_path.name}: {e}")

    print("\nAll processing complete.")


if __name__ == "__main__":
    batch_process_structure()