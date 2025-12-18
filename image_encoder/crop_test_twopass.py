import cv2
import numpy as np
import os


def get_tight_crop(img_chunk):
    """
    Standard tight cropper to remove residual borders.
    """
    if img_chunk is None or img_chunk.size == 0:
        return None
    gray = cv2.cvtColor(img_chunk, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(thresh)
    if coords is None: return None
    x, y, w, h = cv2.boundingRect(coords)
    # Add 1px padding to prevent cutting off thick axis lines
    return img_chunk[max(0, y - 1):y + h + 1, max(0, x - 1):x + w + 1]


def detect_regions(gray_img, kernel_size, min_area_ratio=0.001):
    """
    Detects blobs. Includes safety check for kernel size.
    """
    # SAFETY FIX: Ensure kernel dimensions are at least 1
    k_w = max(1, int(kernel_size[0]))
    k_h = max(1, int(kernel_size[1]))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, k_h))

    # 1. Canny
    edges = cv2.Canny(gray_img, 50, 150)

    # 2. Dilate
    dilated = cv2.dilate(edges, kernel, iterations=1)

    # 3. Contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = [cv2.boundingRect(c) for c in contours]

    # Filter tiny noise
    img_area = gray_img.shape[0] * gray_img.shape[1]
    min_area = img_area * min_area_ratio
    valid_boxes = [b for b in boxes if (b[2] * b[3]) > min_area]

    return valid_boxes


def extract_grid_subplots(image_path, output_dir="grid_crops", crop_top_ratio=0.0, crop_bottom_ratio=0.0):
    """
    Extracts subplots from a grid.

    Args:
        image_path (str): Path to the input image.
        output_dir (str): Folder to save crops.
        crop_top_ratio (float): Percentage of the top to remove (0.0 to 1.0). E.g. 0.05 removes top 5%.
        crop_bottom_ratio (float): Percentage of the bottom to remove (0.0 to 1.0).
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read {image_path}")
        return

    # Extract base name for prefix
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h_img, w_img = gray.shape

    # ==========================================================
    # PASS 1: ROW DETECTION
    # ==========================================================

    row_kernel = (w_img // 25, 1)

    row_boxes = detect_regions(gray, row_kernel, min_area_ratio=0.05)
    row_boxes.sort(key=lambda b: b[1])

    print(f"[{base_name}] Detected {len(row_boxes)} valid rows.")

    total_count = 0

    for i, (rx, ry, rw, rh) in enumerate(row_boxes):
        row_img = img[ry:ry + rh, rx:rx + rw]
        row_gray = gray[ry:ry + rh, rx:rx + rw]

        # ==========================================================
        # PASS 2: COLUMN DETECTION (Per Row)
        # ==========================================================
        col_kernel_h = max(1, rh // 20)
        col_kernel = (1, col_kernel_h)

        col_boxes = detect_regions(row_gray, col_kernel, min_area_ratio=0.01)
        col_boxes.sort(key=lambda b: b[0])

        print(f"  Row {i + 1}: Found {len(col_boxes)} columns")

        for j, (cx, cy, cw, ch) in enumerate(col_boxes):
            subplot = row_img[cy:cy + ch, cx:cx + cw]

            # 1. Tight Crop (remove whitespace)
            final_crop = get_tight_crop(subplot)

            if final_crop is not None:
                # 2. Percentage Crop (Remove Top X% and Bottom Y%)
                fc_h, fc_w = final_crop.shape[:2]

                # Calculate pixel amounts to strip
                strip_top = int(fc_h * crop_top_ratio)
                strip_bottom = int(fc_h * crop_bottom_ratio)

                # Ensure we don't crop the whole image away
                if strip_top + strip_bottom < fc_h:
                    final_crop = final_crop[strip_top: fc_h - strip_bottom, :]
                else:
                    print(f"    Warning: Crop ratios too high for Row {i + 1} Col {j + 1}. Skipping crop.")

                # 3. Check Minimum Size (must be > 100px)
                # Re-check shape after cropping
                fh, fw = final_crop.shape[:2]
                if fh < 100 or fw < 100:
                    print(f"    Skipping Row {i + 1} Col {j + 1}: Too small after crop ({fw}x{fh})")
                    continue

                # Save
                filename = os.path.join(output_dir, f"{base_name}_row{i + 1}_col{j + 1}.png")
                cv2.imwrite(filename, final_crop)
                total_count += 1

    print(f"Done. Extracted {total_count} subplots to '{output_dir}/'.")

# Usage Examples:
if __name__ == "__main__":
    # 1. Normal usage (no extra cropping)
    # extract_grid_subplots('./test_img/88.png')

    # 2. Crop top 5% and bottom 10% of every subplot
    extract_grid_subplots('./test_img/16.png', crop_top_ratio=0.0, crop_bottom_ratio=0.10)