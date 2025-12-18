import torch
import numpy as np
import pickle
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel
from torchvision import transforms

# --- Configuration ---
# DINOv3 Model ID (ViT-Small for efficiency on CPU)
MODEL_HF_ID = "facebook/dinov3-vits16plus-pretrain-lvd1689m"
DEVICE = "cpu"  # Running on CPU as requested

INPUT_ROOT = "./docling_output"  # Root folder containing paper folders
OUTPUT_PICKLE = "dinov3_nested_dict.pkl"  # Output file name

# Image Preprocessing (Standard ImageNet)
PROCESS_SIZE = 256
CROP_SIZE = 256
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------

def get_transforms():
    return transforms.Compose([
        transforms.Resize(PROCESS_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def extract_dinov3_nested_dict():
    # 1. Load Model
    print(f"Loading DINOv3 ({MODEL_HF_ID}) on {DEVICE}...")
    try:
        # trust_remote_code=True is required for DINOv3
        model = AutoModel.from_pretrained(MODEL_HF_ID, trust_remote_code=True).to(DEVICE).eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Setup
    root_path = Path(INPUT_ROOT)
    transform = get_transforms()

    # Outer Dictionary: { "PaperName": { ... } }
    master_dictionary = {}

    # Find all 'images_processed' folders
    processed_folders = list(root_path.rglob("images_processed"))

    if not processed_folders:
        print("No 'images_processed' folders found.")
        return

    print(f"Found {len(processed_folders)} paper folders. Starting extraction...")

    # 3. Processing Loop
    for folder in processed_folders:
        paper_name = folder.parent.name

        # Initialize Inner Dictionary: { "image.png": embedding }
        master_dictionary[paper_name] = {}

        images = list(folder.glob("*.png")) + list(folder.glob("*.jpg"))

        # We use 'leave=False' so the progress bar clears after each paper, keeping output clean
        for img_path in tqdm(images, desc=f"Processing {paper_name}", leave=False):
            try:
                # Load & Preprocess
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    input_tensor = transform(img).unsqueeze(0).to(DEVICE)

                # Inference
                with torch.inference_mode():
                    outputs = model(input_tensor)
                    hidden_states = outputs.last_hidden_state

                    # EXTRACT GLOBAL EMBEDDING
                    # DINOv3 uses the [CLS] token at index 0
                    global_embedding = hidden_states[0, 0, :].cpu().numpy()

                # Store in inner dictionary
                # Key is filename (e.g., "88_row1_col1.png")
                master_dictionary[paper_name][img_path.name] = global_embedding

            except Exception as e:
                print(f"    Error processing {img_path.name}: {e}")

        # Print summary for this paper
        print(f"  Finished {paper_name}: {len(master_dictionary[paper_name])} images processed.")

    # 4. Save to Pickle
    print(f"\nSaving nested dictionary to {OUTPUT_PICKLE}...")
    try:
        with open(OUTPUT_PICKLE, 'wb') as f:
            pickle.dump(master_dictionary, f)
        print("Success.")
    except Exception as e:
        print(f"Error saving pickle: {e}")


if __name__ == "__main__":
    extract_dinov3_nested_dict()