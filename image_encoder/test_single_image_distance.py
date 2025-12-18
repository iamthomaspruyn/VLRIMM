import torch
import numpy as np
import pickle
import os
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoModel
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances

# --- Configuration ---
# 1. The query image
TARGET_IMAGE_PATH = "C:/Users/zkevi/Downloads/0a55e7c93f.png"

# 2. Database paths
DICT_PATH = "dinov3_nested_dict.pkl"
DATASET_ROOT = "./docling_output"

# 3. Search Settings
TOP_K = 10
BOTTOM_K = 10  # How many "least similar" images to show

# 4. Model Settings
MODEL_HF_ID = "facebook/dinov3-vits16plus-pretrain-lvd1689m"
DEVICE = "cpu"
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


def get_single_embedding(image_path):
    print(f"Loading model: {MODEL_HF_ID}...")
    try:
        model = AutoModel.from_pretrained(MODEL_HF_ID, trust_remote_code=True).to(DEVICE).eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    print(f"Processing target image: {image_path}")
    transform = get_transforms()

    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            input_tensor = transform(img).unsqueeze(0).to(DEVICE)

        with torch.inference_mode():
            outputs = model(input_tensor)
            embedding = outputs.last_hidden_state[0, 0, :].cpu().numpy()

        return embedding
    except Exception as e:
        print(f"Error processing image: {e}")
        return None


def load_database(pickle_path, root_path):
    print(f"Loading database from {pickle_path}...")
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    embeddings = []
    paths = []
    root = Path(root_path)

    for paper_name, img_dict in data.items():
        folder = root / paper_name / "images_processed"
        for img_name, emb in img_dict.items():
            full_path = folder / img_name
            embeddings.append(emb)
            paths.append(full_path)

    return np.array(embeddings), paths


def find_and_plot_matches(target_emb, db_embeddings, db_paths, top_k=10, bottom_k=10):
    print("Calculating similarities...")

    target_emb = target_emb.reshape(1, -1)
    scores = cosine_similarity(target_emb, db_embeddings)[0]

    # --- 1. Get Indices ---
    # Top K: Highest scores (Sort ascending -> take last K -> reverse to get Descending)
    top_indices = np.argsort(scores)[-top_k:][::-1]

    # Bottom K: Lowest scores (Sort ascending -> take first K)
    bottom_indices = np.argsort(scores)[:bottom_k]

    # --- 2. Visualization ---
    print(f"Plotting Top {top_k} and Bottom {bottom_k} matches...")

    # Create grid: 2 rows.
    # Columns = 1 (for target) + max(top_k, bottom_k)
    max_cols = max(top_k, bottom_k)
    total_cols = max_cols + 1

    fig, axes = plt.subplots(2, total_cols, figsize=(4 * total_cols, 8))

    # Helper to plot one image
    def plot_image_on_ax(ax, path, title, is_target=False, box_color=None):
        try:
            img = Image.open(path).convert("RGB")
            ax.imshow(img)
            ax.set_title(title, fontsize=8)

            # Label paper name
            if not is_target:
                paper_name = path.parent.parent.name
                ax.set_xlabel(paper_name, fontsize=7)

            ax.set_xticks([])
            ax.set_yticks([])

            if box_color:
                for spine in ax.spines.values():
                    spine.set_edgecolor(box_color)
                    spine.set_linewidth(3)
        except Exception:
            ax.text(0.5, 0.5, "Image Error", ha='center')
            ax.axis('off')

    # --- ROW 1: Target + Top K ---

    # 1A. Target Image (Row 1, Col 0)
    plot_image_on_ax(axes[0, 0], TARGET_IMAGE_PATH, "TARGET IMAGE", is_target=True, box_color='blue')

    # 1B. Top K Images
    for i, idx in enumerate(top_indices):
        col = i + 1
        if col < total_cols:
            score = scores[idx]
            path = db_paths[idx]
            title = f"Top #{i + 1}\nSim: {score:.4f}"
            plot_image_on_ax(axes[0, col], path, title, box_color='green' if score > 0.99 else None)

    # --- ROW 2: Info + Bottom K ---

    # 2A. Empty/Info Slot (Row 2, Col 0)
    axes[1, 0].axis('off')
    axes[1, 0].text(0.5, 0.5, "LEAST SIMILAR\n(Bottom K)", ha='center', fontweight='bold', fontsize=12)

    # 2B. Bottom K Images
    for i, idx in enumerate(bottom_indices):
        col = i + 1
        if col < total_cols:
            score = scores[idx]
            path = db_paths[idx]
            title = f"Bottom #{i + 1}\nSim: {score:.4f}"
            plot_image_on_ax(axes[1, col], path, title, box_color='red')

    plt.tight_layout()
    output_file = "similarity_results_top_bot.png"
    plt.savefig(output_file)
    print(f"Saved plot to {output_file}")
    plt.show()


if __name__ == "__main__":
    if not os.path.exists(TARGET_IMAGE_PATH):
        print(f"Error: Target image not found at {TARGET_IMAGE_PATH}")
        exit()

    target_embedding = get_single_embedding(TARGET_IMAGE_PATH)

    if target_embedding is not None:
        db_embeddings, db_paths = load_database(DICT_PATH, DATASET_ROOT)

        if len(db_embeddings) > 0:
            find_and_plot_matches(
                target_embedding,
                db_embeddings,
                db_paths,
                top_k=TOP_K,
                bottom_k=BOTTOM_K
            )
        else:
            print("Database is empty.")