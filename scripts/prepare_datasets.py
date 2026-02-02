"""
Dataset Preparation Pipeline
Converts raw VQA data into conversation-format JSON files for three tasks:
1. Simple VQA
2. VQA with Grounding
3. Comparison Dataset
"""

import os
import json
import glob
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# Base paths
BASE_DIR = Path(r"D:\WeaklySupervised_MultiTaskLearning")
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "formatted"
IMAGES_DIR = DATA_DIR / "images"

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)

# System prompts
SYSTEM_PROMPTS = {
    "simple_vqa": "You are an expert Gastroenterologist. Answer the question about this endoscopy image in plain text.",
    "grounding_vqa": "You are an expert Gastroenterologist. Answer the question and ground the findings with bounding boxes.",
    "comparison": "You are an expert Gastroenterologist who compares two endoscopy images."
}


def prepare_simple_vqa():
    """Task 1: Simple VQA format from parquet files - outputs train, test, and combined."""
    print("\n=== Task 1: Simple VQA ===")
    
    vqa_dir = DATA_DIR / "vqa_subset"
    
    # Process each split separately
    splits = {
        "train": vqa_dir / "vqa_50k_train.parquet",
        "test": vqa_dir / "vqa_50k_test.parquet",
        "combined": vqa_dir / "vqa_50k_combined.parquet"
    }
    
    all_results = {}
    
    for split_name, pq_file in tqdm(splits.items(), desc="Processing splits"):
        if not pq_file.exists():
            print(f"Warning: {pq_file} not found, skipping {split_name}")
            continue
            
        df = pd.read_parquet(pq_file)
        samples = []
        
        for _, row in df.iterrows():
            img_id = row['img_id']
            question = row['question']
            answer = row['answer']
            
            # Build image path (absolute for checking, relative for output)
            img_path_abs = str(IMAGES_DIR / f"{img_id}.jpg")
            img_path_rel = f"/data/images/{img_id}.jpg"
            
            # Skip if image doesn't exist
            if not os.path.exists(img_path_abs):
                continue
            
            sample = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPTS["simple_vqa"]},
                    {"role": "user", "content": f"<image>{question}"},
                    {"role": "assistant", "content": answer}
                ],
                "images": [img_path_rel]
            }
            samples.append(sample)
        
        # Save output for this split
        output_path = OUTPUT_DIR / f"simple_vqa_{split_name}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(samples)} samples to {output_path}")
        all_results[split_name] = samples
    
    return all_results


def get_bbox_from_mask(mask_path: str, img_width: int = 512, img_height: int = 512):
    """Compute bounding box from a mask image and normalize to 0-1000 scale."""
    mask = np.array(Image.open(mask_path).convert('L'))
    
    # Find non-zero pixels
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)
    
    if not rows.any() or not cols.any():
        return None
    
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    
    # Normalize to 0-1000 scale
    h, w = mask.shape
    x1_norm = int(x1 * 1000 / w)
    y1_norm = int(y1 * 1000 / h)
    x2_norm = int(x2 * 1000 / w)
    y2_norm = int(y2 * 1000 / h)
    
    return (x1_norm, y1_norm, x2_norm, y2_norm)


def get_bbox_from_gradcam(img_id: str, class_name: str, img_path: str):
    """Get bounding box from gradcam_masks bbox_data.json."""
    # Map CSV filename patterns to gradcam subfolder names
    class_mapping = {
        "polyps": "polyp",
        "z-line": "z_line",
        "ulcerative_colitis": "ulcerative_colitis",
        "oesophagitis": "oesophagitis",
        "oesophatigis": "oesophagitis"  # Handle typo in CSV filename
    }
    
    folder_name = class_mapping.get(class_name, class_name)
    bbox_path = DATA_DIR / "gradcam_masks" / folder_name / img_id / "bbox_data.json"
    
    if not bbox_path.exists():
        return None
    
    with open(bbox_path, 'r') as f:
        data = json.load(f)
    
    if not data.get('regions'):
        return None
    
    # Get first region's bbox: [x, y, width, height]
    bbox = data['regions'][0]['bbox']
    x, y, w, h = bbox
    
    # Get actual image dimensions for proper normalization
    try:
        img = Image.open(img_path)
        img_width, img_height = img.size
    except:
        # Fallback to common size if image can't be read
        img_width, img_height = 512, 512
    
    # Convert to (x1, y1, x2, y2) and normalize to 0-1000
    x1_norm = int(x * 1000 / img_width)
    y1_norm = int(y * 1000 / img_height)
    x2_norm = int((x + w) * 1000 / img_width)
    y2_norm = int((y + h) * 1000 / img_height)
    
    # Clamp to 0-1000 range
    x1_norm = max(0, min(1000, x1_norm))
    y1_norm = max(0, min(1000, y1_norm))
    x2_norm = max(0, min(1000, x2_norm))
    y2_norm = max(0, min(1000, y2_norm))
    
    return (x1_norm, y1_norm, x2_norm, y2_norm)


def prepare_grounding_vqa():
    """Task 2: VQA with Grounding format from CSV files."""
    print("\n=== Task 2: VQA with Grounding ===")
    
    grounding_dir = DATA_DIR / "grounding_vqa"
    csv_files = list(grounding_dir.glob("*.csv"))
    
    all_samples = []
    stats = {"total": 0, "with_bbox": 0, "missing_bbox": 0}
    
    for csv_file in tqdm(csv_files, desc="Processing grounding CSVs"):
        df = pd.read_csv(csv_file)
        
        # Determine class from filename
        filename = csv_file.stem  # e.g., "polyps_mask_phrases"
        class_name = filename.replace("_mask_phrases", "")
        
        for _, row in df.iterrows():
            stats["total"] += 1
            img_id = row['img_id']
            question = row['question']
            answer = row['answer']
            
            # Build image path (absolute for checking, relative for output)
            img_path_abs = str(IMAGES_DIR / f"{img_id}.jpg")
            img_path_rel = f"/data/images/{img_id}.jpg"
            if not os.path.exists(img_path_abs):
                continue
            
            # Get bounding box
            if class_name == "z_line" and 'mask_id' in row:
                # Use pseudo_masks for z_line
                mask_id = row['mask_id']
                mask_path = DATA_DIR / "pseudo_masks" / f"{mask_id}.jpg"
                if mask_path.exists():
                    bbox = get_bbox_from_mask(str(mask_path))
                else:
                    # Try gradcam as fallback
                    bbox = get_bbox_from_gradcam(img_id, class_name, img_path_abs)
            else:
                # Use gradcam_masks for other classes
                bbox = get_bbox_from_gradcam(img_id, class_name, img_path_abs)
            
            if bbox is None:
                stats["missing_bbox"] += 1
                continue
            
            stats["with_bbox"] += 1
            
            # Format answer with grounding
            x1, y1, x2, y2 = bbox
            grounded_answer = f"Finding: {answer}\nEvidence: <|box_start|>({x1},{y1}),({x2},{y2})<|box_end|>"
            
            sample = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPTS["grounding_vqa"]},
                    {"role": "user", "content": f"<image>{question}"},
                    {"role": "assistant", "content": grounded_answer}
                ],
                "images": [img_path_rel]
            }
            all_samples.append(sample)
    
    # Save output
    output_path = OUTPUT_DIR / "grounding_vqa.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(all_samples)} samples to {output_path}")
    print(f"Stats: {stats}")
    return all_samples


def prepare_comparison():
    """Task 3: Comparison dataset format."""
    print("\n=== Task 3: Comparison Dataset ===")
    
    comparison_file = DATA_DIR / "comparison_dataset.json"
    
    with open(comparison_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Access the pairs array from the JSON structure
    pairs = data.get('pairs', [])
    all_samples = []
    
    for item in tqdm(pairs, desc="Processing comparison pairs"):
        img_a = item['image_a']
        img_b = item['image_b']
        description = item['description']
        
        # Build image paths (absolute for checking, relative for output)
        img_a_path_abs = str(IMAGES_DIR / f"{img_a}.jpg")
        img_b_path_abs = str(IMAGES_DIR / f"{img_b}.jpg")
        img_a_path_rel = f"/data/images/{img_a}.jpg"
        img_b_path_rel = f"/data/images/{img_b}.jpg"
        
        # Skip if images don't exist
        if not os.path.exists(img_a_path_abs) or not os.path.exists(img_b_path_abs):
            continue
        
        sample = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPTS["comparison"]},
                {"role": "user", "content": "<image><image>What is the difference between Image A (first image) and Image B (second image)?"},
                {"role": "assistant", "content": description}
            ],
            "images": [img_a_path_rel, img_b_path_rel]
        }
        all_samples.append(sample)
    
    # Save output
    output_path = OUTPUT_DIR / "comparison.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(all_samples)} samples to {output_path}")
    return all_samples


def main():
    print("=" * 60)
    print("Dataset Preparation Pipeline")
    print("=" * 60)
    
    # Run all three tasks
    simple_vqa = prepare_simple_vqa()
    grounding_vqa = prepare_grounding_vqa()
    comparison = prepare_comparison()
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("Simple VQA:")
    for split, samples in simple_vqa.items():
        print(f"  - {split}: {len(samples)} samples")
    print(f"Grounding VQA: {len(grounding_vqa)} samples")
    print(f"Comparison: {len(comparison)} samples")
    print(f"\nOutput directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
