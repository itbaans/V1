"""
MS-Swift Training Script for QwenVL-2B on Custom Gastroenterology VQA Dataset

This script trains Qwen2-VL-2B-Instruct model using LoRA on combined datasets:
- Simple VQA (train split)
- Grounding VQA  
- Comparison dataset

Usage:
    python train_qwenvl.py [--config config.yaml]
"""

import os
import subprocess
import argparse
from pathlib import Path


def get_default_config():
    """Default training configuration."""
    return {
        # Model
        "model": "Qwen/Qwen2-VL-2B-Instruct",
        "torch_dtype": "bfloat16",
        
        # Datasets (relative paths)
        "datasets": [
            "data/formatted/simple_vqa_train.json",
            "data/formatted/grounding_vqa.json",
            "data/formatted/comparison.json"
        ],
        
        # LoRA config
        "tuner_type": "lora",
        "lora_rank": 8,
        "lora_alpha": 32,
        "target_modules": "all-linear",
        
        # Training
        "num_train_epochs": 3,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate": 1e-4,
        "warmup_ratio": 0.05,
        "max_length": 2048,
        
        # Logging/Saving
        "logging_steps": 10,
        "eval_steps": 100,
        "save_steps": 500,
        "save_total_limit": 3,
        "output_dir": "output/qwen2vl_gastro_vqa",
        
        # Performance
        "gradient_checkpointing": True,
        "dataloader_num_workers": 4,
        
        # Source
        "use_hf": True,  # Use HuggingFace instead of ModelScope
    }


def build_swift_command(config: dict) -> list:
    """Build the swift sft command from config."""
    cmd = ["swift", "sft"]
    
    # Model
    cmd.extend(["--model", config["model"]])
    cmd.extend(["--torch_dtype", config["torch_dtype"]])
    
    # Datasets
    cmd.append("--dataset")
    cmd.extend(config["datasets"])
    
    # LoRA
    cmd.extend(["--tuner_type", config["tuner_type"]])
    cmd.extend(["--lora_rank", str(config["lora_rank"])])
    cmd.extend(["--lora_alpha", str(config["lora_alpha"])])
    cmd.extend(["--target_modules", config["target_modules"]])
    
    # Training
    cmd.extend(["--num_train_epochs", str(config["num_train_epochs"])])
    cmd.extend(["--per_device_train_batch_size", str(config["per_device_train_batch_size"])])
    cmd.extend(["--per_device_eval_batch_size", str(config["per_device_eval_batch_size"])])
    cmd.extend(["--gradient_accumulation_steps", str(config["gradient_accumulation_steps"])])
    cmd.extend(["--learning_rate", str(config["learning_rate"])])
    cmd.extend(["--warmup_ratio", str(config["warmup_ratio"])])
    cmd.extend(["--max_length", str(config["max_length"])])
    
    # Logging/Saving
    cmd.extend(["--logging_steps", str(config["logging_steps"])])
    cmd.extend(["--eval_steps", str(config["eval_steps"])])
    cmd.extend(["--save_steps", str(config["save_steps"])])
    cmd.extend(["--save_total_limit", str(config["save_total_limit"])])
    cmd.extend(["--output_dir", config["output_dir"]])
    
    # Performance
    if config["gradient_checkpointing"]:
        cmd.extend(["--gradient_checkpointing", "true"])
    cmd.extend(["--dataloader_num_workers", str(config["dataloader_num_workers"])])
    
    # Source
    if config["use_hf"]:
        cmd.extend(["--use_hf", "true"])
    
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Train QwenVL-2B on gastroenterology VQA dataset")
    parser.add_argument("--dry-run", action="store_true", help="Print command without running")
    parser.add_argument("--gpu", type=str, default="0", help="GPU device ID(s)")
    args = parser.parse_args()
    
    # Get config
    config = get_default_config()
    
    # Build command
    cmd = build_swift_command(config)
    
    # Set environment
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    print("=" * 60)
    print("MS-Swift Training for QwenVL-2B")
    print("=" * 60)
    print(f"\nModel: {config['model']}")
    print(f"Datasets: {config['datasets']}")
    print(f"Output: {config['output_dir']}")
    print(f"GPU: {args.gpu}")
    print(f"\nCommand:\n{' '.join(cmd)}")
    print("=" * 60)
    
    if args.dry_run:
        print("\n[DRY RUN] Command not executed.")
        return
    
    # Run training
    print("\nStarting training...")
    subprocess.run(cmd, env=env)


if __name__ == "__main__":
    main()
