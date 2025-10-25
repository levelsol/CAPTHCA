#!/usr/bin/env python3
"""
RECAPTCHA Training Script - Optimized for 6GB RAM
"""

from ultralytics import YOLO
import torch

def train_recaptcha():
    print("=" * 60)
    print("RECAPTCHA TRAINING - 6GB RAM OPTIMIZED")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    model_name = 'yolov8n.pt'
    batch_size = 6
    workers = 3
    cache_mode = False
    
    print(f"Model: {model_name}")
    print(f"Batch size: {batch_size}")
    print(f"Workers: {workers}")
    print()
    
    model = YOLO(model_name)
    
    results = model.train(
        data='dataset/data.yaml',
        epochs=500,
        batch=batch_size,
        imgsz=640,
        device=device,
        workers=workers,
        cache=cache_mode,
        patience=100,
        save=True,
        save_period=25,
        optimizer='AdamW',
        lr0=0.003,
        lrf=0.0001,
        momentum=0.95,
        weight_decay=0.001,
        warmup_epochs=10,
        hsv_h=0.03,
        hsv_s=0.9,
        hsv_v=0.6,
        degrees=10,
        translate=0.25,
        scale=0.95,
        shear=3.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.2,
        copy_paste=0.5,
        auto_augment='randaugment',
        erasing=0.6,
        project='runs/train',
        name='recaptcha_6gb',
        exist_ok=True,
        verbose=True
    )
    
    print(f"\n✅ Training completed!")
    print(f"Best model: {results.save_dir}/weights/best.pt")
    return results

if __name__ == "__main__":
    train_recaptcha()
