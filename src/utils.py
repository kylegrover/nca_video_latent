import os
import cv2
import ffmpeg
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

def extract_frames(video_path, output_dir, max_size=128):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    (
        ffmpeg
        .input(video_path)
        .output(os.path.join(output_dir, 'frame_%05d.png'))
        .run(overwrite_output=True)
    )

    # Resize frames
    frame_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.png')])
    for file in tqdm(frame_files, desc="Resizing frames"):
        img_path = os.path.join(output_dir, file)
        img = Image.open(img_path)
        # Update resampling filter
        if hasattr(Image, 'Resampling'):
            resample_filter = Image.Resampling.LANCZOS
        else:
            resample_filter = Image.ANTIALIAS  # For older Pillow versions
        img.thumbnail((max_size, max_size), resample=resample_filter)
        img.save(img_path)

def load_frames(folder, max_size=128):
    frame_files = sorted([f for f in os.listdir(folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    frames = []
    for file in tqdm(frame_files, desc="Loading frames"):
        img_path = os.path.join(folder, file)
        img = Image.open(img_path).convert('RGB')
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        frames.append(np.array(img))
    return np.array(frames) / 255.0  # Normalize to [0, 1]

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path, device='cuda'):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

def visualize_frames(frames, title='Frames'):
    fig, axes = plt.subplots(1, len(frames), figsize=(15, 5))
    if len(frames) == 1:
        axes = [axes]
    for idx, frame in enumerate(frames):
        axes[idx].imshow(frame)
        axes[idx].axis('off')
    plt.suptitle(title)
    plt.show()