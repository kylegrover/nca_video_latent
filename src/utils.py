# src/utils.py
import os
import cv2
import ffmpeg
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt

def extract_frames(video_path, output_dir, max_size=128):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    (
        ffmpeg
        .input(video_path)
        .filter('crop', 'min(iw,ih)', 'min(iw,ih)')  # Crop to a square
        .filter('scale', max_size, max_size)
        .output(os.path.join(output_dir, 'frame_%05d.png'))
        .run(overwrite_output=True)
    )

def load_frames(folder, max_size=128):
    frame_files = sorted([f for f in os.listdir(folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    frames = []
    for file in tqdm(frame_files, desc="Loading frames"):
        img_path = os.path.join(folder, file)
        img = Image.open(img_path).convert('RGB')
        img.thumbnail((max_size, max_size), resample=Image.Resampling.LANCZOS)
        frames.append(np.array(img))
    return np.array(frames) / 255.0  # Normalize to [0, 1]

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path, device='cuda'):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

def visualize_frames(frames, title='Frames', save_path=None):
    fig, axes = plt.subplots(1, len(frames), figsize=(15, 5))
    if len(frames) == 1:
        axes = [axes]
    for idx, frame in enumerate(frames):
        axes[idx].imshow(frame)
        axes[idx].axis('off')
    plt.suptitle(title)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close(fig)
