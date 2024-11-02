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
        # img.thumbnail((max_size, max_size), resample=Image.Resampling.LANCZOS)
        frames.append(np.array(img, dtype=np.float32))
    return np.array(frames, dtype=np.float32) / 255.0  # Normalize to [0, 1]

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_model(model, path, device='cuda'):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

def visualize_frames(frames, title='Frames', save_path=None):
    # Ensure the frames are in the correct format and range
    processed_frames = []
    for frame in frames:
        if torch.is_tensor(frame):
            frame = frame.detach().cpu().numpy()
        
        # Ensure float32 type
        frame = frame.astype(np.float32)
        
        # Ensure values are in [0, 1]
        if frame.max() > 1.0 or frame.min() < 0.0:
            frame = np.clip(frame, 0.0, 1.0)
            
        processed_frames.append(frame)

    # Create visualization
    fig, axes = plt.subplots(1, len(processed_frames), figsize=(15, 5))
    if len(processed_frames) == 1:
        axes = [axes]
    
    for ax, frame in zip(axes, processed_frames):
        ax.imshow(frame)
        ax.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close(fig)