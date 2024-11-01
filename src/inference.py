import os
import argparse
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm

from nca_model import NCA
from utils import load_model, visualize_frames

import cv2

def generate_video(nca, input_image, num_steps, device='cuda'):
    nca.to(device)
    nca.eval()
    
    # Initialize state with input image
    input_tensor = torch.tensor(input_image).permute(2, 0, 1).unsqueeze(0).float().to(device)  # Shape: (1, C, H, W)
    
    state = torch.zeros_like(input_tensor).to(device)
    state = state + input_tensor  # Initialize state with input image
    
    generated_frames = []
    with torch.no_grad():
        for step in tqdm(range(num_steps), desc="Generating frames"):
            state = nca(state)
            frame = state.squeeze(0).cpu().permute(1, 2, 0).numpy()
            frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
            generated_frames.append(frame)
    
    return generated_frames

def save_video(frames, output_path, fps=24):
    height, width, layers = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(frame_bgr)
    
    video.release()

def main():
    parser = argparse.ArgumentParser(description="Inference for Neural Cellular Automata")
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained NCA model')
    parser.add_argument('--input_image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output_video', type=str, default='outputs/generated_video.mp4', help='Path to save the generated video')
    parser.add_argument('--max_size', type=int, default=128, help='Max width or height for input image')
    parser.add_argument('--num_steps', type=int, default=100, help='Number of steps to run the NCA')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    args = parser.parse_args()
    
    # Load and preprocess input image
    img = Image.open(args.input_image).convert('RGB')
    img.thumbnail((args.max_size, args.max_size), Image.ANTIALIAS)
    input_image = np.array(img) / 255.0  # Normalize to [0, 1]
    
    # Initialize model
    nca = NCA(input_channels=3, hidden_channels=16, num_steps=10)
    load_model(nca, args.model_path, device=args.device)
    
    # Generate frames
    generated_frames = generate_video(nca, input_image, args.num_steps, device=args.device)
    
    # Save video
    os.makedirs(os.path.dirname(args.output_video), exist_ok=True)
    save_video(generated_frames, args.output_video)
    
    print(f"Generated video saved to {args.output_video}")

if __name__ == '__main__':
    main()
