import os
import argparse
import torch
import torch.optim as optim
from tqdm import tqdm

from nca_model import NCA
from utils import extract_frames, load_frames, save_model

import torch.nn as nn

def train_nca(nca, frames, num_epochs=1000, learning_rate=1e-3, device='cuda'):
    nca.to(device)
    optimizer = optim.Adam(nca.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Convert frames to torch tensors
    frames_tensor = torch.tensor(frames).permute(0, 3, 1, 2).float().to(device)  # Shape: (T, C, H, W)
    
    for epoch in tqdm(range(num_epochs), desc="Training NCA"):
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        # Initialize the NCA state (could be random or based on the first frame)
        state = torch.zeros_like(frames_tensor[0]).to(device)
        
        for t in range(frames_tensor.shape[0]):
            target = frames_tensor[t]
            output = nca(state)
            loss = criterion(output, target)
            epoch_loss += loss
            loss.backward()
            # Update state
            state = output.detach()
        
        optimizer.step()
        epoch_loss /= frames_tensor.shape[0]
        
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss.item()}")
    
    print("Training completed.")
    return nca

def main():
    parser = argparse.ArgumentParser(description="Neural Cellular Automata for Video Latent Representation")
    parser.add_argument('--video', type=str, default=None, help='Path to input video file')
    parser.add_argument('--frames_dir', type=str, default='data/frames', help='Directory containing frames')
    parser.add_argument('--max_size', type=int, default=128, help='Max width or height for frames')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--model_save_path', type=str, default='models/nca_model.pth', help='Path to save the trained model')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    args = parser.parse_args()

    if args.video:
        extract_frames(args.video, args.frames_dir, max_size=args.max_size)
    frames = load_frames(args.frames_dir, max_size=args.max_size)  # Shape: (num_frames, H, W, 3)

    nca = NCA(input_channels=3, hidden_channels=16, num_steps=10)
    trained_nca = train_nca(nca, frames, num_epochs=args.num_epochs, learning_rate=args.learning_rate, device=args.device)
    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
    save_model(trained_nca, args.model_save_path)

    # Visualize some results
    from utils import visualize_frames
    state = torch.zeros(1, 3, frames.shape[1], frames.shape[2]).to(args.device)
    generated_frames = []
    with torch.no_grad():
        for t in range(frames.shape[0]):
            state = trained_nca(state)
            generated_frame = state.squeeze(0).cpu().permute(1, 2, 0).numpy()
            generated_frames.append(generated_frame)
    
    # Save visualization images
    os.makedirs('outputs', exist_ok=True)
    visualize_frames([frames[0], frames[frames.shape[0]//2], frames[-1]], title='Original Frames')
    visualize_frames([generated_frames[0], generated_frames[len(generated_frames)//2], generated_frames[-1]], title='Generated Frames')

if __name__ == '__main__':
    main()
