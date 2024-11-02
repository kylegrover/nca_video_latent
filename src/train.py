# src/train.py
import os
import argparse
import torch
import torch.optim as optim
from tqdm import tqdm

from nca_model import NCA
from utils import extract_frames, load_frames, save_model, visualize_frames

import torch.nn as nn
import piq  # For SSIM loss

import logging

def check_gpu_status():
    if torch.cuda.is_available():
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        print(f"Current CUDA Device: {torch.cuda.current_device()}")
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        return True
    else:
        print("CUDA is not available. Running on CPU.")
        return False

def train_nca(nca, frames, num_epochs=1000, learning_rate=1e-4, device='cuda'):
    # Check GPU status before training
    is_cuda_available = check_gpu_status()
    if not is_cuda_available and device == 'cuda':
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        device = 'cpu'
    
    print(f"Training on device: {device}")
    
    # Move model to device and print memory usage if using GPU
    nca = nca.to(device)
    if device == 'cuda':
        print(f"Initial GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    
    optimizer = optim.AdamW(nca.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)
    mse_criterion = nn.MSELoss()
    ssim_criterion = piq.SSIMLoss(data_range=1.0)
    
    # Configure logging
    os.makedirs('outputs', exist_ok=True)
    logging.basicConfig(filename='outputs/training.log', level=logging.INFO, 
                        format='%(asctime)s:%(levelname)s:%(message)s')
    
    # Convert frames to torch tensors
    frames_tensor = torch.tensor(frames).permute(0, 3, 1, 2).float().to(device)  # Shape: (T, C, H, W)
    
    for epoch in tqdm(range(num_epochs), desc="Training NCA"):
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        # Initialize hidden state with the first frame using encoder
        initial_frame = frames_tensor[0].unsqueeze(0)  # Shape: [1, C, H, W]
        hidden_state = nca.encoder(initial_frame)  # Shape: [1, hidden_channels, H, W]
        hidden_state = torch.clamp(hidden_state, 0.0, 1.0)
        
        # Print GPU memory usage during first epoch
        if epoch == 0 and device == 'cuda':
            print(f"GPU Memory after initialization: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        
        alpha = 1.0
        
        for t in range(frames_tensor.shape[0]):
            target = frames_tensor[t].unsqueeze(0)  # Shape: [1, C, H, W]
            output, hidden_state = nca(hidden_state)  # output: [1, C, H, W], hidden_state: [1, hidden_channels, H, W]
            
            # Compute loss
            mse_loss = mse_criterion(output, target)
            ssim_loss = ssim_criterion(output, target)
            total_loss = total_loss = alpha * mse_loss + beta * (1 - ssim_loss)  # Encourages higher SSIM
            epoch_loss += total_loss
            
            # Backpropagation
            total_loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(nca.parameters(), max_norm=1.0)
            # Detach hidden state to prevent backpropagating through time
            hidden_state = hidden_state.detach()
        
        optimizer.step()
        scheduler.step()
        epoch_loss /= frames_tensor.shape[0]
        
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss.item():.6f}")
            logging.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss.item():.6f}")
            # Generate and save sample frames
            with torch.no_grad():
                # Re-initialize hidden state
                hidden_state_sample = nca.encoder(initial_frame)
                hidden_state_sample = torch.clamp(hidden_state_sample, 0.0, 1.0)
                generated_frames = []
                for _ in range(frames_tensor.shape[0]):
                    output_sample, hidden_state_sample = nca(hidden_state_sample)
                    generated_frame = output_sample.squeeze(0).cpu().permute(1, 2, 0).numpy()
                    generated_frames.append(generated_frame)
            # Save visualization images for this epoch
            os.makedirs('outputs/training_progress', exist_ok=True)
            visualize_frames(
                [frames[0], frames[frames.shape[0]//2], frames[-1]],
                title=f'Original Frames - Epoch {epoch + 1}',
                save_path=f'outputs/training_progress/original_frames_epoch_{epoch + 1}.png'
            )
            visualize_frames(
                [generated_frames[0], generated_frames[len(generated_frames)//2], generated_frames[-1]],
                title=f'Generated Frames - Epoch {epoch + 1}',
                save_path=f'outputs/training_progress/generated_frames_epoch_{epoch + 1}.png'
            )
    
    print("Training completed.")
    logging.info("Training completed.")
    return nca

def main():
    parser = argparse.ArgumentParser(description="Neural Cellular Automata for Video Latent Representation")
    parser.add_argument('--video', type=str, default=None, help='Path to input video file')
    parser.add_argument('--frames_dir', type=str, default='data/frames', help='Directory containing frames')
    parser.add_argument('--max_size', type=int, default=128, help='Max width or height for frames')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--model_save_path', type=str, default='models/nca_model.pth', help='Path to save the trained model')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    args = parser.parse_args()

    if args.video:
        extract_frames(args.video, args.frames_dir, max_size=args.max_size)
    frames = load_frames(args.frames_dir, max_size=args.max_size)  # Shape: (num_frames, H, W, 3)

    nca = NCA(input_channels=3, hidden_channels=32, num_steps=10, num_blocks=3)
    trained_nca = train_nca(nca, frames, num_epochs=args.num_epochs, learning_rate=args.learning_rate, device=args.device)
    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
    save_model(trained_nca, args.model_save_path)

    # Final Visualization
    generated_frames = []
    with torch.no_grad():
        initial_frame = torch.tensor(frames[0]).permute(2, 0, 1).unsqueeze(0).float().to(args.device)  # [1, 3, H, W]
        hidden_state = nca.encoder(initial_frame)
        hidden_state = torch.clamp(hidden_state, 0.0, 1.0)
        for t in range(frames.shape[0]):
            output, hidden_state = nca(hidden_state)
            generated_frame = output.squeeze(0).cpu().permute(1, 2, 0).numpy()
            generated_frames.append(generated_frame)

    # Save final visualization images
    os.makedirs('outputs', exist_ok=True)
    visualize_frames(
        [frames[0], frames[frames.shape[0]//2], frames[-1]],
        title='Original Frames',
        save_path='outputs/original_frames_final.png'
    )
    visualize_frames(
        [generated_frames[0], generated_frames[len(generated_frames)//2], generated_frames[-1]],
        title='Generated Frames',
        save_path='outputs/generated_frames_final.png'
    )

if __name__ == '__main__':
    main()
