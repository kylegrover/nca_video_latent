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
from torch.amp import autocast, GradScaler
import torch.profiler

def check_gpu_status():
    if torch.cuda.is_available():
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        print(f"Current CUDA Device: {torch.cuda.current_device()}")
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        return True
    else:
        print("CUDA is not available. Running on CPU.")
        return False

def train_nca(nca, frames, num_epochs=1000, learning_rate=1e-4, device='cuda', batch_size=4):
    is_cuda_available = check_gpu_status()
    if not is_cuda_available and device == 'cuda':
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        device = 'cpu'
    
    print(f"Training on device: {device}")
    
    # Enable cudnn benchmarking
    torch.backends.cudnn.benchmark = True
    
    # Move model to device
    nca = nca.to(device)
    
    # Initialize mixed precision training with fixed deprecation warnings
    scaler = GradScaler('cuda')
    
    optimizer = optim.AdamW(nca.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    mse_criterion = nn.MSELoss()
    ssim_criterion = piq.SSIMLoss(data_range=1.0)
    
    # Convert frames to torch tensors and move to device
    frames_tensor = torch.tensor(frames, device=device).permute(0, 3, 1, 2).float()
    
    # Create batches of sequential frames
    num_frames = frames_tensor.shape[0]
    sequence_length = batch_size
    num_complete_sequences = (num_frames - sequence_length + 1)
    sequence_starts = list(range(0, num_complete_sequences))
    
    # Print debug info
    print(f"Number of frames: {num_frames}")
    print(f"Sequence length: {sequence_length}")
    print(f"Number of sequences: {num_complete_sequences}")
    print(f"Initial GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    # Profile one iteration to debug performance
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        # Run one training iteration for profiling
        sequence = frames_tensor[:sequence_length]
        with autocast(device_type='cuda'):
            initial_frame = sequence[0].unsqueeze(0)
            hidden_state = nca.encoder(initial_frame)
            output, hidden_state = nca(hidden_state)
            
            mse_loss = mse_criterion(output, sequence[0].unsqueeze(0))
            total_loss = mse_loss
            
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        prof.step()
    
    print("\nProfiler Output:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    # Main training loop
    for epoch in tqdm(range(num_epochs), desc="Training NCA"):
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        
        # Shuffle sequence starts at the beginning of each epoch
        sequence_starts_shuffled = torch.randperm(len(sequence_starts))
        
        # Process all sequences in the epoch
        for idx in sequence_starts_shuffled:
            start_idx = sequence_starts[idx]
            sequence = frames_tensor[start_idx:start_idx + sequence_length]
            
            # Run forward pass with mixed precision
            with autocast('cuda'):
                initial_frame = sequence[0].unsqueeze(0)
                hidden_state = nca.encoder(initial_frame)
                hidden_state = torch.clamp(hidden_state, 0.0, 1.0)
                
                sequence_loss = 0.0
                
                # Process each frame in the sequence
                for t in range(sequence_length):
                    target = sequence[t].unsqueeze(0)
                    output, hidden_state = nca(hidden_state)
                    
                    mse_loss = mse_criterion(output, target)
                    ssim_loss = ssim_criterion(output, target)
                    total_loss = mse_loss + 0.3 * (1 - ssim_loss)
                    sequence_loss += total_loss
                
                sequence_loss /= sequence_length
            
            # Scale loss and backpropagate
            scaler.scale(sequence_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(nca.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            optimizer.zero_grad(set_to_none=True)
            epoch_loss += sequence_loss.item()
            
            # Print GPU utilization every 100 sequences
            if idx % 100 == 0:
                print(f"\nGPU Utilization: {get_gpu_utilization()}%")
                print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        
        scheduler.step()
        epoch_loss /= len(sequence_starts)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"\nEpoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.6f}")
            save_visualization(nca, frames_tensor, frames, epoch, device)
    
    return nca

def get_gpu_utilization():
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return info.gpu
    except:
        return "N/A"

def save_visualization(nca, frames_tensor, frames, epoch, device):
    with torch.no_grad(), autocast(device_type='cuda'):
        initial_frame = frames_tensor[0].unsqueeze(0)
        hidden_state = nca.encoder(initial_frame)
        hidden_state = torch.clamp(hidden_state, 0.0, 1.0)
        generated_frames = []
        
        for _ in range(frames_tensor.shape[0]):
            output, hidden_state = nca(hidden_state)
            generated_frame = output.squeeze(0).cpu().permute(1, 2, 0).numpy()
            generated_frames.append(generated_frame)
        
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

def main():
    parser = argparse.ArgumentParser(description="Neural Cellular Automata for Video Latent Representation")
    parser.add_argument('--video', type=str, default=None, help='Path to input video file')
    parser.add_argument('--frames_dir', type=str, default='data/frames', help='Directory containing frames')
    parser.add_argument('--max_size', type=int, default=128, help='Max width or height for frames')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--model_save_path', type=str, default='models/nca_model.pth', help='Path to save the trained model')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for sequence processing')
    args = parser.parse_args()

    if args.video:
        extract_frames(args.video, args.frames_dir, max_size=args.max_size)
    frames = load_frames(args.frames_dir, max_size=args.max_size)  # Shape: (num_frames, H, W, 3)

    nca = NCA(input_channels=3, hidden_channels=32, num_steps=10, num_blocks=3)
    trained_nca = train_nca(nca, frames, num_epochs=args.num_epochs, learning_rate=args.learning_rate, device=args.device, batch_size=args.batch_size)
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
