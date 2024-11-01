# Neural Cellular Automata for Video Latent Representation

This project implements a Neural Cellular Automata (NCA) to learn a low-resolution latent representation of video frames. The NCA captures the spatio-temporal dynamics of input videos and can generate animations based on trained models.

## Project Structure

- `data/`: Contains input data such as videos, extracted frames, and example images.
- `src/`: Source code including model definitions, training scripts, and utilities.
- `scripts/`: Utility scripts for setting up and managing the project.
- `models/`: Directory to save trained models.
- `outputs/`: Generated outputs like videos and visualizations.

## Setup Instructions

1. **Clone the Repository** (if applicable):
    ```bash
    git clone <repository_url>
    cd nca_video_latent
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Download Example Input Image**:
    The setup script automatically downloads an example input image to `data/input_image.png`.

4. **Train the NCA Model**:
    ```bash
    python src/train.py --video path_to_video.mp4 --frames_dir data/frames --max_size 128 --num_epochs 1000 --learning_rate 1e-3 --model_save_path models/nca_model.pth
    ```

5. **Run Inference to Generate Video from Image**:
    ```bash
    python src/inference.py --model_path models/nca_model.pth --input_image data/input_image.png --output_video outputs/generated_video.mp4 --max_size 128 --num_steps 100
    ```

## Usage

- **Training**: Use `train.py` to train the NCA model on a video or a set of frames.
- **Inference**: Use `inference.py` to generate a video from a single input image using a trained model.
- **Utilities**: `utils.py` contains data preprocessing and utility functions.
- **Model Definition**: `nca_model.py` defines the NCA architecture.

## Notes

- Ensure that `ffmpeg` is installed on your system for video processing tasks.
- The project is structured for clarity and scalability, adhering to production-level data science standards.

## License

[MIT](LICENSE)

