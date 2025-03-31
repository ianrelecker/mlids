# Live AI-IDS: Real-time Network Intrusion Detection System

Live AI-IDS is a machine learning-based network intrusion detection system that captures and analyzes network traffic in real-time to detect potential attacks.

## Features

- **Real-time Network Traffic Analysis**: Captures and processes network packets as they occur
- **Deep Learning-based Detection**: Uses a Deep Concatenated CNN architecture for accurate attack classification
- **Multi-platform Support**: Works on macOS (including Apple Silicon) and Linux/Windows with CUDA support
- **Customizable Training**: Train on your own network traffic to detect specific attack patterns
- **Detailed Alerts**: Provides comprehensive information about detected threats

## System Requirements

- Python 3.8+
- PyTorch 1.13.0+
- Scapy 2.4.5+
- For GPU acceleration:
  - NVIDIA GPU with CUDA support, or
  - Apple Silicon Mac (M1/M2/M3) for MPS acceleration

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/live-ai-ids.git
   cd live-ai-ids
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   **Note for GPU users:**
   - For NVIDIA GPUs, install the appropriate CUDA-enabled PyTorch version:
     ```bash
     # Example for CUDA 11.7
     pip install torch==1.13.0+cu117 -f https://download.pytorch.org/whl/cu117/torch_stable.html
     ```
   - For Apple Silicon Macs, ensure you have PyTorch 2.0.0+ for optimal MPS support:
     ```bash
     pip install torch>=2.0.0
     ```

## Usage

### Training a Model

To train a model on your network traffic:

```bash
python train_model.py --duration 300 --save-data
```

This will:
1. Capture 5 minutes of normal network traffic
2. Extract features from the captured packets
3. Train a model on this data
4. Save the model to `models/final_model.pt`

#### Training Options

- `--duration SECONDS`: Duration to capture benign traffic (default: 300)
- `--attack-duration SECONDS`: Duration to capture attack traffic (default: 0)
- `--interface INTERFACE`: Network interface to capture from
- `--save-data`: Save captured data to CSV files
- `--load-data FILE`: Load data from a CSV file instead of capturing
- `--epochs N`: Number of training epochs
- `--batch-size N`: Batch size for training
- `--learning-rate RATE`: Learning rate for training

### Running Live Detection

To start real-time intrusion detection:

```bash
python live_ids.py
```

This will:
1. Load the trained model
2. Start capturing network traffic
3. Analyze packets in real-time
4. Alert when potential attacks are detected

### Testing the System

To run basic tests to verify system functionality:

```bash
python test_system.py
```

## GPU Acceleration

The system automatically detects and uses available GPU acceleration:

- On systems with NVIDIA GPUs, CUDA will be used if available
- On Apple Silicon Macs, Metal Performance Shaders (MPS) will be used if available
- If no GPU acceleration is available, the system will fall back to CPU

You can see which device is being used in the console output when training or running detection.

## Project Structure

- `live_ids.py`: Main script for real-time detection
- `train_model.py`: Script for training a new model
- `test_system.py`: Script for testing system components
- `src/`: Source code modules
  - `data_processor.py`: Packet capture and feature extraction
  - `model.py`: Neural network architecture
  - `trainer.py`: Model training functionality
  - `detector.py`: Real-time detection engine
  - `utils.py`: Utility functions
- `models/`: Saved models and related files
- `data/`: Training data
- `results/`: Training results and logs

## License

This project is licensed under the MIT License - see the LICENSE file for details.
