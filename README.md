source# Live AI-IDS: Real-time Network Intrusion Detection System

A machine learning-based network intrusion detection system that captures and analyzes live network traffic to detect malicious activity in real-time.

## Features

- **Live Network Monitoring**: Captures packets from any network interface in real-time
- **Machine Learning Detection**: Uses a deep learning model to classify network traffic
- **Known Attack Detection**: Identifies common attack patterns
- **Unknown Attack Detection**: Detects anomalous traffic that doesn't match known patterns
- **Model Training**: Includes tools to train custom models on your network traffic
- **Detailed Alerts**: Provides information about detected threats

## Requirements

- Python 3.7+
- PyTorch 1.13.0+
- Scapy (for packet capture and analysis)
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn (for visualization)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/live-ai-ids.git
   cd live-ai-ids
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. (Optional) Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

## Usage

### Training a Model

Before you can detect attacks, you need to train a model on your network traffic:

```
python train_model.py --duration 300 --save-data
```

This will:
1. Capture 5 minutes (300 seconds) of normal network traffic
2. Train a model to recognize this traffic as benign
3. Save the model for later use

For more advanced training, you can also capture attack traffic:

```
python train_model.py --duration 300 --attack-duration 120 --save-data
```

This will capture 5 minutes of normal traffic and 2 minutes of attack traffic.

### Running Live Detection

Once you have a trained model, you can start live detection:

```
python live_ids.py
```

This will:
1. Load the trained model
2. Start capturing live network traffic
3. Analyze each packet for signs of malicious activity
4. Alert you when attacks are detected

### Command-line Options

#### Training Options

```
python train_model.py --help
```

- `--interface`, `-i`: Network interface to capture from
- `--duration`, `-d`: Duration in seconds to capture benign traffic
- `--attack-duration`: Duration to capture attack traffic
- `--save-data`, `-s`: Save captured data to CSV files
- `--load-data`, `-l`: Load data from CSV instead of capturing
- `--epochs`, `-e`: Number of training epochs
- `--list-interfaces`: List available network interfaces

#### Detection Options

```
python live_ids.py --help
```

- `--interface`, `-i`: Network interface to capture from
- `--model`, `-m`: Path to the trained model
- `--threshold`, `-t`: Confidence threshold for attack detection
- `--duration`, `-d`: Duration to run detection (0 for indefinite)
- `--verbose`, `-v`: Enable verbose output
- `--list-interfaces`: List available network interfaces

## How It Works

### Architecture

The system consists of several components:

1. **Packet Capture**: Uses Scapy to capture live network packets
2. **Feature Extraction**: Extracts relevant features from network packets
3. **Deep Learning Model**: A Deep Concatenated CNN classifies traffic
4. **Open-Set Detection**: Identifies unknown attack patterns
5. **Alert System**: Notifies when attacks are detected

### Machine Learning Model

The system uses a Deep Concatenated CNN architecture that:
- Transforms network features into a 2D image-like format
- Uses multiple convolutional blocks with skip connections
- Combines features at different levels for better pattern recognition
- Outputs classification probabilities for different traffic types

### Detection Approach

The system uses two complementary detection methods:
1. **Closed-set detection**: Identifies known attack types the model was trained on
2. **Open-set detection**: Identifies anomalous traffic that doesn't match known patterns

## Customization

### Configuration

You can customize the system by editing the `config.json` file:

```json
{
  "interface": null,
  "max_packets": 10000,
  "buffer_size": 1000,
  "samples_per_class": 5000,
  "batch_size": 256,
  "learning_rate": 0.001,
  "epochs": 50,
  "confidence_threshold": 0.8,
  "detection_batch_size": 32,
  "detection_interval": 1.0
}
```

### Adding Custom Attack Types

To train the system on specific attack types:

1. Capture normal traffic:
   ```
   python train_model.py --duration 300 --save-data
   ```

2. Capture each attack type separately:
   ```
   python train_model.py --duration 0 --attack-duration 120 --attack-label "DOS" --save-data
   ```

3. Combine the data and train a model:
   ```
   python train_model.py --load-data data/combined_traffic.csv
   ```

## Limitations

- Requires elevated privileges to capture network packets
- Performance depends on the quality of training data
- May generate false positives for unusual but benign traffic
- Resource usage increases with network traffic volume

## Troubleshooting

### Common Issues

1. **Permission errors when capturing packets**:
   - Run the script with administrator/root privileges
   - On Linux: `sudo python live_ids.py`
   - On Windows: Run Command Prompt as Administrator

2. **No packets captured**:
   - Verify you're using the correct network interface
   - List available interfaces: `python live_ids.py --list-interfaces`

3. **High false positive rate**:
   - Retrain the model with more diverse benign traffic
   - Adjust the confidence threshold: `python live_ids.py --threshold 0.9`

4. **Model training fails**:
   - Ensure you've captured enough packets
   - Try reducing batch size: `python train_model.py --batch-size 64`

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This project was inspired by the AIS-NIDS system and builds upon concepts from:
- "AIS-NIDS: An intelligent and self-sustaining network intrusion detection system"
- Deep learning approaches for network traffic classification
- Anomaly detection in network security
