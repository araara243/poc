# Malaysian Sign Language (MSL) Recognition System

A real-time Malaysian Sign Language recognition system using multi-stream Transformer architecture with MediaPipe for feature extraction and ONNX for optimized inference.

## 🎯 Overview

This project implements a deep learning-based system capable of recognizing 46 different Malaysian Sign Language (MSL) gestures in real-time. The system uses a novel multi-stream Transformer architecture that processes different semantic subvectors of hand and pose data independently before combining them for final classification.

### Key Features

- **Real-time Recognition**: Live webcam-based sign language detection
- **Multi-Stream Architecture**: Processes 7 semantic subvectors independently
- **High Accuracy**: Achieves 99.3% validation accuracy on 46 MSL gestures
- **Optimized Inference**: ONNX-based deployment for fast real-time performance
- **Interactive Dashboard**: Streamlit-based web interface for visualization
- **Comprehensive Logging**: Detailed training reports and TensorBoard integration

## 🏗️ Architecture

### Multi-Stream Transformer Model

The system uses a sophisticated multi-stream architecture that splits input features into 7 semantic subvectors:

| Subvector | Features | Description |
|-----------|----------|-------------|
| `location_l_hand` | 63 | Left hand 3D coordinates (21 landmarks × 3) |
| `location_r_hand` | 63 | Right hand 3D coordinates (21 landmarks × 3) |
| `location_pose` | 48 | Upper body pose landmarks (16 × 3) |
| `handshape_l` | 210 | Left hand geometric features (angles, distances) |
| `handshape_r` | 210 | Right hand geometric features (angles, distances) |
| `palm_orientation` | 200 | Palm orientation vectors (100 per hand) |
| `movement` | 126 | Hand movement velocities (63 per hand) |

**Total Features**: 920 per frame

### Model Specifications

- **Architecture**: Multi-Stream Transformer
- **Sequence Length**: 30 frames (~1 second at 30 FPS)
- **Transformer Blocks**: 2 per stream
- **Attention Heads**: 4 per block
- **Hidden Dimension**: 64
- **Feed-forward Dimension**: 128
- **Dropout**: 0.2
- **Optimizer**: Nadam (learning rate: 0.0001)

## 📊 Dataset

### Supported Gestures (46 classes)

The system recognizes the following Malaysian Sign Language gestures:

**Family & People**: abang, adik lelaki, adik perempuan, ayah, emak, kakak, dia, saya, orang, kawan, jururawat, doktor

**Medical Symptoms**: asma, batuk, cirit-birit, demam, luka, muntah, sakit, sakit kepala, pening, pengsan, pekak, selsema

**Medical Terms**: hospital, klinik, ubat, vitamin, racun

**Time**: pagi, tengahhari, petang, malam, dari pagi hingga malam

**Actions & Objects**: datang, mahu, makan, tolong, terima kasih, maaf (sorry), langgar, kereta, lori, kaki, kerusi roda

**General**: hello

### Data Format

- **Total Samples**: 1,380 sequences
- **Training Split**: 1,242 sequences (90%)
- **Test Split**: 138 sequences (10%)
- **Sequence Length**: 30 frames per gesture
- **Feature Extraction**: MediaPipe Holistic (pose, hands, face)
- **Input Shape**: `(sequence_length, 920_features)`

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended) or CPU
- Webcam for real-time recognition

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/araara243/poc.git
cd poc
```

2. **Create virtual environment**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download pre-trained model** (if not included)
   - Place `bigger.onnx` in the `model/` directory

5. **Prepare data directory**
   - Place your preprocessed `.npy` files in the `data/` directory
   - Each gesture should have its own subdirectory

### Required Packages

```
tensorflow>=2.10.0
keras
numpy
opencv-python
mediapipe
streamlit
onnxruntime-gpu  # or onnxruntime for CPU
scikit-learn
matplotlib
seaborn
pandas
```

## 📖 Usage

### Training the Model

1. **Prepare your data**
   - Ensure data is in `data/` directory with gesture subdirectories
   - Each gesture sequence should be saved as a `.npy` file

2. **Run training script**
```bash
python src/train_subvectors.py
```

3. **Monitor training**
```bash
tensorboard --logdir=Logs_MultiStream_TF
```

### Real-time Recognition Dashboard

1. **Launch the web application**
```bash
streamlit run dashboard/web.py
```

2. **Using the interface**
   - Click "Start Webcam" to begin real-time recognition
   - Perform gestures clearly in front of the camera
   - Adjust confidence threshold using the sidebar slider
   - Toggle landmark visualization options as needed
   - Configure hands loss timeout for better sequence management

### Features in Dashboard

- **Live Webcam Feed**: Real-time video with MediaPipe landmark visualization
- **Confidence Threshold**: Filter low-confidence predictions (0.4-1.0)
- **Landmark Controls**: Toggle face, pose, and hand landmark display
- **Hands Loss Timeout**: Automatically clear incomplete sequences
- **Sequence Voting**: Majority voting across 10 frames for stable predictions
- **Prediction Display**: Shows recognized gestures with confidence scores

## 📁 Project Structure

```
poc/
├── src/
│   └── train_subvectors.py          # Main training script
├── dashboard/
│   └── web.py                       # Streamlit web application
├── model/
│   └── bigger.onnx                  # Trained ONNX model
├── data/                            # Training data (gesture subdirectories)
├── report/                          # Training reports and visualizations
│   ├── bigger.json                  # Training configuration and results
│   ├── bigger.png                   # Training history plots
│   └── bigger_confusion.png         # Confusion matrix
├── logs/                            # General logs
├── Logs_MultiStream_TF/             # TensorBoard logs
│   └── training_run_YYYYMMDD_HHMMSS/
│       ├── train/                   # Training metrics
│       └── validation/              # Validation metrics
├── .gitignore                       # Git ignore configuration
└── README.md                        # This file
```

## 🎯 Model Performance

### Training Results

- **Final Training Accuracy**: 99.68%
- **Final Validation Accuracy**: 99.28%
- **Best Validation Accuracy**: 100%
- **Training Time**: ~5 minutes (GPU)
- **Epochs Completed**: 58 (with early stopping)

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 32 |
| Learning Rate | 0.0001 |
| Optimizer | Nadam |
| Max Epochs | 500 |
| Early Stopping Patience | 15 |
| Sequence Length | 30 frames |

## 🔧 Technical Details

### Feature Extraction Pipeline

1. **MediaPipe Holistic**: Extracts pose, hand, and face landmarks
2. **Subvector Generation**: Splits features into 7 semantic groups
3. **Geometric Features**: Computes angles, distances, and orientations
4. **Movement Features**: Calculates frame-to-frame velocities
5. **Sequence Buffering**: Maintains 30-frame sliding window

### Inference Optimization

- **ONNX Runtime**: Optimized inference with GPU acceleration
- **Batch Processing**: Processes sequences efficiently
- **Memory Management**: Fixed-size buffers for real-time performance
- **Error Handling**: Robust fallback mechanisms for edge cases

### Real-time Processing

- **Frame Rate**: 30 FPS target
- **Sequence Length**: 30 frames (1 second)
- **Prediction Frequency**: Every frame when buffer is full
- **Majority Voting**: 10-frame window for stable predictions
- **Hands Loss Timeout**: Configurable cleanup of incomplete sequences

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Ensure `bigger.onnx` exists in `model/` directory
   - Check ONNX Runtime installation (CPU vs GPU versions)

2. **Webcam Issues**
   - Verify camera permissions
   - Check if other applications are using the camera
   - Try different camera indices if multiple cameras available

3. **Performance Issues**
   - Use GPU acceleration if available
   - Close unnecessary applications
   - Adjust confidence threshold to reduce processing

4. **Recognition Accuracy**
   - Ensure good lighting conditions
   - Perform gestures clearly and completely
   - Allow adequate time for sequence buffering (30 frames)

### Debug Mode

Enable debug output by checking the console for detailed information:
- `[HIGH CONF]`: High-confidence predictions added to sequence
- `[FILTERED]`: Low-confidence predictions filtered out
- `[SEQUENCE RESULT]`: Final sequence predictions
- `[CLEANUP]`: Automatic cleanup of incomplete sequences

## 🔬 Model Conversion

The project includes model conversion from Keras H5 to ONNX format for optimized inference:

1. **Training**: Model saved as `bigger.h5` (Keras format)
2. **Conversion**: Converted to `bigger.onnx` (ONNX format)
3. **Validation**: Report saved as `bigger_conversion_report.json`

## 📈 Monitoring and Logging

### TensorBoard Integration

Monitor training progress with TensorBoard:
```bash
tensorboard --logdir=Logs_MultiStream_TF
```

### Available Metrics

- Training/Validation Loss
- Training/Validation Accuracy
- Learning Rate Schedule
- Model Architecture Graph

### Training Reports

Each training run generates comprehensive reports:
- JSON configuration and results (`bigger.json`)
- Training history plots (`bigger.png`)
- Confusion matrix (`bigger_confusion.png`)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **MediaPipe**: For robust hand and pose tracking
- **TensorFlow/Keras**: For deep learning framework
- **Streamlit**: For interactive web interface
- **ONNX Runtime**: For optimized model inference

## 📞 Contact

For questions, suggestions, or contributions, please contact:
- GitHub: @araara243
- Project Repository: https://github.com/araara243/poc

---

**Note**: This is a research proof-of-concept (POC) demonstrating the feasibility of real-time Malaysian Sign Language recognition using multi-stream Transformer architectures. The system is designed for academic and research purposes.
