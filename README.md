# Autism Detection using ConvNeXt

Autism spectrum disorder (ASD) detection from facial images using ConvNeXt, a modern convolutional neural network architecture.

## Training Details

- **Architecture**: ConvNeXt
- **Dataset**: 2000+ facial images (ASD classification)
- **GPU**: NVIDIA L40S
- **Epochs**: 50 (30 frozen layers + 20 full fine-tuning)

## Overview

ConvNeXt modernizes traditional CNNs with design elements from vision transformers while maintaining the simplicity and efficiency of convolutional networks.

## Setup

```bash
git clone https://github.com/Pranavharshans/AutismDetection-ConvNext.git
cd AutismDetection-ConvNext
pip install torch torchvision timm opencv-python numpy matplotlib scikit-learn
```

## Usage

1. Prepare the dataset in the expected directory structure
2. Run the training notebook/script
3. Evaluate with the provided metrics

## Project Structure

```
AutismDetection-ConvNext/
├── *.ipynb / *.py       # Training and inference code
├── *.pth                 # Model checkpoints (gitignored)
└── README.md
```

## Dependencies

- PyTorch + torchvision
- timm (PyTorch Image Models)
- OpenCV
- NumPy / Matplotlib / scikit-learn
