# Binary Image Autoencoder

A TensorFlow-based implementation of a binary image autoencoder trained on the MNIST dataset. This project demonstrates how to compress and reconstruct handwritten digit images using deep learning techniques specifically optimized for binary data.

## 🎯 Project Overview

This autoencoder learns to compress 28x28 binary images (784 pixels) into a 32-dimensional latent space representation, achieving a **24.5x compression ratio** while maintaining high-quality reconstructions of handwritten digits.

The project explores the fascinating world of **dimensionality reduction** and **lossy compression** through neural networks, showcasing how deep learning can learn efficient representations of visual data. By focusing on binary images, we eliminate grayscale complexity and create a clean, interpretable compression system.

## 🌟 Project Scope

### Core Functionality
- **Binary image compression** using deep neural networks
- **MNIST digit reconstruction** with high fidelity
- **Latent space exploration** for understanding data representations
- **Real-time encoding/decoding** of handwritten digits

### Technical Scope
- Implementation of encoder-decoder architecture
- Binary crossentropy optimization for binary data
- Comprehensive training and evaluation pipeline
- Interactive visualization and analysis tools
- Google Colab compatibility for easy experimentation

### Educational Value
- Demonstrates fundamental autoencoder concepts
- Showcases binary data optimization techniques
- Provides hands-on experience with TensorFlow/Keras
- Illustrates compression vs. quality trade-offs in machine learning

## 💡 Inspiration

This project was inspired by the **fundamental challenge of data compression** in machine learning and computer vision. Traditional compression algorithms like ZIP or JPEG work well for general data, but neural networks can learn **domain-specific representations** that often achieve superior compression ratios for specific types of data.

### Key Inspirations

**Classical Autoencoders**: Building upon the foundational work of Hinton and Salakhutdinov (2006) on deep autoencoders, this project adapts their architecture for binary image data.

**Information Theory**: The project explores the balance between **information preservation** and **compression efficiency**, demonstrating how neural networks can automatically discover the most important features for reconstruction.

**Practical Applications**: Binary image compression has real-world applications in:
- **Document digitization** and storage
- **Medical imaging** (X-rays, binary masks)
- **Computer vision preprocessing** 
- **Edge computing** with limited storage

**Educational Mission**: The project aims to make autoencoder concepts accessible through clear visualizations and well-documented code, helping students and practitioners understand how neural networks learn compressed representations.

The choice of **MNIST dataset** provides a perfect balance of simplicity and complexity - simple enough for educational purposes, yet complex enough to demonstrate meaningful compression and reconstruction capabilities.

## ✨ Features

- **Binary-optimized architecture** with binary crossentropy loss
- **High compression ratio** (784 → 32 dimensions)
- **Comprehensive visualization** of training progress and results
- **Ready-to-run on Google Colab** with automatic dataset loading
- **Real-time reconstruction testing** with interactive examples

## 🏗️ Model Architecture

**Encoder**: 784 → 512 → 256 → 128 → 32 (latent space)
**Decoder**: 32 → 128 → 256 → 512 → 784

- **Input**: 28x28 binary images (MNIST digits)
- **Latent Space**: 32 dimensions
- **Activation**: ReLU (hidden layers), Sigmoid (output layer)
- **Loss Function**: Binary crossentropy
- **Optimizer**: Adam

## 🚀 Quick Start

### Running on Google Colab

1. Open the notebook in Google Colab
2. Run all cells - the MNIST dataset will download automatically
3. Training takes ~5-10 minutes on Colab's free tier
4. View results with automatic visualization

### Local Installation

```bash
# Clone the repository
git clone https://github.com/vi-jigishu/BinaryImageEncoder.git
cd BinaryImageEncoder

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook binary_image_autoencoder.ipynb
```

## 📋 Requirements

```
tensorflow>=2.8.0
numpy>=1.21.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
jupyter>=1.0.0
```

## 📊 Results

The model achieves excellent reconstruction quality on binary MNIST digits:

- **Training Loss**: Rapidly converges with binary crossentropy
- **Compression Ratio**: 24.5x reduction in data size
- **Reconstruction Quality**: High fidelity binary digit reconstruction
- **Training Time**: ~5-10 minutes on modern hardware

## 🎨 Visualizations

The notebook includes comprehensive visualizations:

- **Training progress plots** (loss and MSE curves)
- **Before/after reconstruction comparisons**
- **Latent space representations**
- **Interactive reconstruction testing**

## 📁 Project Structure

```
BinaryImageEncoder/
├── binary_image_autoencoder.ipynb    # Main notebook
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
└── examples/                         # Sample outputs (optional)
```

## 🔧 Customization

### Modify Architecture
```python
# Change latent dimensions
latent_dimensions = 64  # Default: 32

# Adjust layer sizes
encoder_layers = [512, 256, 128, 64]  # Customize as needed
```

### Try Different Datasets
The code can be adapted for other binary image datasets by modifying the data loading section.

## 📚 Technical Details

### Binary Optimization
- Uses binary crossentropy loss for better binary data handling
- Applies sigmoid activation for proper binary output range
- Implements thresholding at 0.5 for crisp binary reconstructions

### Data Processing
- Automatic MNIST dataset loading and preprocessing
- Pixel normalization to [1] range
- Binary thresholding for clean binary images


## 📊 Results

The model achieves excellent reconstruction quality on binary MNIST digits:

- **Mean Reconstruction Error**: 0.014021 (very low error rate)
- **Training Loss**: Rapidly converges with binary crossentropy
- **Compression Ratio**: 24.5x reduction in data size (784 → 32 dimensions)
- **Reconstruction Quality**: High fidelity binary digit reconstruction
- **Training Time**: ~5-10 minutes on modern hardware

### Performance Metrics
- **Original Image Size**: 784 pixels (28×28)
- **Compressed Size**: 32 values (latent dimensions)
- **Data Reduction**: 95.9% size reduction
- **Reconstruction Accuracy**: 98.6% (based on mean reconstruction error)
