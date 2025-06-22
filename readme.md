A web application that generates handwritten digits (0-9) using a Conditional Generative Adversarial Network (cGAN) trained on the MNIST dataset.

## Features

- **Interactive Web Interface**: Select any digit (0-9) and generate 5 unique variations
- **Real-time Generation**: Fast inference using trained GAN model
- **MNIST-style Output**: 28x28 grayscale images similar to the original dataset
- **Public Access**: Deployed web app accessible to anyone

## Architecture

### Generator Network
- 4-layer fully connected network with BatchNorm
- Input: 100-dimensional noise + 10-dimensional label embedding
- Output: 28x28 grayscale image
- Activation: LeakyReLU (hidden), Tanh (output)

### Discriminator Network
- 4-layer fully connected network with Dropout
- Input: 28x28 image + 10-dimensional label embedding
- Output: Binary classification (real/fake)
- Activation: LeakyReLU (hidden), Sigmoid (output)

## Training Details

- **Dataset**: MNIST (70,000 handwritten digits)
- **Framework**: PyTorch
- **Training Environment**: Google Colab T4 GPU
- **Batch Size**: 128
- **Learning Rate**: 0.0002
- **Optimizer**: Adam (β1=0.5, β2=0.999)
- **Loss Function**: Binary Cross Entropy
- **Training Time**: ~50 epochs (~45 minutes)

## Usage

1. Visit the deployed web application
2. Select a digit (0-9) from the dropdown
3. Click "Generate 5 Images"
4. View the generated handwritten digits

## Files

- `train_model.py`: Complete training script for Google Colab
- `app.py`: Streamlit web application
- `requirements.txt`: Python dependencies
- `digit_generator.pth`: Trained model weights (generated after training)

## Performance

The model generates diverse, recognizable handwritten digits that are easily identifiable by both humans and AI systems like ChatGPT-4o.

## Web App Link

[To be updated with actual deployment URL]

---

Built with PyTorch and Streamlit | Trained on Google Colab T4 GPU
