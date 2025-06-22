# Handwritten Digit Generation Training Script
# Run this in Google Colab with T4 GPU

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 128
LEARNING_RATE = 0.0002
NUM_EPOCHS = 50
LATENT_DIM = 100
IMAGE_SIZE = 28

# Data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

# Load MNIST dataset
train_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform
)

train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    drop_last=True
)

# Generator Network
class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes=10):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Label embedding
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        
        # Generator layers
        self.model = nn.Sequential(
            # Input: latent_dim + num_classes
            nn.Linear(latent_dim + num_classes, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            
            nn.Linear(1024, IMAGE_SIZE * IMAGE_SIZE),
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def forward(self, noise, labels):
        # Embed labels
        label_embedding = self.label_embedding(labels)
        
        # Concatenate noise and label embedding
        gen_input = torch.cat((noise, label_embedding), dim=1)
        
        # Generate image
        img = self.model(gen_input)
        img = img.view(img.size(0), 1, IMAGE_SIZE, IMAGE_SIZE)
        return img

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, num_classes=10):
        super(Discriminator, self).__init__()
        self.num_classes = num_classes
        
        # Label embedding
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        
        # Discriminator layers
        self.model = nn.Sequential(
            # Input: image_size + num_classes
            nn.Linear(IMAGE_SIZE * IMAGE_SIZE + num_classes, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img, labels):
        # Flatten image
        img_flat = img.view(img.size(0), -1)
        
        # Embed labels
        label_embedding = self.label_embedding(labels)
        
        # Concatenate image and label embedding
        disc_input = torch.cat((img_flat, label_embedding), dim=1)
        
        # Discriminate
        validity = self.model(disc_input)
        return validity

# Initialize models
generator = Generator(LATENT_DIM).to(device)
discriminator = Discriminator().to(device)

# Loss function
criterion = nn.BCELoss()

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

# Training loop
def train_gan():
    generator.train()
    discriminator.train()
    
    G_losses = []
    D_losses = []
    
    for epoch in range(NUM_EPOCHS):
        epoch_G_loss = 0
        epoch_D_loss = 0
        
        for i, (real_imgs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            labels = labels.to(device)
            
            # Create labels for real and fake data
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            
            # Real images
            real_validity = discriminator(real_imgs, labels)
            d_real_loss = criterion(real_validity, real_labels)
            
            # Fake images
            noise = torch.randn(batch_size, LATENT_DIM).to(device)
            fake_labels_for_gen = torch.randint(0, 10, (batch_size,)).to(device)
            fake_imgs = generator(noise, fake_labels_for_gen)
            fake_validity = discriminator(fake_imgs.detach(), fake_labels_for_gen)
            d_fake_loss = criterion(fake_validity, fake_labels)
            
            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()
            
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            
            # Generate fake images
            noise = torch.randn(batch_size, LATENT_DIM).to(device)
            fake_labels_for_gen = torch.randint(0, 10, (batch_size,)).to(device)
            fake_imgs = generator(noise, fake_labels_for_gen)
            
            # Fool discriminator
            fake_validity = discriminator(fake_imgs, fake_labels_for_gen)
            g_loss = criterion(fake_validity, real_labels)
            
            g_loss.backward()
            optimizer_G.step()
            
            epoch_G_loss += g_loss.item()
            epoch_D_loss += d_loss.item()
        
        # Record losses
        avg_G_loss = epoch_G_loss / len(train_loader)
        avg_D_loss = epoch_D_loss / len(train_loader)
        G_losses.append(avg_G_loss)
        D_losses.append(avg_D_loss)
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | G_loss: {avg_G_loss:.4f} | D_loss: {avg_D_loss:.4f}")
        
        # Generate sample images
        if (epoch + 1) % 10 == 0:
            generate_sample_images(epoch + 1)
    
    return G_losses, D_losses

# Function to generate sample images
def generate_sample_images(epoch):
    generator.eval()
    with torch.no_grad():
        # Generate one image for each digit
        noise = torch.randn(10, LATENT_DIM).to(device)
        labels = torch.arange(0, 10).to(device)
        fake_imgs = generator(noise, labels)
        
        # Denormalize images
        fake_imgs = (fake_imgs + 1) / 2  # Convert from [-1, 1] to [0, 1]
        
        # Plot images
        fig, axes = plt.subplots(2, 5, figsize=(10, 4))
        for i in range(10):
            row = i // 5
            col = i % 5
            axes[row, col].imshow(fake_imgs[i].cpu().squeeze(), cmap='gray')
            axes[row, col].set_title(f'Digit {i}')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'generated_digits_epoch_{epoch}.png')
        plt.show()
    generator.train()

# Function to generate specific digit
def generate_digit(digit, num_samples=5):
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_samples, LATENT_DIM).to(device)
        labels = torch.full((num_samples,), digit, dtype=torch.long).to(device)
        fake_imgs = generator(noise, labels)
        
        # Denormalize images
        fake_imgs = (fake_imgs + 1) / 2  # Convert from [-1, 1] to [0, 1]
        
        return fake_imgs.cpu().numpy()

# Train the model
print("Starting training...")
G_losses, D_losses = train_gan()

# Plot training losses
plt.figure(figsize=(10, 5))
plt.plot(G_losses, label='Generator Loss')
plt.plot(D_losses, label='Discriminator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Losses')
plt.legend()
plt.savefig('training_losses.png')
plt.show()

# Save the trained generator model
torch.save(generator.state_dict(), 'digit_generator.pth')
print("Model saved as 'digit_generator.pth'")

# Test generation for all digits
print("Testing generation for all digits...")
for digit in range(10):
    print(f"Generating digit {digit}")
    generated_imgs = generate_digit(digit, 5)
    
    plt.figure(figsize=(12, 2))
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(generated_imgs[i].squeeze(), cmap='gray')
        plt.title(f'Digit {digit} - Sample {i+1}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'test_digit_{digit}.png')
    plt.show()

print("Training completed!")