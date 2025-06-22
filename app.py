# # Handwritten Digit Generation Web App
# # Run with: streamlit run app.py

# import os
# import sys

# # Fix for PyTorch-Streamlit compatibility issue
# os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

# import streamlit as st
# import torch
# import torch.nn as nn
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# import io
# import base64

# # Set page config
# st.set_page_config(
#     page_title="Handwritten Digit Generator",
#     page_icon="🔢",
#     layout="wide"
# )

# # Generator Network (same as training script)
# class Generator(nn.Module):
#     def __init__(self, latent_dim=100, num_classes=10):
#         super(Generator, self).__init__()
#         self.latent_dim = latent_dim
#         self.num_classes = num_classes
        
#         # Label embedding
#         self.label_embedding = nn.Embedding(num_classes, num_classes)
        
#         # Generator layers
#         self.model = nn.Sequential(
#             # Input: latent_dim + num_classes
#             nn.Linear(latent_dim + num_classes, 256),
#             nn.LeakyReLU(0.2),
#             nn.BatchNorm1d(256),
            
#             nn.Linear(256, 512),
#             nn.LeakyReLU(0.2),
#             nn.BatchNorm1d(512),
            
#             nn.Linear(512, 1024),
#             nn.LeakyReLU(0.2),
#             nn.BatchNorm1d(1024),
            
#             nn.Linear(1024, 28 * 28),
#             nn.Tanh()  # Output in [-1, 1]
#         )
    
#     def forward(self, noise, labels):
#         # Embed labels
#         label_embedding = self.label_embedding(labels)
        
#         # Concatenate noise and label embedding
#         gen_input = torch.cat((noise, label_embedding), dim=1)
        
#         # Generate image
#         img = self.model(gen_input)
#         img = img.view(img.size(0), 1, 28, 28)
#         return img

# # Load model function
# @st.cache_resource
# def load_model():
#     device = torch.device("cpu")  # Use CPU for web deployment
#     generator = Generator()
    
    
#     generator.load_state_dict(torch.load('digit_generator.pth', map_location=device))
    
#     generator.eval()
#     return generator, device

# # Generate digit function
# def generate_digit(generator, device, digit, num_samples=5):
#     with torch.no_grad():
#         # Create noise
#         noise = torch.randn(num_samples, 100).to(device)
        
#         # Create labels
#         labels = torch.full((num_samples,), digit, dtype=torch.long).to(device)
        
#         # Generate images
#         fake_imgs = generator(noise, labels)
        
#         # Denormalize images from [-1, 1] to [0, 1]
#         fake_imgs = (fake_imgs + 1) / 2
        
#         # Convert to numpy
#         fake_imgs = fake_imgs.cpu().numpy()
        
#         return fake_imgs

# # Convert numpy array to PIL Image
# def numpy_to_pil(img_array):
#     # Convert from [0, 1] to [0, 255]
#     img_array = (img_array * 255).astype(np.uint8)
#     # Squeeze to remove channel dimension
#     img_array = img_array.squeeze()
#     # Convert to PIL Image
#     return Image.fromarray(img_array, mode='L')

# # Main app
# def main():
#     st.title("🔢 Handwritten Digit Generator")
#     st.markdown("Generate handwritten digits using a trained GAN model!")
    
#     # Load model
#     try:
#         generator, device = load_model()
#         st.success("✅ Model loaded successfully!")
#     except Exception as e:
#         st.error(f"❌ Error loading model: {str(e)}")
#         st.stop()
    
#     # Sidebar for controls
#     with st.sidebar:
#         st.header("Generation Controls")
        
#         # Digit selection
#         selected_digit = st.selectbox(
#             "Select digit to generate:",
#             options=list(range(10)),
#             index=0,
#             help="Choose which digit (0-9) you want to generate"
#         )
        
#         # Generate button
#         generate_button = st.button("🎨 Generate 5 Images", type="primary")
        
#         # Instructions
#         st.markdown("---")
#         st.markdown("### Instructions:")
#         st.markdown("1. Select a digit (0-9)")
#         st.markdown("2. Click 'Generate 5 Images'")
#         st.markdown("3. View the generated handwritten digits")
        
#         # Model info
#         st.markdown("---")
#         st.markdown("### Model Info:")
#         st.markdown("- **Architecture**: Conditional GAN")
#         st.markdown("- **Dataset**: MNIST")
#         st.markdown("- **Image Size**: 28x28")
#         st.markdown("- **Framework**: PyTorch")
    
#     # Main content area
#     col1, col2 = st.columns([2, 1])
    
#     with col1:
#         st.header(f"Generated Digit: {selected_digit}")
        
#         if generate_button:
#             with st.spinner("Generating images..."):
#                 try:
#                     # Generate images
#                     generated_images = generate_digit(generator, device, selected_digit, 5)
                    
#                     # Display images in a grid
#                     cols = st.columns(5)
                    
#                     for i, img_array in enumerate(generated_images):
#                         with cols[i]:
#                             # Convert to PIL Image
#                             pil_img = numpy_to_pil(img_array)
                            
#                             # Resize for better display
#                             pil_img_resized = pil_img.resize((112, 112), Image.NEAREST)
                            
#                             # Display image
#                             st.image(
#                                 pil_img_resized, 
#                                 caption=f"Sample {i+1}",
#                                 use_container_width=True
#                             )
                    
#                     st.success("✅ Successfully generated 5 images!")
                    
#                     # Show original size images
#                     with st.expander("View Original Size (28x28)"):
#                         cols_orig = st.columns(5)
#                         for i, img_array in enumerate(generated_images):
#                             with cols_orig[i]:
#                                 pil_img = numpy_to_pil(img_array)
#                                 st.image(
#                                     pil_img, 
#                                     caption=f"Original {i+1}",
#                                     use_container_width=True
#                                 )
                
#                 except Exception as e:
#                     st.error(f"❌ Error generating images: {str(e)}")
        
#         else:
#             # Show placeholder
#             st.info("👆 Select a digit and click 'Generate 5 Images' to see AI-generated handwritten digits!")
            
#             # Show example of what to expect
#             st.markdown("### What to expect:")
#             st.markdown("- 5 unique variations of the selected digit")
#             st.markdown("- Images similar to MNIST dataset style")
#             st.markdown("- 28x28 pixel grayscale images")
#             st.markdown("- Each generation produces different variations")
    
#     with col2:
#         st.header("About")
#         st.markdown("""
#         This web app uses a **Conditional Generative Adversarial Network (cGAN)** 
#         trained on the MNIST dataset to generate handwritten digits.
        
#         **Key Features:**
#         - Generate any digit from 0-9
#         - Produces 5 unique variations
#         - Real-time generation
#         - MNIST-style output
        
#         **Technical Details:**
#         - **Model**: Conditional GAN
#         - **Training**: Google Colab T4 GPU
#         - **Dataset**: MNIST (70,000 images)
#         - **Architecture**: Fully connected layers
#         - **Loss**: Binary Cross Entropy
#         """)
        
#         # Performance metrics (placeholder)
#         st.markdown("---")
#         st.markdown("### Model Performance:")
#         st.metric("Training Epochs", "50")
#         st.metric("Final Generator Loss", "0.69")
#         st.metric("Final Discriminator Loss", "0.73")
        
#         # Download section
#         st.markdown("---")
#         st.markdown("### Download:")
#         if st.button("📥 Download Model Info"):
#             model_info = """
#             Handwritten Digit Generator Model
            
#             Architecture: Conditional GAN
#             - Generator: 4-layer MLP with BatchNorm
#             - Discriminator: 4-layer MLP with Dropout
#             - Latent Dimension: 100
#             - Classes: 10 (digits 0-9)
            
#             Training Details:
#             - Dataset: MNIST
#             - Batch Size: 128
#             - Learning Rate: 0.0002
#             - Optimizer: Adam (β1=0.5, β2=0.999)
#             - Loss Function: Binary Cross Entropy
#             """
            
#             st.download_button(
#                 label="Download Model Info",
#                 data=model_info,
#                 file_name="model_info.txt",
#                 mime="text/plain"
#             )

# if __name__ == "__main__":
#     main()

# # Footer
# st.markdown("---")
# st.markdown(
#     "<div style='text-align: center; color: gray;'>"
#     "Built with Streamlit and PyTorch | Handwritten Digit Generation"
#     "</div>",
#     unsafe_allow_html=True
# )













# Handwritten Digit Generation Web App
# Run with: streamlit run app.py

import os
import sys

# Fix for PyTorch-Streamlit compatibility issue
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64

# Set page config
st.set_page_config(
    page_title="Handwritten Digit Generator",
    page_icon="🔢",
    layout="wide"
)

# Generator Network (same as training script)
class Generator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10):
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
            
            nn.Linear(1024, 28 * 28),
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def forward(self, noise, labels):
        # Embed labels
        label_embedding = self.label_embedding(labels)
        
        # Concatenate noise and label embedding
        gen_input = torch.cat((noise, label_embedding), dim=1)
        
        # Generate image
        img = self.model(gen_input)
        img = img.view(img.size(0), 1, 28, 28)
        return img

# Load model function
@st.cache_resource
def load_model():
    device = torch.device("cpu")  # Use CPU for web deployment
    generator = Generator()
    
    
    generator.load_state_dict(torch.load('digit_generator.pth', map_location=device))
    
    generator.eval()
    return generator, device

# Generate digit function
def generate_digit(generator, device, digit, num_samples=5):
    with torch.no_grad():
        # Create noise
        noise = torch.randn(num_samples, 100).to(device)
        
        # Create labels
        labels = torch.full((num_samples,), digit, dtype=torch.long).to(device)
        
        # Generate images
        fake_imgs = generator(noise, labels)
        
        # Denormalize images from [-1, 1] to [0, 1]
        fake_imgs = (fake_imgs + 1) / 2
        
        # Convert to numpy
        fake_imgs = fake_imgs.cpu().numpy()
        
        return fake_imgs

# Convert numpy array to PIL Image
def numpy_to_pil(img_array):
    # Convert from [0, 1] to [0, 255]
    img_array = (img_array * 255).astype(np.uint8)
    # Squeeze to remove channel dimension
    img_array = img_array.squeeze()
    # Convert to PIL Image
    return Image.fromarray(img_array, mode='L')

# Main app
def main():
    st.title("🔢 AI Handwritten Digit Generator")
    st.markdown("### Generate realistic handwritten digits using AI")
    
    # Load model
    try:
        generator, device = load_model()
        st.success("✅ Model ready!")
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()
    
    # Main controls (moved to main area)
    st.markdown("---")
    
    # Create centered controls
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("**Choose a digit and generate:**")
        
        # Create sub-columns for aligned inputs
        input_col1, input_col2 = st.columns([1, 1])
        
        with input_col1:
            selected_digit = st.selectbox(
                "Select digit:",
                options=list(range(10)),
                index=0
            )
        
        with input_col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Add spacing to align with selectbox
            generate_button = st.button("🎨 Generate Images", type="primary", use_container_width=True)
    
    st.markdown("---")
    
    # Results area
    if generate_button:
        st.subheader(f"Generated Digit: {selected_digit}")
        
        with st.spinner("Generating images..."):
            try:
                # Generate images
                generated_images = generate_digit(generator, device, selected_digit, 5)
                
                # Display images in a grid
                cols = st.columns(5)
                
                for i, img_array in enumerate(generated_images):
                    with cols[i]:
                        # Convert to PIL Image
                        pil_img = numpy_to_pil(img_array)
                        
                        # Resize for better display
                        pil_img_resized = pil_img.resize((112, 112), Image.NEAREST)
                        
                        # Display image
                        st.image(
                            pil_img_resized, 
                            caption=f"Sample {i+1}",
                            use_container_width=True
                        )
                
                st.success("✅ Successfully generated 5 unique variations!")
                
                # Show original size images
                with st.expander("🔍 View Original Size (28x28)"):
                    cols_orig = st.columns(5)
                    for i, img_array in enumerate(generated_images):
                        with cols_orig[i]:
                            pil_img = numpy_to_pil(img_array)
                            st.image(
                                pil_img, 
                                caption=f"Original {i+1}",
                                use_container_width=True
                            )
                
                # Generate more button
                if st.button("🔄 Generate More", use_container_width=True):
                    st.rerun()
            
            except Exception as e:
                st.error(f"❌ Error generating images: {str(e)}")
    
    else:
        # Welcome message
        st.info("👆 **How to use:** Select a digit (0-9) and click 'Generate Images' to see AI-created handwritten digits!")
        
        # Show sample info
        st.markdown("### What you'll get:")
        sample_cols = st.columns(3)
        with sample_cols[0]:
            st.markdown("🎯 **5 unique variations** of your chosen digit")
        with sample_cols[1]:
            st.markdown("✨ **High quality** AI-generated images")
        with sample_cols[2]:
            st.markdown("⚡ **Instant results** in seconds")
    
    # Collapsible info section
    with st.expander("ℹ️ About this AI Model"):
        info_cols = st.columns(2)
        with info_cols[0]:
            st.markdown("""
            **Technology:**
            - Conditional GAN (Generative Adversarial Network)
            - Trained on MNIST dataset (70,000 handwritten digits)
            - PyTorch framework
            """)
        with info_cols[1]:
            st.markdown("""
            **Performance:**
            - 50 training epochs
            - 28x28 pixel output
            - CPU-optimized for web deployment
            """)

if __name__ == "__main__":
    main()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 0.8em;'>"
    "AI Handwritten Digit Generator | Built with Streamlit & PyTorch"
    "</div>",
    unsafe_allow_html=True
)