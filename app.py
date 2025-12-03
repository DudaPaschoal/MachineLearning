import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import io
import os # Importar para verificar o arquivo do modelo

# --- 1. Model Architecture (MUST be identical to train_model.py) ---
# This needs to be defined again so Streamlit can load the saved state_dict
# into the correct model structure.

LATENT_DIM = 20
NUM_CLASSES = 10
IMG_DIM = 28 * 28

# Coloque a DEFINIÇÃO DA CLASSE CVAE AQUI, ANTES DE QUALQUER INSTANCIAÇÃO
class CVAE(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_dim = IMG_DIM

        # Encoder layers (not strictly needed for generation, but good for completeness)
        self.fc1_enc = nn.Linear(self.img_dim + self.num_classes, 512)
        self.fc2_enc = nn.Linear(512, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # Decoder layers
        self.fc1_dec = nn.Linear(latent_dim + self.num_classes, 256)
        self.fc2_dec = nn.Linear(256, 512)
        self.fc3_dec = nn.Linear(512, self.img_dim)

    def encode(self, x, labels):
        h1 = F.relu(self.fc1_enc(torch.cat([x, labels], dim=1)))
        h2 = F.relu(self.fc2_enc(h1))
        return self.fc_mu(h2), self.fc_logvar(h2)

    def decode(self, z, labels):
        h1 = F.relu(self.fc1_dec(torch.cat([z, labels], dim=1)))
        h2 = F.relu(self.fc2_dec(h1))
        return torch.sigmoid(self.fc3_dec(h2))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, labels):
        mu, logvar = self.encode(x.view(-1, self.img_dim), labels)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z, labels)
        return reconstruction, mu, logvar

# --- 2. Load Trained Model ---
@st.cache_resource # ✅ Corrigido: Usando o caching moderno para modelos
def load_model():
    model = CVAE(LATENT_DIM, NUM_CLASSES) # Instanciação da classe CVAE
    model_path = 'cvae_mnist_model.pth'
    if not os.path.exists(model_path):
        st.error(f"Error: Model file '{model_path}' not found. Please train the model using train_model.py first.")
        st.stop() # Stop the app if model is not found
        return None # Return None in case of stop()
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval() # Set model to evaluation mode
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}. Ensure the model architecture matches the saved state_dict.")
        st.stop() # Stop the app if model loading fails
    return model

model = load_model() # A chamada à função de carregamento do modelo

# --- 3. Image Generation Function ---
def generate_images(model, digit, num_images=5):
    with torch.no_grad(): # No need to calculate gradients for inference
        # Convert digit to one-hot encoding
        labels = F.one_hot(torch.tensor([digit] * num_images), num_classes=NUM_CLASSES).float()

        # Sample from a standard normal distribution for the latent vector
        z = torch.randn(num_images, LATENT_DIM) * 0.8

        generated_images_flat = model.decode(z, labels)
        generated_images = generated_images_flat.view(num_images, 28, 28).cpu().numpy()
    return generated_images

# --- 4. Streamlit Web Application Interface ---
st.set_page_config(layout="wide", page_title="Handwritten Digit Image Generator")

st.title("✍️ Handwritten Digit Image Generator")
st.markdown("Generate 5 synthetic MNIST-like images using your trained CVAE model.")

# User input: Choose digit
selected_digit = st.selectbox(
    "Choose which digit (0-9) to generate:",
    options=list(range(10)),
    index=0
)

# Button to trigger generation
if st.button("Generate Images"):
    if model: # Only generate if the model was loaded successfully
        st.subheader(f"Generated Images of digit {selected_digit}:")
        images = generate_images(model, selected_digit, num_images=5)

        cols = st.columns(5) # Create 5 columns for images

        for i, img_array in enumerate(images):
            img_array_scaled = (img_array * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_array_scaled, mode='L')

            buf = io.BytesIO()
            img_pil.save(buf, format="PNG")
            byte_im = buf.getvalue()

            with cols[i]:
                st.image(byte_im, caption=f"Sample {i+1}", use_container_width=True) # ✅ Corrigido: Usando use_container_width
                st.write(f"") # Just a small space for alignment
    else:
        st.warning("Model could not be loaded. Please check the error messages above.")


st.markdown("""
<style>
.stSelectbox {
    margin-bottom: 20px;
}
.stButton {
    margin-top: 20px;
}
.stImage {
    border: 1px solid #ddd;
    border-radius: 4px;
    padding: 5px;
}
</style>
""", unsafe_allow_html=True)