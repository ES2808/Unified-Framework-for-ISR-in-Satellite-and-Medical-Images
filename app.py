import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import os
import io

from src.models.unified_model import UnifiedSRModel
from inference import forward_x8, forward_chop

# Page Config
st.set_page_config(page_title="Unified Super-Resolution", layout="wide")

st.title("Unified Super-Resolution Interface")
st.markdown("Upload a **Medical** or **Satellite** image to enhance its resolution.")

# Sidebar Settings
st.sidebar.header("Settings")

# Model Path
MODEL_PATH = "checkpoints/model_final.pth"

@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UnifiedSRModel(upscale=4).to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        return model, device
    else:
        return None, device

model, device = load_model()

if model is None:
    st.error(f"Model not found at `{MODEL_PATH}`. Please train the model first or check the path.")
else:
    st.sidebar.success("Model Loaded Successfully!")

# Options
domain_option = st.sidebar.selectbox(
    "Domain",
    ("Auto-Detect", "Medical", "Satellite"),
    help="Force a specific domain or let the model decide."
)

use_ensemble = st.sidebar.checkbox(
    "Use Self-Ensemble (x8)",
    value=False,
    help="Slower but produces higher quality results by averaging 8 augmentations."
)

# File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "tif"])

if uploaded_file is not None:
    # Read Image
    image = Image.open(uploaded_file).convert('RGB')
    img_np = np.array(image)
    
    # Display Original
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original (Low Resolution)")
        st.image(image, use_container_width=True)
        st.caption(f"Size: {image.size}")

    # Process Button
    if st.button("Super-Resolve Image"):
        with st.spinner("Processing..."):
            # Preprocess
            lr_tensor = torch.from_numpy(img_np.transpose((2, 0, 1))).float() / 255.0
            lr_tensor = lr_tensor.unsqueeze(0).to(device)
            
            # Determine Domain
            domain = None
            if domain_option == "Medical":
                domain = "medical"
            elif domain_option == "Satellite":
                domain = "satellite"
            
            # Inference
            with torch.no_grad():
                if use_ensemble:
                    # For ensemble, we need a fixed domain. If Auto, predict once first.
                    if domain is None:
                         _, logits = model(lr_tensor, None)
                         pred_idx = torch.argmax(logits, dim=1).item()
                         ens_domain = 'medical' if pred_idx == 0 else 'satellite'
                         st.info(f"Auto-detected Domain: **{ens_domain.capitalize()}**")
                    else:
                         ens_domain = domain
                    
                    sr_tensor, _ = forward_x8(model, lr_tensor, ens_domain)
                else:
                    # Use forward_chop to handle large images (tiling)
                    # min_size=10000 ensures roughly 100x100 patches, safe for attention (~400MB mem)
                    sr_tensor, domain_logits = forward_chop(model, lr_tensor, domain, scale=4, min_size=10000)
                    
                    # Show detected domain if Auto
                    if domain is None:
                        pred_idx = torch.argmax(domain_logits, dim=1).item()
                        detected = 'Medical' if pred_idx == 0 else 'Satellite'
                        st.info(f"Auto-detected Domain: **{detected}**")

            # Postprocess
            sr_img = sr_tensor.squeeze(0).cpu().numpy().transpose((1, 2, 0))
            sr_img = np.clip(sr_img, 0, 1) * 255.0
            sr_img = sr_img.round().astype(np.uint8)
            sr_pil = Image.fromarray(sr_img)
            
            # Display Result
            with col2:
                st.subheader("Result (Super-Resolution)")
                st.image(sr_pil, use_container_width=True)
                st.caption(f"Size: {sr_pil.size}")
            
            # Download Button
            buf = io.BytesIO()
            sr_pil.save(buf, format="PNG")
            byte_im = buf.getvalue()
            
            st.download_button(
                label="Download SR Image",
                data=byte_im,
                file_name=f"sr_{uploaded_file.name}",
                mime="image/png"
            )
