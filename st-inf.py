import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
import torch.serialization

# ----------------- Config -----------------
MODEL_PATH = "ASD_Detection-ConvNext-final-model-2.pth"
NUM_CLASSES = 2
CLASS_NAMES = ["ASD", "Non-ASD"]

# ----------------- Load model -----------------
@st.cache_resource
def load_model():
    try:
        # Trust checkpoint and allow full pickle loading
        torch.serialization.add_safe_globals([np.core.multiarray.scalar])  # Only needed if numpy scalar error happens

        checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
        model = models.convnext_base(weights=None)
        model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, NUM_CLASSES)

        # Try to load 'model_state_dict' if present
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        return model

    except Exception as e:
        st.error("❌ Failed to load the model. See console for details.")
        raise e

# ----------------- Preprocess image -----------------
def preprocess_image(uploaded_file):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(uploaded_file).convert('RGB')
    return transform(image).unsqueeze(0), image

# ----------------- Predict -----------------
def get_prediction(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        predicted_class = torch.argmax(probs).item()
        confidence = probs[predicted_class].item()
    return predicted_class, confidence, probs

# ----------------- Streamlit UI -----------------
st.title("🧠 ASD Detection (ConvNeXt Model)")
st.markdown("Upload an image to detect ASD or Non-ASD.")

uploaded_image = st.file_uploader("Upload an image (JPG or PNG)", type=["jpg", "jpeg", "png"])

if uploaded_image:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    
    with st.spinner("🔄 Loading model and making prediction..."):
        model = load_model()
        image_tensor, _ = preprocess_image(uploaded_image)
        pred_class, confidence, probs = get_prediction(model, image_tensor)
    
    st.success(f"**Prediction:** {CLASS_NAMES[pred_class]} ({confidence * 100:.2f}%)")

 
