import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

# Load your trained model
model = torch.load("best_model.pth", map_location=torch.device("cuda"))
model.eval()

# Define preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

# Streamlit UI
st.title("Eye Disease Detection App")
st.write("Upload an image to classify eye diseases.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "tiff"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

    # Display result
    class_names = ["normal", "epiphora", "keratitis"]  # Update with your labels
    st.write(f"Prediction: **{class_names[predicted_class]}**")
