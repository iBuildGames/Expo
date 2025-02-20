import torch
import torch.nn as nn
from torchvision import models
import streamlit as st
from PIL import Image
from torchvision import transforms

# Define the model architecture (must match the trained model)
class YourModel(nn.Module):
    def __init__(self):
        super(YourModel, self).__init__()
        self.model = models.resnet34(pretrained=True)  # Change to your actual model
        self.model.fc = nn.Linear(512, 3)  # Change output size to match number of classes

    def forward(self, x):
        return self.model(x)

# Initialize model and load weights
model = YourModel()
model.load_state_dict(torch.load("newer_model.pth", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))

model.eval()  # Now it works!

# Define preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

# Streamlit UI
st.title("Vision IA : Détection préventive de maladies occulaires")
st.write("Soumettez une image afin de réaliser une prédiction.")

# Upload image
uploaded_file = st.file_uploader("Choisissez une image...", type=["png", "jpg", "jpeg", "tiff"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Image téléchargée", use_container_width=True)

    # Preprocess and predict
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

    # Display result
    class_names = ["normal", "epiphore", "keratite"]  
    st.write(f"Prediction: **{class_names[predicted_class]}**")
