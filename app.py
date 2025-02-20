import torch
import torch.nn as nn
from torchvision import models
import streamlit as st
from PIL import Image
from torchvision import transforms

class YourModel(nn.Module):
    def __init__(self):
        super(YourModel, self).__init__()
        self.model = models.resnet34(pretrained=True)  
        self.model.fc = nn.Linear(512, 3)  

    def forward(self, x):
        return self.model(x)


model = YourModel()
model.load_state_dict(torch.load("newer_model.pth", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))

model.eval() 


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

st.title("Vision IA : Détection préventive de maladies occulaires")
st.write("Soumettez une image afin de réaliser une prédiction.")


uploaded_file = st.file_uploader("Choisissez une image...", type=["png", "jpg", "jpeg", "tiff"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Image téléchargée", use_container_width=True)

    
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

    class_names = ["normal", "epiphore", "keratite"]  
    st.write(f"Prediction: **{class_names[predicted_class]}**")
