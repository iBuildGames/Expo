import torch
from torchvision import transforms
from PIL import Image
from cyclegan_architecture import Generator  # Import your Generator architecture

class CycleGAN:
    def __init__(self, model_path="cyclegan_weights.pth", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = Generator().to(self.device)  # Load the CycleGAN Generator
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.inv_transform = transforms.Compose([
            transforms.Normalize((-1, -1, -1), (2, 2, 2)),
            transforms.ToPILImage()
        ])

    def translate(self, image):
        image = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            translated_image = self.model(image)
        translated_image = self.inv_transform(translated_image.squeeze(0).cpu())
        return translated_image

# Initialize CycleGAN for testing
if __name__ == "__main__":
    cyclegan = CycleGAN(model_path="cyclegan_weights.pth")
    test_image = Image.open("test_white_light.jpg").convert("RGB")
    translated_image = cyclegan.translate(test_image)
    translated_image.show()  # Display the converted image
