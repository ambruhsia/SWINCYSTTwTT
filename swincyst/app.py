import os
import torch
import torchvision.transforms as transforms
import timm

import numpy as np
import io
import uvicorn
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import json
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.datasets as datasets

# Initialize FastAPI app
app = FastAPI(title="Swin Transformer PCOS Classifier")

# Paths
MODEL_PATH = r"best_swin_pcos_model(4).pth"  # Update if needed
# TEST_DATA_DIR = r"C:\Users\rajpu\swincyst\test"  # Test images directory


# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Swin Transformer Model
def load_swin_model(model_path):
    model = timm.create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=2)
    model = model.to(device)

    # Load the trained weights
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f"Model loaded successfully from {model_path}")
    return model

model = load_swin_model("best_swin_pcos_model(4).pth")

# Define transformations for preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Class names (Assuming 2 classes: "Not Infected" and "Infected")
class_names = ["infected", "not infected"]

# Custom ImageFolder class to skip invalid images
class ImageFolderEX(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            sample = self.loader(path)
            if self.transform:
                sample = self.transform(sample)
            if self.target_transform:
                target = self.target_transform(target)
            return sample, target
        except Exception as e:
            print(f"Skipping invalid image: {path} (Error: {e})")
            return None  # Skip corrupted images

# Function to predict an image
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
    
    return class_names[predicted.item()]

# # Evaluate model on the test dataset
# def evaluate_model():
#     dataset = ImageFolderEX(TEST_DATA_DIR, transform=transform)
    
#     # Remove invalid (None) samples
#     valid_samples = [s for s in dataset if s is not None]
#     dataloader = DataLoader(valid_samples, batch_size=32, shuffle=False, num_workers=0)

#     all_preds, all_labels = [], []
    
#     with torch.no_grad():
#         for inputs, labels in tqdm(dataloader, desc="Evaluating"):
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)

#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())

#     accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
#     report = classification_report(all_labels, all_preds, target_names=class_names)
#     cm = confusion_matrix(all_labels, all_preds)

#     print(f"\n🔹 Model Accuracy: {accuracy:.4f}")
#     print("\n🔹 Classification Report:\n", report)

#     plt.figure(figsize=(6, 5))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
#     plt.title("Confusion Matrix")
#     plt.xlabel("Predicted")
#     plt.ylabel("Actual")
#     plt.show()

#     return accuracy, report, cm

# # Run evaluation before starting the FastAPI app
# print("🔍 Evaluating Model on Test Dataset...")
# evaluate_model()

# Test a random image from the dataset

# def test_random_images(num_samples=5):
#     infected_images = [os.path.join(TEST_DATA_DIR, "infected", f) for f in os.listdir(os.path.join(TEST_DATA_DIR, "infected")) if f.endswith(('.png', '.jpg', '.jpeg'))]
#     not_infected_images = [os.path.join(TEST_DATA_DIR, "notinfected", f) for f in os.listdir(os.path.join(TEST_DATA_DIR, "notinfected")) if f.endswith(('.png', '.jpg', '.jpeg'))]

#     random_samples = random.sample(infected_images, min(num_samples//2, len(infected_images))) + \
#                      random.sample(not_infected_images, min(num_samples//2, len(not_infected_images)))

#     for img_path in random_samples:
#         prediction = predict_image(img_path)
#         print(f"📌 {os.path.basename(img_path)} - Predicted: {prediction}")

# # Run sample predictions
# print("\n🖼️ Testing Random Images from Test Dataset...")
# test_random_images()

# FastAPI Endpoints

@app.get("/")
def home():
    return {"message": "Welcome to Swin Transformer PCOS Classification API"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Predict status from uploaded image"""
    image_bytes = await file.read()
    image = io.BytesIO(image_bytes)

    prediction = predict_image(image)

    return {"prediction": f"The following image seems to be: {prediction}"}


# @app.get("/batch_predict/")
# def batch_predict():
#     results = {}
#     for label in ["infected", "not infected"]:
#         folder_path = os.path.join(TEST_DATA_DIR, label)
#         if not os.path.exists(folder_path):
#             continue
        
#         results[label] = []
#         for img_name in os.listdir(folder_path):
#             if not img_name.endswith(('.png', '.jpg', '.jpeg')):
#                 continue  # Skip non-image files
             
#             img_path = os.path.join(folder_path, img_name)
#             prediction = predict_image(img_path)
#             results[label].append({"image": img_name, "prediction": prediction})

#     return results

# Run the FastAPI application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)