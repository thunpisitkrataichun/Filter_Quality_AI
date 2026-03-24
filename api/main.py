import os
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile
from torchvision import transforms
from PIL import Image

app = FastAPI()

# 1. Setup Device & Paths
device = torch.device("cpu")  # ⭐ บน Render ใช้ CPU เท่านั้น

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "catordog_nn.pth")  

# 2. Model Definition
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(32 * 32 * 32, 128) 
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 3. Load Model
model = SimpleCNN()

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# 4. Preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

class_names = ['cat', 'dog']

# 5. Routes
@app.get("/")
def root():
    return {"status": "online", "model": "SimpleCNN", "device": str(device)}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image.verify()  # เช็คว่าเป็นรูป
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        img = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

        return {
            "class": class_names[pred.item()],
            "confidence": round(float(conf.item()), 4)
        }

    except Exception as e:
        return {"error": str(e)}