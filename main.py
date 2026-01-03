from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
from fastapi import HTTPException

# --- Define the same model architecture ---
class MNISTCNN(nn.Module):
    def __init__(self):
        super(MNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- Load model ---
model = MNISTCNN()
model.load_state_dict(torch.load("mnist_cnn.pt", map_location="cpu"))
model.eval()

app = FastAPI()

class DigitInput(BaseModel):
    pixels: list   # 28x28 flattened list (784 values)

@app.get("/")
def home():
    return {"message": "MNIST API is running!"}

@app.post("/predict")
def predict(data: DigitInput):

    n = len(data.pixels)
    if n != 784:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 784 pixels, got {n}"
        )

    x = torch.tensor(data.pixels, dtype=torch.float32)
    x = x.view(1, 1, 28, 28)

    with torch.no_grad():
        output = model(x)
        pred = output.argmax(dim=1).item()

    return {"prediction": int(pred)}