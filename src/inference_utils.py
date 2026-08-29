import io

import torch
from PIL import Image
from torchvision import transforms

from src.model import BaselineCNN

# Must match train_ds.class_to_idx from the notebook: {'Cat': 0, 'Dog': 1}
IDX_TO_CLASS = {0: "Cat", 1: "Dog"}

_eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model(weights_path: str = "models/baseline_cnn.pt", device: str = "cpu") -> BaselineCNN:
    """Load the trained BaselineCNN with saved weights, ready for inference."""
    model = BaselineCNN(num_classes=2)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess_image_bytes(image_bytes: bytes) -> torch.Tensor:
    """Convert raw uploaded image bytes into a normalized model-ready tensor batch of size 1."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = _eval_tf(img)
    return tensor.unsqueeze(0)  # add batch dimension


def predict(model: BaselineCNN, image_bytes: bytes) -> dict:
    """Run inference on raw image bytes, return label + class probabilities."""
    input_tensor = preprocess_image_bytes(image_bytes)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)
    pred_idx = int(torch.argmax(probs).item())
    return {
        "label": IDX_TO_CLASS[pred_idx],
        "confidence": round(float(probs[pred_idx]), 4),
        "probabilities": {IDX_TO_CLASS[i]: round(float(p), 4) for i, p in enumerate(probs)},
    }