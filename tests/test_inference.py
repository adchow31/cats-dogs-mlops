import io

import torch
from PIL import Image

from src.inference_utils import preprocess_image_bytes, predict
from src.model import BaselineCNN


def _make_fake_image_bytes(size=(224, 224), color=(255, 0, 0)):
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_preprocess_image_bytes_returns_correct_shape():
    fake_bytes = _make_fake_image_bytes()
    tensor = preprocess_image_bytes(fake_bytes)

    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == torch.float32


def test_preprocess_image_bytes_handles_non_square_input():
    fake_bytes = _make_fake_image_bytes(size=(500, 300))
    tensor = preprocess_image_bytes(fake_bytes)

    assert tensor.shape == (1, 3, 224, 224)


def test_predict_returns_expected_keys():
    model = BaselineCNN(num_classes=2)
    model.eval()
    fake_bytes = _make_fake_image_bytes()

    result = predict(model, fake_bytes)

    assert "label" in result
    assert "confidence" in result
    assert "probabilities" in result
    assert result["label"] in ("Cat", "Dog")
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_probabilities_sum_to_one():
    model = BaselineCNN(num_classes=2)
    model.eval()
    fake_bytes = _make_fake_image_bytes()

    result = predict(model, fake_bytes)
    total = sum(result["probabilities"].values())

    assert abs(total - 1.0) < 1e-4