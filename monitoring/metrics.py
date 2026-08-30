"""
Post-deployment model performance tracking.
Sends a small batch of labeled test images to the live API and reports accuracy.
Run manually after deployment: python monitoring/metrics.py
"""
import os

import requests

API_URL = os.environ.get("API_URL", "http://localhost:8000")

# Small labeled batch — point these at a handful of real images from your test set
TEST_SAMPLES = [
    {"path": "data/raw/test/cats/10001.jpg", "true_label": "Cat"},
    {"path": "data/raw/test/cats/10004.jpg", "true_label": "Cat"},
    {"path": "data/raw/test/dogs/10001.jpg", "true_label": "Dog"},
    {"path": "data/raw/test/dogs/10004.jpg", "true_label": "Dog"},
]


def run_performance_check():
    correct = 0
    results = []

    for sample in TEST_SAMPLES:
        if not os.path.exists(sample["path"]):
            print(f"SKIP (not found): {sample['path']}")
            continue

        with open(sample["path"], "rb") as f:
            filename = os.path.basename(sample["path"])
            resp = requests.post(f"{API_URL}/predict", files={"file": (filename, f, "image/jpeg")})
        if resp.status_code != 200:
            print(f"FAILED request for {sample['path']}: {resp.status_code}")
            continue

        pred = resp.json()
        is_correct = pred["label"] == sample["true_label"]
        correct += int(is_correct)
        results.append({
            "file": sample["path"],
            "true_label": sample["true_label"],
            "predicted": pred["label"],
            "confidence": pred["confidence"],
            "correct": is_correct,
        })
        print(f"{sample['path']}: true={sample['true_label']} pred={pred['label']} "
              f"conf={pred['confidence']} {'✓' if is_correct else '✗'}")

    if results:
        accuracy = correct / len(results)
        print(f"\nPost-deployment accuracy on {len(results)} samples: {accuracy:.2%}")
    else:
        print("No valid samples were tested.")


if __name__ == "__main__":
    run_performance_check()