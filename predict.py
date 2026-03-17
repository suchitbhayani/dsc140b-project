"""Prediction code for the DSC 140B SoCalGuessr project.

The autograder calls `predict(test_dir)`, where `test_dir` contains .jpg images.
This loads `model.pt` (weights + metadata) and returns {filename: city}.
"""

import pathlib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
from PIL import Image


def build_model(num_classes: int) -> nn.Module:
    model = mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


class TestImageDataset(Dataset):
    def __init__(self, root: pathlib.Path, transform):
        self.root = pathlib.Path(root)
        self.transform = transform
        self.paths = sorted(self.root.glob("*.jpg"))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return path.name, image


def predict(test_dir):
    """Load the saved model and predict a label for every image in `test_dir`.

    Parameters
    ----------
    test_dir : pathlib.Path
        Path to a directory containing .jpg test images.

    Returns
    -------
    dict[str, str]
        A dictionary mapping each image filename (e.g. "00001.jpg") to a predicted
        class label (e.g. "Los_Angeles").

    """
    test_dir = pathlib.Path(test_dir)

    payload = torch.load("model.pt", map_location="cpu")
    classes = payload["classes"]
    image_size = int(payload["image_size"])
    mean = tuple(payload["normalize"]["mean"])
    std = tuple(payload["normalize"]["std"])

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    dataset = TestImageDataset(test_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    model = build_model(len(classes))
    model.load_state_dict(payload["state_dict"])
    model.eval()

    predictions = {}
    with torch.no_grad():
        for names, images in loader:
            outputs = model(images)
            predicted = outputs.argmax(dim=1).tolist()
            for name, idx in zip(names, predicted):
                predictions[name] = classes[idx]

    return predictions


if __name__ == "__main__":
    preds = predict("./testdata")
    print("Predictions:")
    for filename, label in sorted(preds.items()):
        print(f"{filename}: {label}")
