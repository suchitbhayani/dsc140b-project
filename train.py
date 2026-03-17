"""Training code for the DSC 140B SoCalGuessr project.

Trains a small *pretrained* vision model (MobileNetV3-Small) on the training images in
`./data` and saves `model.pt` containing weights + metadata needed by `predict.py`.
"""

import pathlib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image
import time


# configuration ------------------------------------------------------------------------

# the path to the directory containing the training data
TRAIN_DIR = pathlib.Path("./data")

# the six classes of cities in our dataset
CLASSES = sorted(
    [
        "Anaheim",
        "Bakersfield",
        "Los_Angeles",
        "Riverside",
        "SLO",
        "San_Diego",
    ]
)

# we will later use this to convert a string class (like "Los_Angeles") to a numerical
# label (like 2), since PyTorch models work with numerical labels rather than strings.
CLASS_TO_NUMBER = {name: i for i, name in enumerate(CLASSES)}

# ImageNet-style preprocessing for pretrained weights.
# We'll train at 224x224 for best transfer performance.
IMAGE_SIZE = 224

BATCH_SIZE = 64
LEARNING_RATE_HEAD = 1e-3
EPOCHS = 6

# the percentage of the training data to set aside as a validation set.
VALIDATION_FRACTION = 0.2

SEED = 140


# dataset ------------------------------------------------------------------------------

# this is a custom dataset loader that loads the images from disk and applies any
# transformations we specify.

class SoCalDataset(Dataset):
    """Loads images from the training set."""

    def __init__(self, root, transform=None):
        self.root = pathlib.Path(root)
        self.transform = transform
        self.samples = []  # list of (path, label_index)

        # this determines the class label of an image based on its filename. For
        # example, the image "Los_Angeles-123.jpg" would be labeled as "Los_Angeles". We
        # then convert that string to a numerical label using the CLASS_TO_INDEX
        # dictionary defined above.
        for path in sorted(self.root.glob("*.jpg")):
            label = path.name.rsplit("-", 1)[0]
            self.samples.append((path, CLASS_TO_NUMBER[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# model --------------------------------------------------------------------------------

def build_model(num_classes: int) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)
    # Replace final classifier layer.
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


# training -----------------------------------------------------------------------------


def main():
    start_time = time.time()
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, (torch.get_num_threads() // 2)))

    weights = MobileNet_V3_Small_Weights.DEFAULT
    # Some torchvision builds don't expose mean/std in weights.meta.
    # MobileNetV3 ImageNet weights use standard ImageNet normalization.
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    print("Loading dataset...", flush=True)
    # Step 2) split the data into a train set and a validation set, and create data
    # loaders for each. See DSC 80 for why doing this is important!
    full_dataset = SoCalDataset(TRAIN_DIR, transform=None)
    val_size = int(len(full_dataset) * VALIDATION_FRACTION)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = eval_transform

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Step 3) define the model, loss function, and optimizer. The model is a simple
    # logistic regression model defined above. The loss function is cross-entropy loss,
    # which is commonly used for multi-class classification problems. The optimizer is
    # Adam.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(CLASSES)).to(device)

    criterion = nn.CrossEntropyLoss()
    # Freeze everything except the final Linear layer.
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier[-1].parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(
        model.classifier[-1].parameters(), lr=LEARNING_RATE_HEAD, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, EPOCHS))

    # Step 4) the training loop.

    def save_checkpoint():
        payload = {
            "state_dict": model.state_dict(),
            "classes": CLASSES,
            "image_size": IMAGE_SIZE,
            "normalize": {"mean": mean, "std": std},
            "arch": "mobilenet_v3_small_frozen_backbone",
        }
        torch.save(payload, "model.pt")

    print(f"Training on device={device} ...", flush=True)
    for epoch in range(EPOCHS):
        total_loss = 0.0
        correct = 0
        total = 0

        for step, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

            if step % 50 == 0:
                print(
                    f"Epoch {epoch + 1}/{EPOCHS} step {step}/{len(train_loader)} "
                    f"loss={loss.item():.4f}",
                    flush=True,
                )

        avg_loss = total_loss / total
        accuracy = correct / total

        # evaluate on the validation set
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                val_total += images.size(0)
        val_accuracy = val_correct / val_total
        model.train()
        scheduler.step()

        print(
            f"Epoch {epoch + 1}/{EPOCHS}  "
            f"loss: {avg_loss:.4f}  "
            f"accuracy: {accuracy:.4f}  "
            f"val_accuracy: {val_accuracy:.4f}"
        )

        save_checkpoint()
        print("Saved model.pt", flush=True)

    elapsed_s = time.time() - start_time
    print(f"Total training time: {elapsed_s/60:.2f} minutes", flush=True)

if __name__ == "__main__":
    main()
