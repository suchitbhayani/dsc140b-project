import csv
from pathlib import Path

import matplotlib.pyplot as plt

log_path = Path("training_log.csv")
if not log_path.exists():
    raise FileNotFoundError(
        "training_log.csv not found. Run `python -u train.py` first to generate it."
    )

epochs = []
train_loss = []
val_acc = []

with log_path.open("r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        epochs.append(int(row["epoch"]))
        train_loss.append(float(row["train_loss"]))
        val_acc.append(float(row["val_accuracy"]))

fig, ax1 = plt.subplots(figsize=(6, 4))

color_loss = "tab:blue"
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Training loss", color=color_loss)
ax1.plot(epochs, train_loss, marker="o", color=color_loss, label="Training loss")
ax1.tick_params(axis="y", labelcolor=color_loss)

ax2 = ax1.twinx()
color_acc = "tab:red"
ax2.set_ylabel("Validation accuracy", color=color_acc)
ax2.plot(epochs, val_acc, marker="s", linestyle="--", color=color_acc, label="Validation accuracy")
ax2.tick_params(axis="y", labelcolor=color_acc)

fig.suptitle("Training Loss and Validation Accuracy vs. Epoch")
fig.tight_layout()
fig.savefig("images/training_curves.png", dpi=300)
print("Saved images/training_curves.png")