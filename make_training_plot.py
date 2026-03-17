import matplotlib.pyplot as plt

# Epoch indices (1-based)
epochs = [1, 2, 3, 4, 5, 6]

train_loss = [1.1435, 0.8309, 0.7601, 0.7217, 0.7012, 0.6861]
val_acc =   [0.6509, 0.7179, 0.7266, 0.7337, 0.7309, 0.7315]

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
fig.savefig("training_curves.png", dpi=300)
print("Saved training_curves.png")