import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import config


MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def denormalize(img_tensor):
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * STD + MEAN
    return np.clip(img, 0, 1)


@torch.no_grad()
def plot_predictions(model, dataset, model_name, n=8, save=True):
    device = torch.device(config.DEVICE)
    model = model.to(device)
    model.eval()

    indices = np.random.choice(len(dataset), min(n, len(dataset)), replace=False)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))

    for row, idx in enumerate(indices):
        img, mask = dataset[idx]
        pred = model(img.unsqueeze(0).to(device))
        pred = (pred > 0.5).float()

        axes[row, 0].imshow(denormalize(img))
        axes[row, 0].set_title("Input Image")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(mask[0].cpu().numpy(), cmap="gray")
        axes[row, 1].set_title("Ground Truth")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(pred[0, 0].cpu().numpy(), cmap="gray")
        axes[row, 2].set_title(f"{model_name} Prediction")
        axes[row, 2].axis("off")

    plt.suptitle(f"Qualitative Results — {model_name}", fontsize=16, y=1.01)
    plt.tight_layout()

    if save:
        path = f"{config.FIGURE_DIR}/{model_name}_predictions.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.show()


def plot_training_curves(history1, history2, name1="Attention U-Net", name2="TransUNet", save=True):
    # Set a professional seaborn theme for LaTeX reports
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2, 
                  rc={"lines.linewidth": 2, "axes.edgecolor": "black"})

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    axes[0].plot(history1["train_loss"], label=f"{name1} Train", color="blue", alpha=0.9)
    axes[0].plot(history1["val_loss"], label=f"{name1} Val", color="blue", linestyle="--", alpha=0.9)
    axes[0].plot(history2["train_loss"], label=f"{name2} Train", color="red", alpha=0.9)
    axes[0].plot(history2["val_loss"], label=f"{name2} Val", color="red", linestyle="--", alpha=0.9)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(frameon=True, shadow=True)

    # Dice plot
    axes[1].plot(history1["train_dice"], label=f"{name1} Train", color="blue", alpha=0.9)
    axes[1].plot(history1["val_dice"], label=f"{name1} Val", color="blue", linestyle="--", alpha=0.9)
    axes[1].plot(history2["train_dice"], label=f"{name2} Train", color="red", alpha=0.9)
    axes[1].plot(history2["val_dice"], label=f"{name2} Val", color="red", linestyle="--", alpha=0.9)
    axes[1].set_title("Dice Score")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice")
    axes[1].set_ylabel("Dice")
    axes[1].legend(frameon=True, shadow=True)

    plt.tight_layout()

    if save:
        path = f"{config.FIGURE_DIR}/training_curves.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.show()


def load_history_from_csv(csv_path):
    history = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": [], "lr": []}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            history["train_loss"].append(float(row["train_loss"]))
            history["val_loss"].append(float(row["val_loss"]))
            history["train_dice"].append(float(row["train_dice"]))
            history["val_dice"].append(float(row["val_dice"]))
            history["lr"].append(float(row["lr"]))
    return history


def plot_training_curves_from_csv(csv1, csv2, name1="Attention U-Net", name2="TransUNet", save=True):
    h1 = load_history_from_csv(csv1)
    h2 = load_history_from_csv(csv2)
    plot_training_curves(h1, h2, name1, name2, save)
