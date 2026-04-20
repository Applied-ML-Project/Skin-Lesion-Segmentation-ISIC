import time
import csv
import copy
import torch
from tqdm import tqdm
import config
from losses import DiceBCELoss
from metrics import dice_score


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    running_dice = 0.0

    for images, masks in tqdm(loader, desc="Train", leave=False):
        images, masks = images.to(device), masks.to(device)
        preds = model(images)
        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        pred_np = (preds > 0.5).float()
        for i in range(images.size(0)):
            running_dice += dice_score(
                pred_np[i, 0].cpu().numpy(),
                masks[i, 0].cpu().numpy(),
            )

    n = len(loader.dataset)
    return running_loss / n, running_dice / n


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0

    for images, masks in tqdm(loader, desc="Val", leave=False):
        images, masks = images.to(device), masks.to(device)
        preds = model(images)
        loss = criterion(preds, masks)

        running_loss += loss.item() * images.size(0)
        pred_np = (preds > 0.5).float()
        for i in range(images.size(0)):
            running_dice += dice_score(
                pred_np[i, 0].cpu().numpy(),
                masks[i, 0].cpu().numpy(),
            )

    n = len(loader.dataset)
    return running_loss / n, running_dice / n


def train_model(model, train_loader, val_loader, model_name="model"):
    device = torch.device(config.DEVICE)
    model = model.to(device)

    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5, min_lr=config.MIN_LR)

    best_dice = 0.0
    patience_counter = 0
    best_weights = None
    history = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": [], "lr": []}

    print(f"\n{'='*50}")
    print(f"Training {model_name}")
    print(f"{'='*50}")

    start_time = time.time()

    for epoch in range(config.EPOCHS):
        train_loss, train_dice = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_dice = validate(model, val_loader, criterion, device)

        scheduler.step(val_dice)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_dice"].append(train_dice)
        history["val_dice"].append(val_dice)

        lr = optimizer.param_groups[0]["lr"]
        history["lr"].append(lr)
        print(
            f"Epoch {epoch+1}/{config.EPOCHS} | "
            f"Train Loss: {train_loss:.4f} Dice: {train_dice:.4f} | "
            f"Val Loss: {val_loss:.4f} Dice: {val_dice:.4f} | "
            f"LR: {lr:.2e}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            patience_counter = 0
            best_weights = copy.deepcopy(model.state_dict())
            ckpt_path = f"{config.CHECKPOINT_DIR}/{model_name}_best.pth"
            torch.save(best_weights, ckpt_path)
            print(f"  -> Saved best model (Dice: {best_dice:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    train_time = time.time() - start_time
    model.load_state_dict(best_weights)
    print(f"Training time: {train_time:.1f}s | Best Val Dice: {best_dice:.4f}")

    csv_path = f"{config.OUTPUT_DIR}/{model_name}_training_log.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "train_dice", "val_dice", "lr"])
        for i in range(len(history["train_loss"])):
            writer.writerow([
                i + 1,
                f"{history['train_loss'][i]:.6f}",
                f"{history['val_loss'][i]:.6f}",
                f"{history['train_dice'][i]:.6f}",
                f"{history['val_dice'][i]:.6f}",
                f"{history['lr'][i]:.2e}",
            ])
    print(f"Saved training log: {csv_path}")

    return model, history, train_time
