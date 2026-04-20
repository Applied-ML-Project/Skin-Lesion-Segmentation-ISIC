import torch
import numpy as np
import random
import config
from dataset import get_loaders
from attention_unet import AttentionUNet
from transunet import TransUNet
from train import train_model
from evaluate import evaluate_model, print_comparison
from visualize import plot_predictions, plot_training_curves


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    set_seed(config.SEED)
    train_loader, val_loader, test_loader, test_ds = get_loaders()

    # --- Attention U-Net ---
    att_unet = AttentionUNet(in_channels=3, out_channels=1)
    att_unet, hist_att, time_att = train_model(att_unet, train_loader, val_loader, model_name="AttentionUNet")

    # --- TransUNet ---
    trans_unet = TransUNet(img_size=config.IMAGE_SIZE, in_channels=3, out_channels=1)
    trans_unet, hist_trans, time_trans = train_model(trans_unet, train_loader, val_loader, model_name="TransUNet")

    # --- Evaluation ---
    results_att, _, _ = evaluate_model(att_unet, test_loader, model_name="Attention U-Net")
    results_trans, _, _ = evaluate_model(trans_unet, test_loader, model_name="TransUNet")

    results_att["train_time"] = time_att
    results_trans["train_time"] = time_trans

    print_comparison([results_att, results_trans])

    # --- Visualizations ---
    plot_predictions(att_unet, test_ds, "Attention_UNet", n=8)
    plot_predictions(trans_unet, test_ds, "TransUNet", n=8)
    plot_training_curves(hist_att, hist_trans)

    print("Done!")


if __name__ == "__main__":
    main()
