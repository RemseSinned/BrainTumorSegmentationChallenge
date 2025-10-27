import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from tqdm import tqdm
import os
import yaml
from datasets import BrainTumorDataset


def print_train_setup():
    print("Current device:", torch.cuda.current_device() if torch.cuda.is_available() else "CPU")
    print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")


def load_config(path="../configs/training.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def preprocess_labels(labels):
    """Ensure labels are in the correct range and shape"""
    labels = labels.unsqueeze(1) if labels.ndim == 4 else labels
    labels = labels.clone()
    labels[labels == 4] = 2  # map ET to 2 (BraTS convention)
    labels[labels > 2] = 2   # safety: ensure within [0, 2]
    return labels


def compute_region_dice(preds, labels):
    """Compute Dice for WT, TC, and ET regions"""
    dice_wt = DiceMetric(include_background=False, reduction="mean")
    dice_tc = DiceMetric(include_background=False, reduction="mean")
    dice_et = DiceMetric(include_background=False, reduction="mean")

    lab_np = labels.cpu().numpy()
    pred_np = preds.cpu().numpy()

    for i in range(lab_np.shape[0]):
        gt = lab_np[i, 0]
        pr = pred_np[i, 0]

        gt_wt, pr_wt = (gt > 0).astype(float), (pr > 0).astype(float)
        gt_tc, pr_tc = ((gt == 1) | (gt == 4)).astype(float), ((pr == 1) | (pr == 4)).astype(float)
        gt_et, pr_et = (gt == 4).astype(float), (pr == 4).astype(float)

        dice_wt(y_pred=torch.tensor(pr_wt[None, None]), y=torch.tensor(gt_wt[None, None]))
        dice_tc(y_pred=torch.tensor(pr_tc[None, None]), y=torch.tensor(gt_tc[None, None]))
        dice_et(y_pred=torch.tensor(pr_et[None, None]), y=torch.tensor(gt_et[None, None]))

    dice_scores = {
        "WT": dice_wt.aggregate().item(),
        "TC": dice_tc.aggregate().item(),
        "ET": dice_et.aggregate().item(),
    }

    return dice_scores


def validate_one_epoch(model, val_loader, loss_fn, device):
    model.eval()
    val_loss = 0.0
    all_dice = {"WT": 0.0, "TC": 0.0, "ET": 0.0}
    post_pred, post_label = AsDiscrete(threshold=0.5), AsDiscrete(threshold=0.5)

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            labels = preprocess_labels(labels)
            loss = loss_fn(outputs, labels.float())
            val_loss += loss.item()

            preds = post_pred(outputs)
            labs = post_label(labels)
            dice_scores = compute_region_dice(preds, labs)

            for k in all_dice:
                all_dice[k] += dice_scores[k]

    # average dice and loss
    num_batches = len(val_loader)
    avg_dice = {k: v / num_batches for k, v in all_dice.items()}
    avg_loss = val_loss / num_batches

    return avg_loss, avg_dice


def train_one_epoch(model, train_loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(train_loader, desc="Training", leave=False)

    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        labels = preprocess_labels(labels)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels.long())

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    return total_loss / len(train_loader)


def train():
    print_train_setup()
    cfg = load_config()
    device = torch.device(cfg["training"]["device"])
    os.makedirs(cfg["training"]["checkpoint_dir"], exist_ok=True)
    writer = SummaryWriter(cfg["training"]["log_dir"])

    # --- Data ---
    train_ds = BrainTumorDataset(split="train", augment=True)
    val_ds = BrainTumorDataset(split="val", augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
    )

    # --- Model, loss, optimizer ---
    model = UNet(**cfg["model"]).to(device)
    loss_fn = DiceLoss(include_background=False, to_onehot_y=True, sigmoid=False)
    optimizer = optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])

    # --- Loop ---
    best_val_dice = 0.0
    epochs = cfg["training"]["epochs"]

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, dice_scores = validate_one_epoch(model, val_loader, loss_fn, device)

        # Logging
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        for k, v in dice_scores.items():
            writer.add_scalar(f"Dice/{k}", v, epoch)

        mean_dice = sum(dice_scores.values()) / len(dice_scores)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Mean Dice: {mean_dice:.4f}")
        print(f"Dice - WT: {dice_scores['WT']:.4f}, TC: {dice_scores['TC']:.4f}, ET: {dice_scores['ET']:.4f}")

        # Save best checkpoint
        if mean_dice > best_val_dice:
            best_val_dice = mean_dice
            ckpt_path = os.path.join(cfg["training"]["checkpoint_dir"], f"best_model_epoch{epoch + 1}.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": dice_scores,
            }, ckpt_path)
            print(f"Saved new best model: {ckpt_path}")

    writer.close()
