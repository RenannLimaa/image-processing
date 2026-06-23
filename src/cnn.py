"""
CNN module — transfer learning baseline using MobileNetV2.

Uses torchvision's MobileNetV2 with a frozen backbone and a replaced
binary classifier head.  Designed to pair with the classical pipeline for
a direct performance comparison.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])

EVAL_TRANSFORMS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])

class TyreDataset(Dataset):
    """
    Wraps a list of image paths and integer labels (0=good, 1=defective).

    Images are loaded as RGB (H×W×3 uint8) and transformed to normalised
    float tensors ready for MobileNetV2.
    """

    def __init__(
        self,
        paths: list[str],
        labels: list[int],
        transform=None,
    ) -> None:
        if len(paths) != len(labels):
            raise ValueError("paths and labels must have the same length")
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        bgr = cv2.imread(self.paths[idx])
        if bgr is None:
            raise FileNotFoundError(f"Cannot read image: {self.paths[idx]}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            tensor = self.transform(rgb)
        else:
            tensor = transforms.ToTensor()(rgb)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return tensor, label

def build_model(num_classes: int = 2, freeze_backbone: bool = True) -> nn.Module:
    """
    Return a MobileNetV2 with:
      - Backbone initialised from ImageNet pre-trained weights (frozen by default)
      - Classifier head replaced with a Linear(1280 → num_classes) layer

    Parameters
    ----------
    num_classes : int
        Number of output classes (2 for binary good/defective).
    freeze_backbone : bool
        If True, all backbone parameters are frozen and only the head is trained.
        Set to False to fine-tune the entire network.
    """
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model

def get_device() -> torch.device:
    """Return the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Run one training epoch.

    Returns
    -------
    avg_loss : float
    accuracy : float
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference on a DataLoader.

    Returns
    -------
    y_true : np.ndarray of shape (N,)
    y_pred : np.ndarray of shape (N,)
    """
    model.eval()
    all_true: list[int] = []
    all_pred: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_pred.extend(preds.tolist())
            all_true.extend(labels.numpy().tolist())

    return np.array(all_true), np.array(all_pred)

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    n_epochs: int = 15,
    lr: float = 1e-3,
    checkpoint_path: str | Path | None = None,
) -> tuple[nn.Module, list[dict]]:
    """
    Full training loop with early-best-checkpoint saving based on val F1.

    Parameters
    ----------
    model : nn.Module
    train_loader, val_loader : DataLoader
    device : torch.device
    n_epochs : int
    lr : float
    checkpoint_path : path to save the best model weights

    Returns
    -------
    model : nn.Module  (best weights loaded)
    history : list of dicts with keys epoch, train_loss, train_acc, val_loss, val_acc
    """
    from sklearn.metrics import f1_score

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_f1 = -1.0
    best_state = None
    history = []

    for epoch in range(1, n_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)

        val_loss_total = 0.0
        val_correct = 0
        val_total = 0
        model.eval()
        all_true, all_pred = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss_total += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += images.size(0)
                all_true.extend(labels.cpu().numpy().tolist())
                all_pred.extend(preds.cpu().numpy().tolist())

        val_loss = val_loss_total / val_total
        val_acc  = val_correct / val_total
        val_f1   = f1_score(all_true, all_pred, zero_division=0)

        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc":  train_acc,
            "val_loss":   val_loss,
            "val_acc":    val_acc,
            "val_f1":     val_f1,
        })

        print(
            f"Epoch {epoch:02d}/{n_epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  val_f1={val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if checkpoint_path is not None:
                torch.save(best_state, checkpoint_path)
                print(f"  ✓ Saved best checkpoint (val_f1={best_f1:.4f})")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history
