from collections import defaultdict
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from loader import get_datasets
from models.pretrained import ArtClassifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRINT_FREQUENCE = 20

def format_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"

def compute_accuracy(outputs: torch.Tensor, targets: torch.Tensor):
    _, top5_preds = outputs.topk(5, dim=1, largest=True, sorted=True)
    top1_preds = top5_preds[:, 0]
    top1_correct = (top1_preds == targets).sum().item()
    top5_correct = (top5_preds == targets.unsqueeze(1)).sum().item()
    return top1_correct, top5_correct, top1_preds, targets

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    start_time = time.time()

    running_loss = 0.0
    correct_top1 = 0.0
    correct_top5 = 0.0
    total = 0

    for i, (images, targets) in enumerate(loader):
        images, targets = images.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)
        batch_size = targets.size(0)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        top1, top5, preds, targs = compute_accuracy(outputs.detach(), targets)
        correct_top1 += top1
        correct_top5 += top5
        total += batch_size

        if (i + 1) % PRINT_FREQUENCE == 0 or (i + 1) == len(loader):
            elapsed = time.time() - start_time
            speed = total / elapsed if elapsed > 0 else 0.0
            remaining = len(loader) - (i + 1)
            eta_sec = remaining * (elapsed / (i + 1))
            print(
                f"Étape [{i+1:4d}/{len(loader):4d}] | "
                f"Loss: {loss.item():.4f} | Top-1: {100.0*top1/batch_size:5.1f}% | Top-5: {100.0*top5/batch_size:5.1f}% | "
                f"Vitesse: {speed:5.1f} img/s | ETA: {format_time(eta_sec)}"
            )

    return running_loss / total, 100.0 * correct_top1 / total, 100.0 * correct_top5 / total

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct_top1 = 0.0
    correct_top5 = 0.0
    total = 0

    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    for images, targets in loader:
        images, targets = images.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)
        batch_size = targets.size(0)

        outputs = model(images)

        loss = criterion(outputs, targets)

        running_loss += loss.item() * batch_size
        top1, top5, preds, targs = compute_accuracy(outputs, targets)
        correct_top1 += top1
        correct_top5 += top5
        total += batch_size

        for p, t in zip(preds.tolist(), targs.tolist()):
            class_total[t] += 1
            if p == t:
                class_correct[t] += 1

    epoch_loss = running_loss / total
    micro_acc = 100.0 * correct_top1 / total
    top5_acc = 100.0 * correct_top5 / total

    recalls = [class_correct[c] / class_total[c] for c in class_total if class_total[c] > 0]
    macro_acc = 100.0 * sum(recalls) / len(recalls) if recalls else 0.0

    return epoch_loss, micro_acc, macro_acc, top5_acc, class_correct, class_total

def train(model, loss, optimizer, train_loader, val_loader, nb_epoch = 30, checkpoint_dir = "./checkpoints/"):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history = []
    start_time = time.time()
    best_acc = 0
    for epoch in range(1, nb_epoch + 1):
            epoch_start = time.time()
    
            train_loss, train_top1, train_top5 = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=loss,
                optimizer=optimizer,
            )
    
            val_loss, micro_acc, macro_acc, top5_acc, class_correct, class_total = evaluate(
                model=model,
                loader=val_loader,
                criterion=loss,
            )
    
            epoch_dur = time.time() - epoch_start
            total_elapsed = time.time() - start_time
            remaining = (nb_epoch+1 - epoch) * (total_elapsed / epoch)
    
            epoch_stats = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_top1": train_top1,
                "train_top5": train_top5,
                "val_loss": val_loss,
                "val_micro_top1": micro_acc,
                "val_macro_top1": macro_acc,
                "val_top5": top5_acc,
                "duration_sec": epoch_dur
            }
            history.append(epoch_stats)
    
            print(
                f"\n--- Fin Époque {epoch}/{nb_epoch+1} ({format_time(epoch_dur)}) ---\n"
                f"  Train -> Loss: {train_loss:.4f} | Top-1: {train_top1:5.2f}% | Top-5: {train_top5:5.2f}%\n"
                f"  Val   -> Loss: {val_loss:.4f} | Micro Top-1: {micro_acc:5.2f}% | Macro Top-1: {macro_acc:5.2f}% | Top-5: {top5_acc:5.2f}%\n"
                f"  Temps -> Écoulé: {format_time(total_elapsed)} | ETA: {format_time(remaining)}"
            )
    
            payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_micro_top1": micro_acc,
                "val_macro_top1": macro_acc,
                "val_top5": top5_acc,
                "val_loss": val_loss,
            }
            torch.save(payload, checkpoint_dir / "latest_model.pth")
    
            with open(checkpoint_dir / "history.json", "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

            if micro_acc > best_micro_acc:
                best_micro_acc = micro_acc
                torch.save(payload, checkpoint_dir / "best_model.pth")
                print(f"  >>> Nouveau record Micro Top-1 : {best_micro_acc:5.2f}% (Sauvegardé !) <<<")
    
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(payload, checkpoint_dir / "best_loss_model.pth")



if __name__ == "__main__":
    train_dt, val_dt, classes = get_datasets("./data/Art Movement/")
    
    BATCH_SIZE = 16
    train_loader = DataLoader(train_dt, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dt, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = ArtClassifier(len(classes))
    model = model.to(DEVICE)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train(
        model=model, 
        loss=loss, 
        optimizer=optimizer, 
        train_loader=train_loader, 
        val_loader=val_loader,
        nb_epoch=15
    )