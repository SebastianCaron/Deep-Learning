import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from loader import get_datasets
from models.pretrained import ArtClassifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for i, (images, targets) in enumerate(loader):
        images, targets = images.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return 100. * correct / total

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    return 100. * correct / total

def train(model, loss, optimizer, train_loader, val_loader, nb_epoch = 30, checkpoint_dir = "./checkpoints/"):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history = []
    start_time = time.time()
    best_acc = 0
    for epoch in range(1, nb_epoch + 1):
            epoch_start = time.time()
    
            train_acc = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=loss,
                optimizer=optimizer,
            )
            val_acc = evaluate(model, val_loader)
            print(f"Epoch {epoch} | Train Acc: {train_acc}% | Val Acc: {val_acc}%")
    
            epoch_dur = time.time() - epoch_start
            total_elapsed = time.time() - start_time
            remaining = (nb_epoch+1 - epoch) * (total_elapsed / epoch)
    
            epoch_stats = {
                "epoch": epoch,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "duration_sec": epoch_dur
            }
            history.append(epoch_stats)
    
            print(
                f"\n--- Fin Époque {epoch}/{nb_epoch+1} ({epoch_dur}) ---\n"
                f"  Temps -> Écoulé: {total_elapsed} | ETA: {remaining}"
            )
    
            with open(checkpoint_dir / "history.json", "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), "best_model.pth")

    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
    }
    torch.save(payload, checkpoint_dir / "latest_model.pth")



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