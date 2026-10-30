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

    for i, (images, targets) in enumerate(loader):
        images, targets = images.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss

def train(model, loss, optimizer, train_loader, val_loader, nb_epoch = 30, checkpoint_dir = "./checkpoints/"):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history = []
    start_time = time.time()
    for epoch in range(1, nb_epoch + 1):
            epoch_start = time.time()
    
            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=loss,
                optimizer=optimizer,
            )
    
            epoch_dur = time.time() - epoch_start
            total_elapsed = time.time() - start_time
            remaining = (nb_epoch+1 - epoch) * (total_elapsed / epoch)
    
            epoch_stats = {
                "epoch": epoch,
                "train_loss": train_loss,
                "duration_sec": epoch_dur
            }
            history.append(epoch_stats)
    
            print(
                f"\n--- Fin Époque {epoch}/{nb_epoch+1} ({epoch_dur}) ---\n"
                f"  Temps -> Écoulé: {total_elapsed} | ETA: {remaining}"
            )
    
            with open(checkpoint_dir / "history.json", "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

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