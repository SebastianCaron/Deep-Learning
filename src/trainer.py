import time

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    start_time = time.time()

    running_loss = 0.0

    for i, (images, targets) in enumerate(loader):
        images, targets = images.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'elapsed : {time.time() - start_time}')
    return running_loss

