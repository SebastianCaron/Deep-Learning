import random
from pathlib import Path
from collections import defaultdict

import torch
from torchvision.transforms import v2
from torchvision import datasets

class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.classes = getattr(dataset, 'classes', None)
        self.class_to_idx = getattr(dataset, 'class_to_idx', None)

    def __getitem__(self, idx):
        img, target = self.dataset[self.indices[idx]]
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.indices)

def get_datasets(data_dir=None, train_ratio=0.8, seed=42, img_size=288):

    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dossier de données introuvable : {data_dir.resolve()}"
        )

    base_dataset = datasets.ImageFolder(root=str(data_dir))
    classes = base_dataset.classes

    random.seed(seed)
    class_to_indices = defaultdict(list)
    for idx, (_, target) in enumerate(base_dataset.samples):
        class_to_indices[target].append(idx)

    train_indices = []
    val_indices = []
    for target, indices in class_to_indices.items():
        random.shuffle(indices)
        if len(indices) == 1:
            train_indices.append(indices[0])
        else:
            n_val = max(1, int(len(indices) * (1 - train_ratio)))
            if n_val >= len(indices):
                n_val = len(indices) - 1
            val_indices.extend(indices[:n_val])
            train_indices.extend(indices[n_val:])

    train_tf = v2.Compose([
        v2.RandomResizedCrop(size=(img_size, img_size), scale=(0.7, 1.0), interpolation=v2.InterpolationMode.BICUBIC),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=15),
        v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.04),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        v2.RandomErasing(p=0.2, scale=(0.02, 0.2), value='random')
    ])

    val_resize = int(img_size * 256 / 224)
    val_tf = v2.Compose([
        v2.Resize(val_resize, interpolation=v2.InterpolationMode.BICUBIC),
        v2.CenterCrop(img_size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = TransformedDataset(base_dataset, train_indices, transform=train_tf)
    val_dataset = TransformedDataset(base_dataset, val_indices, transform=val_tf)

    return train_dataset, val_dataset, classes