import random
from pathlib import Path
from PIL import Image
from collections import defaultdict

import torch
from torchvision.transforms import v2
from torchvision import datasets
from torch.utils.data import Dataset

TAXONOMY_MAPPING = {
    # 1. Renaissance (Italienne, Nordique, Allemande, Flamande, etc.)
    'Renaissance': 'Renaissance',
    'Northern Renaissance': 'Renaissance',
    'German Renaissance': 'Renaissance',
    'Italian Renaissance': 'Renaissance',
    'Early renaissance': 'Renaissance',
    'High Renaissance': 'Renaissance',
    'Florentine painting': 'Renaissance',
    'Venetian school': 'Renaissance',
    'Bolognese School': 'Renaissance',
    'Sienese School': 'Renaissance',
    'Paduan School': 'Renaissance',
    'School of Ferrara': 'Renaissance',
    'American Renaissance': 'Renaissance',
    'Spanish Renaissance': 'Renaissance',
    'French Renaissance': 'Renaissance',
    'Dutch and Flemish Renaissance painting': 'Renaissance',

    # 2. Baroque & Siècle d'or hollandais
    'Baroque': 'Baroque & Siècle d\'Or',
    'Dutch Golden Age': 'Baroque & Siècle d\'Or',
    'Caravaggisti': 'Baroque & Siècle d\'Or',
    'Utrecht Caravaggism': 'Baroque & Siècle d\'Or',
    'Antwerp school': 'Baroque & Siècle d\'Or',

    # 3. Rococo
    'Rococo': 'Rococo',

    # 4. Maniérisme
    'Mannerism': 'Maniérisme',

    # 5. Néoclassicisme & Art Académique
    'Neoclassicism': 'Néoclassicisme & Académisme',
    'Academic art': 'Néoclassicisme & Académisme',
    'Classicism': 'Néoclassicisme & Académisme',

    # 6. Romantisme & Écoles paysagistes
    'Romanticism': 'Romantisme & Paysagisme',
    'Hudson River School': 'Romantisme & Paysagisme',
    'Luminism': 'Romantisme & Paysagisme',
    'Tonalism': 'Romantisme & Paysagisme',
    'Sturm und Drang': 'Romantisme & Paysagisme',

    # 7. Réalisme & Naturalisme
    'Realism': 'Réalisme & Naturalisme',
    'American Realism': 'Réalisme & Naturalisme',
    'Naturalism': 'Réalisme & Naturalisme',
    'Barbizon school': 'Réalisme & Naturalisme',
    'Ashcan School': 'Réalisme & Naturalisme',
    'Hague School': 'Réalisme & Naturalisme',

    # 8. Réalisme Social
    'Social realism': 'Réalisme Social',
    'Socialist realism': 'Réalisme Social',

    # 9. Impressionnisme
    'Impressionism': 'Impressionnisme',
    'American Impressionism': 'Impressionnisme',
    'Amsterdam Impressionism': 'Impressionnisme',

    # 10. Post-Impressionnisme & Néo-Impressionnisme
    'Post-Impressionism': 'Post-Impressionnisme',
    'Neo-impressionism': 'Post-Impressionnisme',
    'Synthetism': 'Post-Impressionnisme',

    # 11. Art Nouveau & Symbolisme
    'Art Nouveau': 'Art Nouveau & Symbolisme',
    'Symbolism': 'Art Nouveau & Symbolisme',
    'Vienna Secession': 'Art Nouveau & Symbolisme',
    'Arts and Crafts movement': 'Art Nouveau & Symbolisme',
    'Les Nabis': 'Art Nouveau & Symbolisme',
    'Aestheticism': 'Art Nouveau & Symbolisme',

    # 12. Expressionnisme
    'Expressionism': 'Expressionnisme',
    'German Expressionism': 'Expressionnisme',
    'Der Blaue Reiter': 'Expressionnisme',
    'Die Brücke': 'Expressionnisme',
    'Neo-expressionism': 'Expressionnisme',

    # 13. Cubisme & Futurisme
    'Cubism': 'Cubisme & Futurisme',
    'Futurism': 'Cubisme & Futurisme',
    'Section d\'Or': 'Cubisme & Futurisme',

    # 14. Surréalisme & Dada
    'Surrealism': 'Surréalisme & Dada',
    'Dada': 'Surréalisme & Dada',
    'Magical Realism': 'Surréalisme & Dada',

    # 15. Art Abstrait & Géométrique
    'Abstract art': 'Art Abstrait & Géométrique',
    'Geometric abstraction': 'Art Abstrait & Géométrique',
    'Color Field': 'Art Abstrait & Géométrique',
    'Constructivism': 'Art Abstrait & Géométrique',
    'Suprematism': 'Art Abstrait & Géométrique',
    'De Stijl': 'Art Abstrait & Géométrique',
    'Abstract expressionism': 'Art Abstrait & Géométrique',

    # 16. Art Japonais
    'Ukiyo-e': 'Art Japonais (Ukiyo-e)',
    'Shin-hanga': 'Art Japonais (Ukiyo-e)',
    'Nihonga': 'Art Japonais (Ukiyo-e)',

    # 17. Art Gothique & Médiéval
    'Gothic art': 'Art Gothique & Médiéval',

    # 18. Art Conceptuel & Minimalisme
    'Conceptual art': 'Art Conceptuel & Minimalisme',
    'Minimalism': 'Art Conceptuel & Minimalisme',
    'Kinetic art': 'Art Conceptuel & Minimalisme',
    'Fluxus': 'Art Conceptuel & Minimalisme',

    # 19. Street Art & Art Public
    'Street art': 'Street Art & Art Public',
    'Public art': 'Street Art & Art Public',

    # 20. Art Contemporain
    'Contemporary art': 'Art Contemporain',
    'Postmodernism': 'Art Contemporain',
    'Young British Artists': 'Art Contemporain',

    # 21. Art Moderne (Avant-Garde)
    'Modern art': 'Art Moderne',
    'Modernism': 'Art Moderne',
    'American modernism': 'Art Moderne',
}

GENERIC_NAME = {'Art Moderne', 'Réalisme Social'}

class ArtDataset(Dataset):
    def __init__(self, samples, transform=None, classes=None):
        self.samples = samples
        self.transform = transform
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)} if classes else None
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, target = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert('RGB')
            
        if self.transform is not None:
            img = self.transform(img)
            
        return img, target

def get_training_transform(img_size = 288):
    return v2.Compose([
        v2.RandomResizedCrop(size=(img_size, img_size), scale=(0.7, 1.0), interpolation=v2.InterpolationMode.BICUBIC),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=15),
        v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.04),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        v2.RandomErasing(p=0.2, scale=(0.02, 0.2), value='random')
    ])

def get_eval_transfomrm(img_size = 288):
    val_resize = int(img_size * 256 / 224)
    return v2.Compose([
        v2.Resize(val_resize, interpolation=v2.InterpolationMode.BICUBIC),
        v2.CenterCrop(img_size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def load_canonical_data(data_dir="./data/Art Movement/", train_ratio=0.8, seed=42, img_size=288):
    
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Dossier de données introuvable : {data_dir.resolve()}")

    file_to_groups = defaultdict(set)
    file_to_path = {}

    for d in data_dir.iterdir():
        if not d.is_dir() or d.name not in TAXONOMY_MAPPING:
            continue
        group_name = TAXONOMY_MAPPING[d.name]
        for img in d.glob('*'):
            file_to_groups[img.name].add(group_name)
            file_to_path[img.name] = img

    class_to_paths = defaultdict(list)
    for fname, groups in file_to_groups.items():
        if len(groups) == 1:
            chosen = list(groups)[0]
        else:
            not_generic_name = [g for g in groups if g not in GENERIC_NAME]
            chosen = sorted(not_generic_name)[0] if not_generic_name else sorted(groups)[0]
        class_to_paths[chosen].append(file_to_path[fname])

    classes = sorted(class_to_paths.keys())
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    random.seed(seed)
    train_samples = []
    val_samples = []

    for cls in classes:
        paths = class_to_paths[cls]
        random.shuffle(paths)

        n_val = max(1, int(len(paths) * (1 - train_ratio)))
        cls_idx = class_to_idx[cls]

        val_paths = paths[:n_val]
        train_paths = paths[n_val:]

        for p in train_paths:
            train_samples.append((p, cls_idx))
        for p in val_paths:
            val_samples.append((p, cls_idx))

    train_tf = get_training_transform(img_size=img_size)
    val_tf = get_eval_transfomrm(img_size=img_size)

    train_dataset = ArtDataset(train_samples, transform=train_tf, classes=classes)
    val_dataset = ArtDataset(val_samples, transform=val_tf, classes=classes)

    return train_dataset, val_dataset, classes, class_to_paths



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

    train_tf = get_training_transform(img_size=img_size)
    val_tf = get_eval_transfomrm(img_size=img_size)
    
    train_samples = [base_dataset.samples[i] for i in train_indices]
    val_samples = [base_dataset.samples[i] for i in val_indices]
    train_dataset = ArtDataset(train_samples, transform=train_tf, classes=classes)
    val_dataset = ArtDataset(val_samples, transform=val_tf, classes=classes)

    return train_dataset, val_dataset, classes