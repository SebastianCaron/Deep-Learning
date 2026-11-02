import sys
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.pretrained import ArtClassifierTinyBased
from src.loader import load_canonical_data, get_eval_transfomrm
from src.serv_mcp.utils import load_metadata_map


def build_gallery_index(
    data_dir: str = "data/Art Movement/",
    checkpoint_path: str = "checkpoints/best_model.pth",
    label_csv_path: str = "data/label_list.csv",
    output_path: str = "checkpoints/gallery_index.pt",
    batch_size: int = 64,
    device: str = None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", file=sys.stderr)

    # La data
    train_ds, val_ds, test_ds, classes, _ = load_canonical_data(data_dir=data_dir)
    print(f"Canonical classes ({len(classes)}): {classes}", file=sys.stderr)
    print(f"Gallery will index {len(val_ds)} validation artworks.", file=sys.stderr)

    # Le model
    model = ArtClassifierTinyBased(nb_classes=len(classes)).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # les metadata
    meta_map = load_metadata_map(Path(label_csv_path))

    # et puis les embeddings quoi
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    all_embeddings = []
    all_pred_classes = []
    all_confidences = []

    t0 = time.time()
    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc="Extracting embeddings"):
            imgs = imgs.to(device)
            feats = model.base_model(imgs)
            latents = model.latent_space(feats)
            logits = model.classifier(latents)
            probs = torch.softmax(logits, dim=-1)
            top_conf, top_cls = probs.max(dim=-1)
            all_embeddings.append(latents.cpu())
            all_pred_classes.append(top_cls.cpu())
            all_confidences.append(top_conf.cpu())

    embeddings = torch.cat(all_embeddings, dim=0)
    pred_classes = torch.cat(all_pred_classes, dim=0)
    confidences = torch.cat(all_confidences, dim=0)

    # Metadata pour les samples
    paths = []
    image_ids = []
    titles = []
    authors = []
    true_styles = []
    pred_styles = []

    for idx, (path_obj, cls_idx) in enumerate(val_ds.samples):
        p = Path(path_obj)
        paths.append(str(p))
        image_ids.append(p.stem)
        true_styles.append(classes[cls_idx])
        pred_styles.append(classes[pred_classes[idx].item()])

        if p.name in meta_map:
            author, title = meta_map[p.name]
        else:
            title = p.stem.split("##")[0]
            author = "Artiste inconnu"
        titles.append(title)
        authors.append(author)

    gallery_data = {
        "embeddings": embeddings,
        "pred_confidences": confidences,
        "image_ids": image_ids,
        "paths": paths,
        "titles": titles,
        "authors": authors,
        "true_styles": true_styles,
        "pred_styles": pred_styles,
        "classes": classes,
        "checkpoint_used": checkpoint_path
    }

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(gallery_data, out_p)
    duration = time.time() - t0
    print(f"Saved gallery index with {len(paths)} artworks to {output_path} in {duration:.1f}s.", file=sys.stderr)


if __name__ == "__main__":
    build_gallery_index()
