import json
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from src.loader import load_canonical_data
    from src.models.pretrained import ArtClassifierTinyBased
    from src.trainer import DEVICE
except ModuleNotFoundError:
    from loader import load_canonical_data
    from models.pretrained import ArtClassifierTinyBased
    from trainer import DEVICE


def evaluate_mcp_similarity(
    gallery_path: str = "./checkpoints/gallery_index.pt",
    checkpoint_path: str = "./checkpoints/best_model.pth",
    data_dir: str = "./data/Art Movement/",
    output_path: str = "./rapports/mcp_precision_at_k.json",
    batch_size: int = 64,
    device: str = DEVICE,
):
    print("=" * 80)
    print("ÉVALUATION DE LA RECHERCHE PAR SIMILARITÉ MCP (PRECISION@K)")
    print(f"Galerie : {gallery_path} | Modèle : {checkpoint_path}")
    print("=" * 80)

    gallery = torch.load(gallery_path, map_location=device, weights_only=False)
    gallery_embs = gallery["embeddings"].to(device)
    gallery_styles = gallery["true_styles"]
    gallery_embs_norm = F.normalize(gallery_embs, p=2, dim=1)

    _, _, test_dt, classes, _ = load_canonical_data(data_dir=data_dir)
    test_loader = DataLoader(test_dt, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = ArtClassifierTinyBased(len(classes)).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    k_values = [1, 3, 5, 10]
    precisions = {k: [] for k in k_values}

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            feats = model.base_model(images)
            latents = model.latent_space(feats)
            latents_norm = F.normalize(latents, p=2, dim=1)

            sim_matrix = torch.mm(latents_norm, gallery_embs_norm.t())
            top10_scores, top10_indices = torch.topk(sim_matrix, k=max(k_values), dim=1)

            top10_indices = top10_indices.cpu().numpy()
            targets = targets.cpu().numpy()

            for i in range(len(targets)):
                true_style = classes[targets[i]]
                neighbor_styles = [gallery_styles[idx] for idx in top10_indices[i]]

                matches = [1 if s == true_style else 0 for s in neighbor_styles]
                for k in k_values:
                    precisions[k].append(sum(matches[:k]) / float(k))

    results = {
        "nb_query_samples": len(test_dt),
        "gallery_size": len(gallery_styles),
        "metrics": {
            f"Precision@{k}": round(sum(precisions[k]) / len(precisions[k]) * 100.0, 2)
            for k in k_values
        },
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nRésultats Precision@k sur {len(test_dt)} requêtes inédites :")
    for k in k_values:
        print(f"  Precision@{k:<2} : {results['metrics'][f'Precision@{k}']:.2f}%")
    print(f"\nRésultats sauvegardés dans : {out_file.resolve()}\n")

    return results


if __name__ == "__main__":
    evaluate_mcp_similarity()
