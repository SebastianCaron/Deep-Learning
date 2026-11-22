from pathlib import Path
import torch
import torch.nn.functional as F

from src.serv_mcp.mcp_model_interface import ModelEngine


def predict_style(image_path_or_id: str) -> dict:
    engine = ModelEngine.get_instance()
    path, gal_idx = engine.resolve_image(image_path_or_id)
    if path is None:
        return {"error": f"Image '{image_path_or_id}' introuvable."}
    meta = engine.get_metadata(path, gal_idx)

    tensor = engine.load_tensor(path)
    _, probs = engine.extract_embedding_and_probs(tensor)

    top_k = min(5, engine.nb_classes)
    top_probs, top_indices = torch.topk(probs, k=top_k)

    top_styles = [
        {"style": engine.classes[idx.item()], "probability": round(prob.item(), 4)}
        for prob, idx in zip(top_probs, top_indices)
    ]

    predicted_style = top_styles[0]["style"]
    confidence = top_styles[0]["probability"]

    return {
        "image_id": path.stem,
        "image_path": str(path),
        "painting_name": meta["painting_name"],
        "author_name": meta["author_name"],
        "true_style": meta.get("true_style"),
        "predicted_style": predicted_style,
        "confidence": confidence,
        "is_correct": (predicted_style == meta.get("true_style")) if meta.get("true_style") else None,
        "top_5_styles": top_styles,
    }


def find_similar_artworks(image_path_or_id: str, k: int = 5) -> dict:
    engine = ModelEngine.get_instance()
    path, gal_idx = engine.resolve_image(image_path_or_id)
    if path is None:
        return {"error": f"Image '{image_path_or_id}' introuvable."}
    meta = engine.get_metadata(path, gal_idx)

    tensor = engine.load_tensor(path)
    query_emb, _ = engine.extract_embedding_and_probs(tensor)

    sims = F.cosine_similarity(query_emb, engine.gallery_embeddings, dim=-1).cpu()

    if gal_idx is not None:
        sims[gal_idx] = -1.0

    top_scores, top_indices = torch.topk(sims, k=min(k, len(sims)))

    neighbors = []
    for score, idx in zip(top_scores, top_indices):
        i = idx.item()
        neighbors.append({
            "image_id": engine.gallery["image_ids"][i],
            "painting_name": engine.gallery["titles"][i],
            "author_name": engine.gallery["authors"][i],
            "style": engine.gallery["true_styles"][i],
            "predicted_style": engine.gallery["pred_styles"][i],
            "cosine_similarity": round(score.item(), 4),
            "similarity_score": round(score.item(), 4),
            "similarity_scale": "[-1.0, 1.0]",
            "path": engine.gallery["paths"][i],
        })

    return {
        "query_artwork": {
            "image_id": path.stem,
            "painting_name": meta["painting_name"],
            "author_name": meta["author_name"],
            "style": meta.get("true_style"),
        },
        "k": k,
        "total_gallery_size": len(engine.gallery["paths"]),
        "metric": "cosine_similarity (valeur entre -1.0 et 1.0, non assimilable à une probabilité)",
        "similar_artworks": neighbors,
    }


def find_cross_style_neighbors(image_path_or_id: str, k: int = 5) -> dict:
    engine = ModelEngine.get_instance()
    path, gal_idx = engine.resolve_image(image_path_or_id)
    if path is None:
        return {"error": f"Image '{image_path_or_id}' introuvable."}
    meta = engine.get_metadata(path, gal_idx)

    tensor = engine.load_tensor(path)
    query_emb, probs = engine.extract_embedding_and_probs(tensor)

    has_true_style = bool(meta.get("true_style"))
    ref_style = meta.get("true_style") or engine.classes[probs.argmax().item()]

    sims = F.cosine_similarity(query_emb, engine.gallery_embeddings, dim=-1).cpu()

    for i, st in enumerate(engine.gallery["true_styles"]):
        if st.lower() == ref_style.lower():
            sims[i] = -2.0

    top_scores, top_indices = torch.topk(sims, k=min(k, len(sims)))

    cross_neighbors = []
    for score, idx in zip(top_scores, top_indices):
        i = idx.item()
        if score.item() < -1.0:
            continue
        cross_neighbors.append({
            "image_id": engine.gallery["image_ids"][i],
            "painting_name": engine.gallery["titles"][i],
            "author_name": engine.gallery["authors"][i],
            "different_style": engine.gallery["true_styles"][i],
            "cosine_similarity": round(score.item(), 4),
            "similarity_score": round(score.item(), 4),
            "similarity_scale": "[-1.0, 1.0]",
            "path": engine.gallery["paths"][i],
            "stylistic_observation": (
                f"Bien qu'étiqueté comme '{engine.gallery['true_styles'][i]}', ce tableau partage "
                f"une similarité cosinus de {score.item():.2f} (sur une échelle de -1.0 à 1.0) avec '{meta['painting_name']}'."
            )
        })

    return {
        "query_artwork": {
            "image_id": path.stem,
            "painting_name": meta["painting_name"],
            "author_name": meta["author_name"],
            "reference_style": ref_style,
            "reference_style_source": "dataset_metadata" if has_true_style else "predicted_by_model",
            "reference_style_note": (
                "Style réel vérifié issu des métadonnées du dataset."
                if has_true_style
                else f"Attention : oeuvre externe sans étiquette officielle ; le style exclu est le style prédit par le modèle ('{ref_style}') et non le style réel vérifié."
            ),
        },
        "k": k,
        "metric": "cosine_similarity (valeur entre -1.0 et 1.0, non assimilable à une probabilité)",
        "cross_style_neighbors": cross_neighbors,
    }


def get_model_info() -> dict:
    engine = ModelEngine.get_instance()
    return {
        "architecture": "ConvNeXt-Tiny (ArtClassifierTinyBased)",
        "latent_space_dimension": 512,
        "device": str(engine.device),
        "nb_classes": engine.nb_classes,
        "classes": engine.classes,
        "gallery_size": len(engine.gallery["paths"]),
        "checkpoint": str(engine.checkpoint_path.name),
    }
