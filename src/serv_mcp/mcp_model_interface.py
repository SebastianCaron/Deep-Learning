import torch
import torch.nn.functional as F

#dans catalog : torch.load("path"), là il s'agit d'un cqtalogue généré pour test rapidos
CATALOG = {
    "art_001": {
        "painting_name": "Impression, soleil levant",
        "author_name": "Claude Monet",
        "true_style": "Impressionism",
        "path": "images/monet_soleil.jpg",
        "pred_probs": {"Impressionism": 0.88, "Baroque": 0.05, "Renaissance": 0.03, "Surrealism": 0.02, "Cubism": 0.02},
        "embedding": torch.tensor([0.9, 0.1, -0.2, 0.4])
    },
    "art_002": {
        "painting_name": "La Nuit étoilée",
        "author_name": "Vincent van Gogh",
        "true_style": "Impressionism",
        "path": "images/vangogh_nuit.jpg",
        "pred_probs": {"Impressionism": 0.76, "Surrealism": 0.15, "Baroque": 0.05, "Renaissance": 0.02, "Cubism": 0.02},
        "embedding": torch.tensor([0.85, 0.15, -0.1, 0.35])
    },
    "art_003": {
        "painting_name": "La Joconde",
        "author_name": "Leonardo da Vinci",
        "true_style": "Renaissance",
        "path": "images/davinci_joconde.jpg",
        "pred_probs": {"Renaissance": 0.94, "Baroque": 0.04, "Impressionism": 0.01, "Surrealism": 0.01, "Cubism": 0.00},
        "embedding": torch.tensor([-0.5, 0.8, 0.6, -0.2])
    }}

def resolve_id(identifier:str) ->str | None:
    """
    -> Permet d'utiliser le path
    Retourne None si l'id donné n'est ni un id existant, ni un path existant
    """
    if identifier in CATALOG:
        return identifier
    for img_id, data in CATALOG.items():
        if data["path"] == identifier:
            return img_id
    return None

def classify(identifier:str)-> dict:
    """
    -> Analyse de l'oeuvre (prédictions, probas...)
    """

    img_id = resolve_id(identifier)
    if not img_id:
        return {"error": f"Image '{identifier}' introuvable."}
    
    item = CATALOG[img_id]
    top_class = max(item["pred_probs"], key=item["pred_probs"].get)
    return {
        "image_id": img_id,
        "painting_name": item["painting_name"],
        "author_name": item["author_name"],
        "true_style": item["true_style"],
        "predicted_style": top_class,
        "confidence": item["pred_probs"][top_class],
        "all_probabilities": item["pred_probs"]
    }

#Utilitaire pour find_similar juste en dessous
def extract_score(x: dict)-> float:
    return x["similarity_score"]

def find_similar(identifier:str, k=3)-> dict:
    """
    -> Donner les k oeuvres les plus proches
    """
    img_id = resolve_id(identifier)
    if not img_id:
        return {"error": f"Image '{identifier}' introuvable."}

    target_emb = CATALOG[img_id]["embedding"].unsqueeze(0)
    similarities = []

    for other_id, data in CATALOG.items():
        if other_id == img_id:
            continue
        other_emb = data["embedding"].unsqueeze(0)
        sim = F.cosine_similarity(target_emb, other_emb).item()
        top_pred = max(data["pred_probs"], key=data["pred_probs"].get) #Pour check si ils ont (ou pas) tous la même classe, cf sujet
        similarities.append({
            "image_id": other_id,
            "painting_name": data["painting_name"],
            "author_name": data["author_name"],
            "predicted_style": top_pred,
            "similarity_score": round(sim, 4)
        })

    similarities.sort(key=extract_score, reverse=True)
    return {"query_image_id": img_id, "neighbors": similarities[:k]}

def filter_and_count(style:str = "", author:str = "", min_confidence:float = 0.0)-> dict:
    """
    -> Filtre et compte le nombre d'oeuvres passant le.s filtre.s
    """
    matches = []
    for img_id, data in CATALOG.items():
        top_class = max(data["pred_probs"], key=data["pred_probs"].get)
        conf = data["pred_probs"][top_class]

        if style and top_class.lower() != style.lower():
            continue
        if author and author.lower() not in data["author_name"].lower():
            continue
        if conf < min_confidence:
            continue

        matches.append({
            "image_id": img_id,
            "painting_name": data["painting_name"],
            "author_name": data["author_name"],
            "predicted_style": top_class,
            "confidence": conf
        })

    return {"total_matching_images": len(matches), "results": matches[:10]} #Voir si 10 bonne limite ou + ou - ?