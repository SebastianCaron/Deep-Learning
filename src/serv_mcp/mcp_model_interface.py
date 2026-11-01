import sys
from pathlib import Path

import torch
from PIL import Image

from src.models.pretrained import ArtClassifierTinyBased
from src.loader import get_eval_transfomrm
import src.serv_mcp.utils as utils


class ModelEngine:
    _instance: "ModelEngine" | None = None

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/best_model.pth",
        gallery_index_path: str = "checkpoints/gallery_index.pt",
        label_csv_path: str = "data/label_list.csv",
        img_size: int = 288,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = img_size
        self.transform = get_eval_transfomrm(img_size=img_size)

        # Gallery
        self.gallery_path = Path(gallery_index_path)
        if not self.gallery_path.exists():
            raise FileNotFoundError(
                f"Gallery index introuvable: {self.gallery_path}. "
                "Exécutez 'python3 src/serv_mcp/build_index.py' pour le générer."
            )
        print(f"Chargement de la galerie depuis {self.gallery_path}...", file=sys.stderr)
        self.gallery = torch.load(self.gallery_path, map_location="cpu")
        self.classes: list[str] = self.gallery["classes"]
        self.nb_classes = len(self.classes)

        self.gallery_embeddings = self.gallery["embeddings"].to(self.device)

        self.id_to_gallery_idx: dict[str, int] = {}
        for idx, (img_id, path_str) in enumerate(zip(self.gallery["image_ids"], self.gallery["paths"])):
            self.id_to_gallery_idx[img_id] = idx
            self.id_to_gallery_idx[Path(path_str).name] = idx

        # le modele
        self.checkpoint_path = Path(checkpoint_path)
        print(f"Chargement du modèle ConvNeXt depuis {self.checkpoint_path} sur {self.device}...", file=sys.stderr)
        self.model = ArtClassifierTinyBased(nb_classes=self.nb_classes).to(self.device)
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        # les metadata
        self.meta_map = utils.load_metadata_map(Path(label_csv_path))

    @classmethod
    def get_instance(cls) -> "ModelEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def resolve_image(self, identifier: str) -> tuple[Path | None, int | None]:
        return utils.resolve_image(
            identifier,
            id_to_gallery_idx=self.id_to_gallery_idx,
            gallery_paths=self.gallery["paths"]
        )

    def get_metadata(self, path: Path, gallery_idx: int | None = None) -> dict:
        return utils.get_metadata_for_path(
            path,
            gallery_idx=gallery_idx,
            gallery=self.gallery,
            meta_map=self.meta_map
        )

    def load_tensor(self, path: Path) -> torch.Tensor:
        with Image.open(path) as img:
            rgb_img = img.convert("RGB")
            tensor = self.transform(rgb_img)
        return tensor.unsqueeze(0).to(self.device)

    def extract_embedding_and_probs(self, tensor: torch.Tensor):
        with torch.no_grad():
            feats = self.model.base_model(tensor)
            latent = self.model.latent_space(feats)
            logits = self.model.classifier(latent)
            probs = torch.softmax(logits, dim=-1)
        return latent, probs.squeeze(0)
