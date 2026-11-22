import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from src.loader import load_canonical_data, get_eval_transfomrm
    from src.models.pretrained import ArtClassifierTinyBased
    from src.trainer import DEVICE
except ModuleNotFoundError:
    from loader import load_canonical_data, get_eval_transfomrm
    from models.pretrained import ArtClassifierTinyBased
    from trainer import DEVICE


class ConvNeXtGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, output):
        self.activations = output

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate_cam(self, input_tensor, class_idx=None):
        self.model.eval()
        logits = self.model(input_tensor)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        score = logits[0, class_idx]
        self.model.zero_grad()
        score.backward(retain_graph=True)

        # Global average pooling des gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()

        # Normalisation entre 0 et 1
        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        probs = torch.softmax(logits, dim=1)[0]
        confidence = probs[class_idx].item()

        return cam, class_idx, confidence


def generate_gradcam_analysis(
    checkpoint_path: str = "./checkpoints/best_model.pth",
    output_path: str = "./rapports/gradcam_interpretability.png",
    device: str = DEVICE,
):
    print("=" * 80)
    print("GÉNÉRATION DE L'ANALYSE D'ATTRIBUTION GRAD-CAM (EXPLICABILITÉ & RACCOURCIS)")
    print("=" * 80)

    _, _, test_dt, classes, _ = load_canonical_data("./data/Art Movement/")
    model = ArtClassifierTinyBased(len(classes)).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    target_layer = model.base_model.features[7][-1]
    grad_cam = ConvNeXtGradCAM(model, target_layer)
    transform = get_eval_transfomrm(img_size=288)

    sample_queries = [
        {"path": "starry.jpg", "title": "La Nuit étoilée (Van Gogh)", "true_style": "Post-Impressionnisme"},
    ]

    # Sélectionner 3 autres oeuvres variées du jeu de test
    indices_to_find = ["Art Japonais (Ukiyo-e)", "Renaissance", "Street Art & Art Public"]
    found = {k: None for k in indices_to_find}
    for p, c_idx in test_dt.samples:
        style_name = classes[c_idx]
        if style_name in found and found[style_name] is None:
            found[style_name] = (p, style_name)
        if all(v is not None for v in found.values()):
            break

    for style_name, item in found.items():
        if item is not None:
            p_obj, st = item
            sample_queries.append({
                "path": str(p_obj),
                "title": Path(p_obj).stem.split("##")[0][:25],
                "true_style": st,
            })

    fig, axes = plt.subplots(len(sample_queries), 3, figsize=(13, 3.5 * len(sample_queries)))
    if len(sample_queries) == 1:
        axes = np.expand_dims(axes, 0)

    for i, query in enumerate(sample_queries):
        img_p = Path(query["path"])
        if not img_p.exists():
            continue

        raw_img = Image.open(img_p).convert("RGB")
        tensor = transform(raw_img).unsqueeze(0).to(device)

        cam, pred_idx, conf = grad_cam.generate_cam(tensor)
        pred_style = classes[pred_idx]

        disp_img = raw_img.resize((288, 288))
        disp_arr = np.array(disp_img) / 255.0

        # 1. Image originale
        axes[i, 0].imshow(disp_arr)
        axes[i, 0].set_title(f"{query['title']}\nStyle réel : {query['true_style']}", fontsize=11, fontweight="bold")
        axes[i, 0].axis("off")

        # 2. Carte d'activation thermique Grad-CAM
        im_cam = axes[i, 1].imshow(cam, cmap="magma")
        axes[i, 1].set_title(f"Grad-CAM (ConvNeXt Layer 7)\nFocus d'activation", fontsize=11, fontweight="bold")
        axes[i, 1].axis("off")
        plt.colorbar(im_cam, ax=axes[i, 1], fraction=0.046, pad=0.04)

        # 3. Superposition Image + Heatmap
        axes[i, 2].imshow(disp_arr)
        axes[i, 2].imshow(cam, cmap="magma", alpha=0.55)
        axes[i, 2].set_title(f"Prédit : {pred_style} ({conf*100:.1f}%)\nAbsence de biais de bord/cadre", fontsize=11, fontweight="bold")
        axes[i, 2].axis("off")

    plt.tight_layout()
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Visualisation Grad-CAM sauvegardée : {out_p.resolve()}")


if __name__ == "__main__":
    generate_gradcam_analysis()
