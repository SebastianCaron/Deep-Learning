import sys
import hashlib
import urllib.request
from pathlib import Path

from src.loader import TAXONOMY_MAPPING


def download_url(url: str, cache_dir: Path | None = None) -> Path | None:
    if cache_dir is None:
        cache_dir = Path("data/cache_mcp")
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()[:12]
        ext = Path(url.split("?")[0]).suffix
        if ext.lower() not in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
            ext = ".jpg"

        target_file = cache_dir / f"url_{url_hash}{ext}"
        if not target_file.exists():
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (ArtExplorerMCP/1.0)"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp, open(target_file, "wb") as f:
                f.write(resp.read())
        return target_file
    except Exception as e:
        print(f"Erreur lors du téléchargement de l'URL {url}: {e}", file=sys.stderr)
        return None


def resolve_image(
    identifier: str,
    id_to_gallery_idx: dict[str, int] | None = None,
    gallery_paths: list[str] | None = None
) -> tuple[Path | None, int | None]:
    identifier = str(identifier).strip()

    if identifier.startswith("http://") or identifier.startswith("https://"):
        dl_path = download_url(identifier)
        return dl_path, None

    p = Path(identifier).expanduser()
    if p.exists() and p.is_file():
        gal_idx = None
        if id_to_gallery_idx:
            gal_idx = id_to_gallery_idx.get(p.name) or id_to_gallery_idx.get(p.stem)
        return p, gal_idx

    if id_to_gallery_idx and identifier in id_to_gallery_idx:
        idx = id_to_gallery_idx[identifier]
        if gallery_paths and idx < len(gallery_paths):
            gal_path = Path(gallery_paths[idx])
            if gal_path.exists():
                return gal_path, idx

    clean_id = Path(identifier).stem
    matches = list(Path("data/Art Movement").glob(f"*/*{clean_id}*"))
    if not matches:
        matches = list(Path("data").glob(f"*/*{clean_id}*"))
    if matches:
        found = matches[0]
        gal_idx = None
        if id_to_gallery_idx:
            gal_idx = id_to_gallery_idx.get(found.name) or id_to_gallery_idx.get(found.stem)
        return found, gal_idx

    return None, None


def load_metadata_map(csv_path: Path) -> dict[str, tuple[str, str]]:
    meta_map = {}
    if not csv_path.exists():
        return meta_map
    try:
        with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) >= 2:
                    author = parts[0].strip()
                    key = parts[1].strip() + ".jpg"
                    title = parts[1].split("##")[0].strip()
                    meta_map[key] = (author, title)
    except Exception as e:
        print(f"Warning: Impossible de lire {csv_path}: {e}", file=sys.stderr)
    return meta_map


def get_metadata_for_path(
    path: Path,
    gallery_idx: int | None = None,
    gallery: dict | None = None,
    meta_map: dict | None = None
) -> dict:
    if gallery is not None and gallery_idx is not None and gallery_idx < len(gallery["titles"]):
        return {
            "painting_name": gallery["titles"][gallery_idx],
            "author_name": gallery["authors"][gallery_idx],
            "true_style": gallery["true_styles"][gallery_idx],
        }

    if meta_map and path.name in meta_map:
        author, title = meta_map[path.name]
    else:
        title = path.stem.split("##")[0]
        author = "Image externe"

    if "Art Movement" in path.parts:
        parent_dir = path.parent.name
        true_style = TAXONOMY_MAPPING.get(parent_dir, parent_dir)
    else:
        true_style = None

    return {
        "painting_name": title,
        "author_name": author,
        "true_style": true_style,
    }
