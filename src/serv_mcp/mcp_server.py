from mcp.server.mcpserver import MCPServer
from src.serv_mcp import tools

mcp = MCPServer("ArtExplorerMCP")


@mcp.tool()
def predict_style(image_path_or_id: str) -> dict:
    """
    Prédit le style artistique d'une image (peinture), sa confiance,
    le top 5 des styles les plus probables, et extrait les métadonnées de l'oeuvre
    (titre, artiste, style réel si répertorié dans la base).
    
    Args:
        image_path_or_id: Chemin vers l'image (/chemin/image.jpg), URL web ou identifiant d'oeuvre.
    """
    return tools.predict_style(image_path_or_id)


@mcp.tool()
def find_similar_artworks(image_path_or_id: str, k: int = 5) -> dict:
    """
    Trouve les k oeuvres les plus ressemblantes visuellement dans la galerie d'art,
    en s'appuyant sur la similarité cosinus de leurs représentations dans l'espace
    latent à 512 dimensions du modèle.
    
    Args:
        image_path_or_id: Chemin vers l'image, URL web ou identifiant de l'oeuvre requête.
        k: Nombre d'oeuvres similaires à retourner (par défaut: 5).
    """
    return tools.find_similar_artworks(image_path_or_id, k=k)


@mcp.tool()
def find_cross_style_neighbors(image_path_or_id: str, k: int = 5) -> dict:
    """
    Trouve les k oeuvres les plus proches visuellement mais appartenant à un STYLE DIFFÉRENT.
    Permet d'explorer les influences croisées, similitudes de palette, de lumière
    ou de composition entre mouvements artistiques distincts.
    
    Args:
        image_path_or_id: Chemin vers l'image, URL web ou identifiant de l'oeuvre requête.
        k: Nombre d'oeuvres de styles différents à retourner (par défaut: 5).
    """
    return tools.find_cross_style_neighbors(image_path_or_id, k=k)

@mcp.tool()
def get_model_info() -> dict:
    """
    Retourne les informations techniques sur le modèle de Deep Learning actuellement
    chargé, son architecture, ses dimensions d'espace latent, les 21 classes de style
    et le nombre d'oeuvres répertoriées dans la galerie.
    """
    return tools.get_model_info()


if __name__ == "__main__":
    mcp.run(transport="stdio")
