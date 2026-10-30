from mcp.server.mcpserver import MCPServer
import mcp_model_interface

mcp = MCPServer("ArtExplorerMCP")

@mcp.tool()
def analyze_artwork(image_id:str)-> dict:
    """
    Retourne le style prédit, les probabilités et les métadonnées 
    (titre, auteur, vrai style) d'une oeuvre.
    """
    return mcp_model_interface.classify(image_id)

@mcp.tool()
def find_similar_artworks(image_id:str, k:int =3)-> dict:
    """
    Trouve les k oeuvres les plus proches visuellement 
    de image_id grâce aux embeddings.
    """
    return mcp_model_interface.find_similar(image_id, k)

@mcp.tool()
def search_artworks(style:str = "", author:str = "", min_confidence:float = 0.0)-> dict:
    """
    Filtre et compte les oeuvres par style prédit, 
    auteur et confiance minimale (entre 0.0 et 1.0).
    """
    return mcp_model_interface.filter_and_count(style, author, min_confidence)

if __name__ == "__main__":
    mcp.run(transport="stdio")