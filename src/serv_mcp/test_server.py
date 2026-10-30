#Genere Gemini pour test, à suppr ?
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test():
    server_params = StdioServerParameters(command="python", args=["mcp_server.py"])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            tools = await session.list_tools()
            print("Outils détectés :", [t.name for t in tools.tools])
            
            res = await session.call_tool("search_artworks", arguments={"style": "Impressionism"})
            print("Résultat de search_artworks :", res.content[0].text)

            res_analyze = await session.call_tool("analyze_artwork", arguments={"image_id": "art_001"})
            print("Résultat de analyze_artwork :", res_analyze.content[0].text)

            res_sim = await session.call_tool("find_similar_artworks", arguments={"image_id": "art_001", "k": 2})
            print("Résultat de find_similar_artworks :", res_sim.content[0].text)

asyncio.run(test())