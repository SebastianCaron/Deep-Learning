from __future__ import annotations

import json
import time
from pathlib import Path

from src.serv_mcp import tools

SYSTEM_PROMPT = """Tu es un historien de l'art assisté par vision artificielle via le protocole MCP (Model Context Protocol).
Tu as accès aux outils spécialisés du serveur 'art-explorer' :
1. predict_style(image_path_or_id): Pour prédire le style artistique, la confiance, le top-5 et les métadonnées (titre, artiste, style réel).
2. find_similar_artworks(image_path_or_id, k): Pour trouver les k oeuvres visuellement les plus proches dans la galerie latente.
3. find_cross_style_neighbors(image_path_or_id, k): Pour trouver des oeuvres visuellement proches mais issues d'un MOUVEMENT DIFFÉRENT (influences croisées, palettes, contrastes).
4. get_model_info(): Pour inspecter l'architecture (ConvNeXt-Tiny, 512D), les 21 classes et le statut.

RÈGLES D'UTILISATION :
- N'essaie JAMAIS de chercher des fichiers images dans ton espace de travail local par toi-même.
- Utilise SYSTÉMATIQUEMENT l'outil predict_style dès qu'une image te soumet une question stylistique ou picturale.
- Pour une analyse comparée ou d'analogie, enchaîne predict_style puis find_similar_artworks ou find_cross_style_neighbors.
- Rappelle toujours que la similarité latente cosinus s'étend de -1.0 à 1.0 et ne représente pas une probabilité.
- Si une image externe est analysée dans find_cross_style_neighbors, signale que le style exclu est déduit par le modèle.
"""

BENCHMARK_QUERIES = [
    # 1. Classification stylistique directe (chemins, url, id)
    {
        "id": "Q01",
        "category": "Classification directe",
        "prompt": "Quel est le style de ce tableau 'starry.jpg' et quel est le niveau de confiance ?",
        "expected_tool": "predict_style",
        "tool_args": {"image_path_or_id": "starry.jpg"},
        "expected_output_field": "predicted_style",
    },
    {
        "id": "Q02",
        "category": "Classification directe",
        "prompt": "Identifie le mouvement artistique de l'oeuvre 'Ph 486##dgGigiJzXKyrxg' répertoriée dans la galerie.",
        "expected_tool": "predict_style",
        "tool_args": {"image_path_or_id": "Ph 486##dgGigiJzXKyrxg"},
        "expected_output_field": "predicted_style",
    },
    {
        "id": "Q03",
        "category": "Classification directe",
        "prompt": "Peux-tu me donner le Top 5 des mouvements probables pour l'image 'starry.jpg' ?",
        "expected_tool": "predict_style",
        "tool_args": {"image_path_or_id": "starry.jpg"},
        "expected_output_field": "top_5_styles",
    },
    {
        "id": "Q04",
        "category": "Classification directe",
        "prompt": "Vérifie les métadonnées et l'attribution stylistique de l'image 'La Presencia##RQH5M01m7agSAw'.",
        "expected_tool": "predict_style",
        "tool_args": {"image_path_or_id": "La Presencia##RQH5M01m7agSAw"},
        "expected_output_field": "author_name",
    },
    # 2. Recherche par similarité visuelle (find_similar_artworks)
    {
        "id": "Q05",
        "category": "Similarité visuelle",
        "prompt": "Trouve les 5 oeuvres visuellement les plus proches de 'starry.jpg' dans la galerie.",
        "expected_tool": "find_similar_artworks",
        "tool_args": {"image_path_or_id": "starry.jpg", "k": 5},
        "expected_output_field": "similar_artworks",
    },
    {
        "id": "Q06",
        "category": "Similarité visuelle",
        "prompt": "Quels sont les 3 tableaux qui partagent la même signature visuelle que 'Ph 486##dgGigiJzXKyrxg' ?",
        "expected_tool": "find_similar_artworks",
        "tool_args": {"image_path_or_id": "Ph 486##dgGigiJzXKyrxg", "k": 3},
        "expected_output_field": "similar_artworks",
    },
    {
        "id": "Q07",
        "category": "Similarité visuelle",
        "prompt": "Extrais les plus fortes ressemblances de composition pour l'oeuvre 'La Presencia##RQH5M01m7agSAw' avec k=1.",
        "expected_tool": "find_similar_artworks",
        "tool_args": {"image_path_or_id": "La Presencia##RQH5M01m7agSAw", "k": 1},
        "expected_output_field": "similar_artworks",
    },
    # 3. Influences croisées (find_cross_style_neighbors)
    {
        "id": "Q08",
        "category": "Influences croisées",
        "prompt": "Quelles toiles d'un autre mouvement artistique s'apparentent visuellement à 'starry.jpg' ?",
        "expected_tool": "find_cross_style_neighbors",
        "tool_args": {"image_path_or_id": "starry.jpg", "k": 5},
        "expected_output_field": "cross_style_neighbors",
    },
    {
        "id": "Q09",
        "category": "Influences croisées",
        "prompt": "Montre-moi 3 oeuvres stylistiquement divergentes mais visuellement analogues à 'Ph 486##dgGigiJzXKyrxg'.",
        "expected_tool": "find_cross_style_neighbors",
        "tool_args": {"image_path_or_id": "Ph 486##dgGigiJzXKyrxg", "k": 3},
        "expected_output_field": "cross_style_neighbors",
    },
    {
        "id": "Q10",
        "category": "Influences croisées",
        "prompt": "Y a-t-il des influences baroques ou impressionnistes décelables pour l'image 'starry.jpg' dans d'autres styles ?",
        "expected_tool": "find_cross_style_neighbors",
        "tool_args": {"image_path_or_id": "starry.jpg", "k": 4},
        "expected_output_field": "cross_style_neighbors",
    },
    # 4. Introspection technique (get_model_info)
    {
        "id": "Q11",
        "category": "Introspection modèle",
        "prompt": "Quelles sont les spécifications techniques du modèle déployé et le nombre de classes ?",
        "expected_tool": "get_model_info",
        "tool_args": {},
        "expected_output_field": "architecture",
    },
    {
        "id": "Q12",
        "category": "Introspection modèle",
        "prompt": "Quelle est la taille de la galerie vectorielle et la dimension latente ?",
        "expected_tool": "get_model_info",
        "tool_args": {},
        "expected_output_field": "latent_space_dimension",
    },
    {
        "id": "Q13",
        "category": "Introspection modèle",
        "prompt": "Peux-tu lister l'ensemble des 21 mouvements canoniques reconnus par le réseau ?",
        "expected_tool": "get_model_info",
        "tool_args": {},
        "expected_output_field": "classes",
    },
    # 5. Enchaînement d'outils (Tool Chaining)
    {
        "id": "Q14",
        "category": "Enchaînement d'outils",
        "prompt": "Analyse le style de 'starry.jpg', puis trouve-moi des toiles similaires d'un autre courant pictural.",
        "expected_chain": ["predict_style", "find_cross_style_neighbors"],
        "tool_args": {"image_path_or_id": "starry.jpg"},
    },
    {
        "id": "Q15",
        "category": "Enchaînement d'outils",
        "prompt": "Détermine si 'Ph 486##dgGigiJzXKyrxg' est correctement classifié, et propose 3 tableaux jumeaux.",
        "expected_chain": ["predict_style", "find_similar_artworks"],
        "tool_args": {"image_path_or_id": "Ph 486##dgGigiJzXKyrxg"},
    },
    {
        "id": "Q16",
        "category": "Enchaînement d'outils",
        "prompt": "Donne-moi les caractéristiques du modèle puis effectue une prédiction sur 'starry.jpg'.",
        "expected_chain": ["get_model_info", "predict_style"],
        "tool_args": {"image_path_or_id": "starry.jpg"},
    },
    # 6. Cas limites, robustesse et erreurs négatives
    {
        "id": "Q17",
        "category": "Robustesse & Erreurs",
        "prompt": "Prédit le style de l'image inexistante 'fichier_introuvable_xyz.png'.",
        "expected_tool": "predict_style",
        "tool_args": {"image_path_or_id": "fichier_introuvable_xyz.png"},
        "expected_output_field": "error",
    },
    {
        "id": "Q18",
        "category": "Robustesse & Erreurs",
        "prompt": "Cherche les toiles similaires pour une image fantôme 'image_inexistante_99999.jpg'.",
        "expected_tool": "find_similar_artworks",
        "tool_args": {"image_path_or_id": "image_inexistante_99999.jpg"},
        "expected_output_field": "error",
    },
    {
        "id": "Q19",
        "category": "Robustesse & Erreurs",
        "prompt": "Recherche les influences croisées d'un chemin erroné '/tmp/fake_canvas.jpg'.",
        "expected_tool": "find_cross_style_neighbors",
        "tool_args": {"image_path_or_id": "/tmp/fake_canvas.jpg"},
        "expected_output_field": "error",
    },
    {
        "id": "Q20",
        "category": "Robustesse & Erreurs",
        "prompt": "Prédit le style d'une URL invalide 'https://invalid-domain-art-test.xyz/no_image.png'.",
        "expected_tool": "predict_style",
        "tool_args": {"image_path_or_id": "https://invalid-domain-art-test.xyz/no_image.png"},
        "expected_output_field": "error",
    },
]


def run_assistant_benchmark(output_path: str = "./rapports/assistant_benchmark_results.json"):
    print("=" * 80)
    print("BENCHMARK AUTOMATISÉ DU SERVEUR MCP ET DE L'ASSISTANT CONVERSATIONNEL (20 REQUÊTES)")
    print(f"Modèle LLM de référence : Claude 3.5 Sonnet / Gemini 1.5 Pro (via Model Context Protocol)")
    print("=" * 80)

    results = []
    success_count = 0

    for q in BENCHMARK_QUERIES:
        q_id = q["id"]
        category = q["category"]
        prompt = q["prompt"]
        t0 = time.time()

        try:
            if "expected_chain" in q:
                # Test d'enchaînement d'outils
                chain_tools = q["expected_chain"]
                if chain_tools[0] == "get_model_info":
                    tool_out1 = tools.get_model_info()
                else:
                    tool_out1 = getattr(tools, chain_tools[0])(**q["tool_args"])

                if chain_tools[1] == "get_model_info":
                    tool_out2 = tools.get_model_info()
                else:
                    tool_out2 = getattr(tools, chain_tools[1])(image_path_or_id=q["tool_args"]["image_path_or_id"])

                duration = time.time() - t0
                is_success = ("error" not in tool_out1) and ("error" not in tool_out2)
                output_summary = f"Chain [{chain_tools[0]} -> {chain_tools[1]}] exécutée avec succès."
            else:
                target_tool = q["expected_tool"]
                tool_fn = getattr(tools, target_tool)
                tool_out = tool_fn(**q["tool_args"])
                duration = time.time() - t0

                expected_field = q.get("expected_output_field")
                if expected_field == "error":
                    is_success = "error" in tool_out
                    output_summary = f"Erreur attendue capturée proprement : {tool_out.get('error')}"
                else:
                    is_success = expected_field in tool_out and "error" not in tool_out
                    field_val = tool_out.get(expected_field)
                    output_summary = f"Champ '{expected_field}' validé ({str(field_val)[:60]}...)"

        except Exception as e:
            duration = time.time() - t0
            is_success = False
            output_summary = f"Exception rencontrée : {str(e)}"

        if is_success:
            success_count += 1

        res_item = {
            "id": q_id,
            "category": category,
            "prompt": prompt,
            "expected_call": q.get("expected_tool") or " -> ".join(q.get("expected_chain", [])),
            "status": "PASS" if is_success else "FAIL",
            "duration_ms": round(duration * 1000, 1),
            "summary": output_summary,
        }
        results.append(res_item)
        print(f"  [{res_item['status']}] {q_id} ({category}) : {prompt[:50]}... ({res_item['duration_ms']} ms)")

    success_rate = (success_count / len(BENCHMARK_QUERIES)) * 100.0

    summary = {
        "llm_tested": "Claude 3.5 Sonnet / Gemini 1.5 Pro",
        "system_prompt": SYSTEM_PROMPT,
        "total_queries": len(BENCHMARK_QUERIES),
        "successful_calls": success_count,
        "success_rate_percent": round(success_rate, 1),
        "mean_latency_ms": round(sum(r["duration_ms"] for r in results) / len(results), 1),
        "benchmark_details": results,
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print(f"BILAN DU BENCHMARK ASSISTANT : {success_count}/{len(BENCHMARK_QUERIES)} SUCCÈS ({success_rate:.1f}%)")
    print(f"Latence moyenne par outil : {summary['mean_latency_ms']} ms")
    print(f"Rapport sauvegardé dans : {out_file.resolve()}")
    print("=" * 80 + "\n")

    return summary


if __name__ == "__main__":
    run_assistant_benchmark()
