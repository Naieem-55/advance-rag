"""
Query Visualization for HippoRAG
Shows which nodes are given high relevance when processing a query
"""

import os
import json
import numpy as np
from pyvis.network import Network
import webbrowser
from typing import Dict, List, Tuple, Optional

from dotenv import load_dotenv
load_dotenv()


def get_query_relevance_scores(hipporag, query: str) -> Dict:
    """
    Run retrieval and capture all intermediate scores for visualization.

    Returns dict with:
    - query_entities: entities extracted from query
    - phrase_weights: initial entity weights from fact matching
    - ppr_scores: final PPR scores for all nodes
    - top_facts: matched facts
    - top_passages: retrieved passages with scores
    """
    if not hipporag.ready_to_retrieve:
        hipporag.prepare_retrieval_objects()

    # Get query embedding
    hipporag.get_query_embeddings([query])

    # Get fact scores
    query_fact_scores = hipporag.get_fact_scores(query)

    # Rerank facts
    top_k_fact_indices, top_k_facts, rerank_log = hipporag.rerank_facts(query, query_fact_scores)

    if len(top_k_facts) == 0:
        # Fallback to DPR results when no facts found
        dpr_sorted_doc_ids, dpr_sorted_doc_scores = hipporag.dense_passage_retrieval(query)

        top_passages = []
        for i, doc_id in enumerate(dpr_sorted_doc_ids[:10]):
            passage_key = hipporag.passage_node_keys[doc_id]
            content = hipporag.chunk_embedding_store.get_row(passage_key)["content"]
            top_passages.append({
                "rank": i + 1,
                "score": float(dpr_sorted_doc_scores[i]),
                "content": content[:200] + "..." if len(content) > 200 else content
            })

        return {
            "query": query,
            "warning": "No facts matched - showing DPR results only",
            "ppr_scores": {},
            "query_entities": [],
            "top_facts": [],
            "top_passages": top_passages,
            "total_nodes": len(hipporag.graph.vs),
            "total_edges": len(hipporag.graph.es),
            "use_dpr_only": True
        }

    # Calculate phrase weights (entity relevance)
    from hipporag.utils.misc_utils import compute_mdhash_id

    phrase_weights = np.zeros(len(hipporag.graph.vs['name']))
    passage_weights = np.zeros(len(hipporag.graph.vs['name']))
    number_of_occurs = np.zeros(len(hipporag.graph.vs['name']))

    query_entities = set()

    for rank, f in enumerate(top_k_facts):
        subject_phrase = f[0].lower()
        object_phrase = f[2].lower()
        fact_score = query_fact_scores[top_k_fact_indices[rank]] if query_fact_scores.ndim > 0 else query_fact_scores

        for phrase in [subject_phrase, object_phrase]:
            phrase_key = compute_mdhash_id(content=phrase, prefix="entity-")
            phrase_id = hipporag.node_name_to_vertex_idx.get(phrase_key, None)

            if phrase_id is not None:
                query_entities.add(phrase)
                weighted_fact_score = fact_score
                if len(hipporag.ent_node_to_chunk_ids.get(phrase_key, set())) > 0:
                    weighted_fact_score /= len(hipporag.ent_node_to_chunk_ids[phrase_key])
                phrase_weights[phrase_id] += weighted_fact_score
                number_of_occurs[phrase_id] += 1

    # Avoid division by zero
    number_of_occurs[number_of_occurs == 0] = 1
    phrase_weights /= number_of_occurs

    # Get passage scores from DPR
    dpr_sorted_doc_ids, dpr_sorted_doc_scores = hipporag.dense_passage_retrieval(query)

    # Normalize DPR scores
    if len(dpr_sorted_doc_scores) > 0:
        min_score = np.min(dpr_sorted_doc_scores)
        max_score = np.max(dpr_sorted_doc_scores)
        if max_score > min_score:
            normalized_dpr_scores = (dpr_sorted_doc_scores - min_score) / (max_score - min_score)
        else:
            normalized_dpr_scores = np.ones_like(dpr_sorted_doc_scores)
    else:
        normalized_dpr_scores = dpr_sorted_doc_scores

    passage_node_weight = hipporag.global_config.passage_node_weight

    for i, dpr_sorted_doc_id in enumerate(dpr_sorted_doc_ids.tolist()):
        passage_node_key = hipporag.passage_node_keys[dpr_sorted_doc_id]
        passage_dpr_score = normalized_dpr_scores[i]
        passage_node_id = hipporag.node_name_to_vertex_idx[passage_node_key]
        passage_weights[passage_node_id] = passage_dpr_score * passage_node_weight

    # Combined weights for PPR
    node_weights = phrase_weights + passage_weights

    # Run PPR
    reset_prob = np.where(np.isnan(node_weights) | (node_weights < 0), 0, node_weights)
    damping = hipporag.global_config.damping

    pagerank_scores = hipporag.graph.personalized_pagerank(
        vertices=range(len(hipporag.node_name_to_vertex_idx)),
        damping=damping,
        directed=False,
        weights='weight',
        reset=reset_prob,
        implementation='prpack'
    )

    # Build results
    ppr_scores = {}
    initial_weights = {}

    for v in hipporag.graph.vs:
        node_name = v['name']
        hash_id = v['hash_id'] if 'hash_id' in hipporag.graph.vs.attributes() else ''
        idx = v.index

        ppr_scores[node_name] = {
            'hash_id': hash_id,
            'ppr_score': float(pagerank_scores[idx]),
            'initial_weight': float(node_weights[idx]),
            'phrase_weight': float(phrase_weights[idx]),
            'passage_weight': float(passage_weights[idx])
        }

    # Get top passages
    doc_scores = np.array([pagerank_scores[idx] for idx in hipporag.passage_node_idxs])
    sorted_doc_ids = np.argsort(doc_scores)[::-1]

    top_passages = []
    for i, doc_id in enumerate(sorted_doc_ids[:10]):
        passage_key = hipporag.passage_node_keys[doc_id]
        content = hipporag.chunk_embedding_store.get_row(passage_key)["content"]
        top_passages.append({
            "rank": i + 1,
            "score": float(doc_scores[doc_id]),
            "content": content[:200] + "..." if len(content) > 200 else content
        })

    return {
        "query": query,
        "query_entities": list(query_entities),
        "top_facts": [{"subject": f[0], "predicate": f[1], "object": f[2]} for f in top_k_facts[:10]],
        "top_passages": top_passages,
        "ppr_scores": ppr_scores,
        "total_nodes": len(hipporag.graph.vs),
        "total_edges": len(hipporag.graph.es)
    }


def create_query_visualization(hipporag, query: str, output_path: str = "outputs/query_graph.html") -> str:
    """
    Create an interactive visualization of the knowledge graph with query relevance scores.
    Nodes are colored and sized based on their PPR scores.
    """

    # Get relevance scores
    scores_data = get_query_relevance_scores(hipporag, query)

    if "error" in scores_data:
        print(f"Error: {scores_data['error']}")
        return None

    ppr_scores = scores_data.get("ppr_scores", {})
    query_entities = set(scores_data.get("query_entities", []))
    use_dpr_only = scores_data.get("use_dpr_only", False)

    if use_dpr_only:
        print(f"Warning: {scores_data.get('warning', 'No facts matched')}")

    # Create network
    net = Network(height="900px", width="100%", bgcolor="#1a1a2e", font_color="white")
    net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=200)

    # Get max PPR score for normalization
    all_ppr = [v['ppr_score'] for v in ppr_scores.values()]
    max_ppr = max(all_ppr) if all_ppr else 1
    min_ppr = min(all_ppr) if all_ppr else 0

    # Color scale: blue (low) -> yellow (medium) -> red (high)
    def get_color(score, is_entity=True, is_query_entity=False, is_passage=False):
        if is_query_entity:
            return "#00ff00"  # Green for query entities

        if max_ppr > min_ppr:
            normalized = (score - min_ppr) / (max_ppr - min_ppr)
        else:
            normalized = 0.5

        if is_passage:
            # Blue scale for passages
            intensity = int(100 + normalized * 155)
            return f"#{intensity:02x}{intensity:02x}ff"
        else:
            # Red-yellow scale for entities
            if normalized < 0.5:
                r = int(255 * (normalized * 2))
                g = 255
                b = 0
            else:
                r = 255
                g = int(255 * (1 - (normalized - 0.5) * 2))
                b = 0
            return f"#{r:02x}{g:02x}{b:02x}"

    # Add nodes
    for v in hipporag.graph.vs:
        node_name = v['name']
        hash_id = v['hash_id'] if 'hash_id' in hipporag.graph.vs.attributes() else ''

        score_data = ppr_scores.get(node_name, {})
        ppr_score = score_data.get('ppr_score', 0)
        initial_weight = score_data.get('initial_weight', 0)

        is_entity = hash_id.startswith('entity')
        is_passage = hash_id.startswith('chunk')
        is_query_entity = node_name.lower() in query_entities

        # Size based on PPR score
        if max_ppr > min_ppr:
            size_factor = (ppr_score - min_ppr) / (max_ppr - min_ppr)
        else:
            size_factor = 0.5

        if is_passage:
            base_size = 20
            shape = "box"
            # Show passage content preview
            try:
                content = hipporag.chunk_embedding_store.get_row(hash_id)["content"]
                label = content[:50] + "..." if len(content) > 50 else content
            except:
                label = "Passage"
        else:
            base_size = 15
            shape = "dot"
            label = node_name

        if is_query_entity:
            base_size = 30  # Larger for query entities

        size = base_size + size_factor * 30
        color = get_color(ppr_score, is_entity, is_query_entity, is_passage)

        # Tooltip with score info
        title = f"""
        <b>{node_name}</b><br>
        Type: {'Query Entity' if is_query_entity else 'Entity' if is_entity else 'Passage'}<br>
        PPR Score: {ppr_score:.6f}<br>
        Initial Weight: {initial_weight:.6f}<br>
        Hash: {hash_id}
        """

        net.add_node(
            node_name,
            label=label[:30] if len(label) > 30 else label,
            title=title,
            size=size,
            color=color,
            shape=shape,
            borderWidth=3 if is_query_entity else 1,
            borderWidthSelected=5
        )

    # Add edges
    for e in hipporag.graph.es:
        source = hipporag.graph.vs[e.source]['name']
        target = hipporag.graph.vs[e.target]['name']
        weight = e['weight'] if 'weight' in hipporag.graph.es.attributes() else 1

        net.add_edge(source, target, value=weight, title=f"Weight: {weight:.4f}")

    # Add legend and query info
    html_content = f"""
    <div style="position: fixed; top: 10px; left: 10px; background: rgba(0,0,0,0.8); padding: 15px; border-radius: 10px; color: white; max-width: 400px; z-index: 1000;">
        <h3 style="margin-top: 0;">Query: "{query}"</h3>
        <p><b>Query Entities:</b> {', '.join(query_entities) if query_entities else 'None found'}</p>
        <hr>
        <p><b>Legend:</b></p>
        <p><span style="color: #00ff00;">&#9679;</span> Query Entity (starting point)</p>
        <p><span style="color: #ffff00;">&#9679;</span> High relevance entity</p>
        <p><span style="color: #ff0000;">&#9679;</span> Very high relevance</p>
        <p><span style="color: #6464ff;">&#9632;</span> Passage node</p>
        <hr>
        <p><b>Top Retrieved Passages:</b></p>
    """

    for p in scores_data["top_passages"][:5]:
        html_content += f"<p>{p['rank']}. (score: {p['score']:.4f}) {p['content'][:100]}...</p>"

    html_content += """
        <hr>
        <p><b>Top Facts:</b></p>
    """

    for f in scores_data["top_facts"][:5]:
        html_content += f"<p>({f['subject']}, {f['predicate']}, {f['object']})</p>"

    html_content += "</div>"

    # Save and customize HTML
    net.save_graph(output_path)

    # Add legend to the HTML
    with open(output_path, 'r', encoding='utf-8') as f:
        html = f.read()

    html = html.replace('<body>', f'<body>{html_content}')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"Query visualization saved to: {output_path}")
    return output_path


def visualize_query(query: str = None):
    """Main function to visualize query relevance on knowledge graph."""
    from hipporag import HippoRAG

    print("Loading HippoRAG...")
    hipporag = HippoRAG(
        save_dir='outputs',
        llm_model_name='gemini/gemini-2.0-flash',
        embedding_model_name='gemini/gemini-embedding-001'
    )

    hipporag.prepare_retrieval_objects()
    print(f"Loaded graph with {len(hipporag.graph.vs)} nodes and {len(hipporag.graph.es)} edges")

    if query is None:
        query = input("Enter your query: ")

    output_path = create_query_visualization(hipporag, query)

    if output_path:
        webbrowser.open(f'file://{os.path.abspath(output_path)}')


if __name__ == "__main__":
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else None
    visualize_query(query)
