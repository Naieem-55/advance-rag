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
    Run retrieval using the SAME pipeline as /ask endpoint and capture all scores.

    This uses the full hybrid retrieval pipeline:
    1. Fact matching + reranking
    2. DPR (dense passage retrieval)
    3. PPR (if facts found) + Hybrid scoring
    4. Cross-encoder reranking (SAME as /ask)

    Returns dict with:
    - query_entities: entities extracted from matched facts
    - top_facts: matched facts from knowledge graph
    - top_passages: FINAL retrieved passages (after cross-encoder reranking)
    - ppr_scores: PPR scores for all nodes (for visualization)
    """
    if not hipporag.ready_to_retrieve:
        hipporag.prepare_retrieval_objects()

    # Get query embedding
    hipporag.get_query_embeddings([query])

    # Get fact scores
    query_fact_scores = hipporag.get_fact_scores(query)

    # Rerank facts
    top_k_fact_indices, top_k_facts, rerank_log = hipporag.rerank_facts(query, query_fact_scores)

    # Always get DPR results for hybrid scoring
    dpr_sorted_doc_ids, dpr_sorted_doc_scores = hipporag.dense_passage_retrieval(query)

    # Calculate fact confidence
    if len(top_k_fact_indices) > 0 and len(query_fact_scores) > 0:
        fact_confidence = float(np.max([query_fact_scores[i] for i in top_k_fact_indices]))
    else:
        fact_confidence = 0.0

    # Hybrid retrieval (same as /ask endpoint)
    if len(top_k_facts) == 0:
        # No facts - use DPR only
        sorted_doc_ids, sorted_doc_scores = dpr_sorted_doc_ids, dpr_sorted_doc_scores
        use_dpr_only = True
    else:
        # PPR + Hybrid scoring
        ppr_doc_ids, ppr_doc_scores = hipporag.graph_search_with_fact_entities(
            query=query,
            link_top_k=hipporag.global_config.linking_top_k,
            query_fact_scores=query_fact_scores,
            top_k_facts=top_k_facts,
            top_k_fact_indices=top_k_fact_indices,
            passage_node_weight=hipporag.global_config.passage_node_weight
        )

        # Apply adaptive hybrid scoring
        sorted_doc_ids, sorted_doc_scores = hipporag.compute_adaptive_hybrid_scores(
            ppr_doc_ids=ppr_doc_ids,
            ppr_doc_scores=ppr_doc_scores,
            dpr_doc_ids=dpr_sorted_doc_ids,
            dpr_doc_scores=dpr_sorted_doc_scores,
            fact_confidence=fact_confidence
        )
        use_dpr_only = False

    # Get candidate documents for reranking
    num_to_retrieve = hipporag.global_config.retrieval_top_k
    candidate_count = min(num_to_retrieve * 2, len(sorted_doc_ids))
    candidate_docs = [hipporag.chunk_embedding_store.get_row(hipporag.passage_node_keys[idx])["content"]
                     for idx in sorted_doc_ids[:candidate_count]]
    candidate_scores = sorted_doc_scores[:candidate_count]

    # Apply cross-encoder reranking (SAME as /ask endpoint)
    if hipporag.use_reranker and len(candidate_docs) > 1:
        try:
            reranked_indices, reranked_scores = hipporag.reranker.rerank(
                query=query,
                documents=candidate_docs,
                top_k=num_to_retrieve
            )
            final_docs = [candidate_docs[i] for i in reranked_indices[:num_to_retrieve]]
            final_scores = reranked_scores[:num_to_retrieve]
        except Exception as e:
            print(f"Reranking failed, using original order: {e}")
            final_docs = candidate_docs[:num_to_retrieve]
            final_scores = list(candidate_scores[:num_to_retrieve])
    else:
        final_docs = candidate_docs[:num_to_retrieve]
        final_scores = list(candidate_scores[:num_to_retrieve])

    # Build top_passages from FINAL reranked results
    top_passages = []
    for i, (doc, score) in enumerate(zip(final_docs[:10], final_scores[:10])):
        top_passages.append({
            "rank": i + 1,
            "score": float(score),
            "content": doc[:200] + "..." if len(doc) > 200 else doc
        })

    # Calculate phrase weights for visualization (entity relevance in KG)
    from src.hipporag.utils.misc_utils import compute_mdhash_id

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

    # Normalize DPR scores for passage weights
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

    # Combined weights for PPR visualization
    node_weights = phrase_weights + passage_weights

    # Run PPR for visualization
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

    # Build PPR scores for visualization
    ppr_scores = {}
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

    result = {
        "query": query,
        "query_entities": list(query_entities),
        "top_facts": [{"subject": f[0], "predicate": f[1], "object": f[2]} for f in top_k_facts[:10]],
        "top_passages": top_passages,  # Now shows FINAL reranked passages
        "ppr_scores": ppr_scores,
        "total_nodes": len(hipporag.graph.vs),
        "total_edges": len(hipporag.graph.es),
        "retrieval_method": "dpr_only" if use_dpr_only else "hybrid_ppr_dpr",
        "fact_confidence": fact_confidence
    }

    if use_dpr_only:
        result["warning"] = "No facts matched in KG - using DPR + cross-encoder reranking"

    return result


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

    # Create network with professional styling
    net = Network(height="100vh", width="100vw", bgcolor="#0f1419", font_color="#e7e9ea")

    # Professional physics settings - STABLE after initial layout
    net.set_options('''
    {
        "nodes": {
            "font": {"size": 12, "face": "Inter, -apple-system, BlinkMacSystemFont, sans-serif", "color": "#e7e9ea"},
            "borderWidth": 2,
            "borderWidthSelected": 4
        },
        "edges": {
            "color": {"color": "#38444d", "highlight": "#1d9bf0"},
            "smooth": {"type": "continuous", "roundness": 0.5}
        },
        "physics": {
            "enabled": true,
            "forceAtlas2Based": {
                "gravitationalConstant": -100,
                "centralGravity": 0.02,
                "springLength": 150,
                "springConstant": 0.1,
                "avoidOverlap": 0.8
            },
            "solver": "forceAtlas2Based",
            "stabilization": {
                "enabled": true,
                "iterations": 200,
                "updateInterval": 25,
                "fit": true
            },
            "maxVelocity": 50,
            "minVelocity": 0.1
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 50,
            "hideEdgesOnDrag": true,
            "hideEdgesOnZoom": true,
            "zoomView": true,
            "dragView": true
        }
    }
    ''')

    # Get PPR scores and compute percentiles for better color distribution
    all_ppr = [v['ppr_score'] for v in ppr_scores.values()]
    max_ppr = max(all_ppr) if all_ppr else 1
    min_ppr = min(all_ppr) if all_ppr else 0

    # Sort scores to compute percentiles
    sorted_scores = sorted(all_ppr)

    def get_percentile(score):
        """Get percentile rank of a score (0-100)"""
        if len(sorted_scores) <= 1:
            return 50
        # Find position in sorted list
        count_below = sum(1 for s in sorted_scores if s < score)
        return (count_below / len(sorted_scores)) * 100

    # Store chunk contents for click-to-view
    chunk_contents = {}

    # Professional color palette based on percentile ranking
    def get_color(score, is_entity=True, is_query_entity=False, is_passage=False):
        if is_query_entity:
            return "#00ba7c"  # Green - query seed nodes

        percentile = get_percentile(score)

        if is_passage:
            # Passage nodes: Blue gradient (professional)
            if percentile < 50:
                return "#536471"  # Muted gray - bottom 50%
            elif percentile < 75:
                return "#1d9bf0"  # Twitter blue - 50-75%
            elif percentile < 90:
                return "#794bc4"  # Purple - 75-90%
            else:
                return "#f91880"  # Pink/Magenta - top 10%
        else:
            # Entity nodes: Warm gradient
            if percentile < 50:
                return "#536471"  # Muted gray - bottom 50%
            elif percentile < 70:
                return "#ffd400"  # Gold - 50-70%
            elif percentile < 85:
                return "#ff7a00"  # Orange - 70-85%
            elif percentile < 95:
                return "#f4212e"  # Red - 85-95%
            else:
                return "#ff00ff"  # Magenta - top 5%

    # Import for hash ID computation
    from src.hipporag.utils.misc_utils import compute_mdhash_id

    # Pre-compute query entity keys and their connected chunks
    query_entity_keys = set()
    connected_chunk_keys = set()
    for qe in query_entities:
        qe_key = compute_mdhash_id(content=qe, prefix="entity-")
        query_entity_keys.add(qe_key)
        chunk_ids = hipporag.ent_node_to_chunk_ids.get(qe_key, set())
        connected_chunk_keys.update(chunk_ids)

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

        # Check if this chunk is connected to a query entity
        is_connected_to_query = hash_id in connected_chunk_keys

        # Size based on percentile (for better visual distribution)
        percentile = get_percentile(ppr_score)
        size_factor = percentile / 100  # 0 to 1 scale

        if is_passage:
            base_size = 20
            shape = "box"
            # Show passage content preview and store full content
            try:
                content = hipporag.chunk_embedding_store.get_row(hash_id)["content"]
                label = content[:50] + "..." if len(content) > 50 else content
                # Store full content for click-to-view (escape for JavaScript)
                chunk_contents[node_name] = content.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '')
            except:
                label = "Passage"
                chunk_contents[node_name] = "Content not available"
        else:
            base_size = 15
            shape = "dot"
            label = node_name

        if is_query_entity:
            base_size = 30  # Larger for query entities

        if is_connected_to_query:
            base_size = 35  # Even larger for connected chunks

        size = base_size + size_factor * 30

        # Special color for connected chunks
        if is_connected_to_query:
            color = "#00ba7c"  # Green for query-connected chunks
            border_color = "#00ff9d"
            border_width = 4
        elif is_query_entity:
            color = get_color(ppr_score, is_entity, is_query_entity, is_passage)
            border_color = "#00ba7c"
            border_width = 3
        else:
            color = get_color(ppr_score, is_entity, is_query_entity, is_passage)
            border_color = color
            border_width = 1

        # Tooltip with score info
        node_type = 'Query Entity' if is_query_entity else 'Connected Chunk' if is_connected_to_query else 'Entity' if is_entity else 'Passage'
        title = f"""
        <b>{node_name}</b><br>
        Type: {node_type}<br>
        PPR Score: {ppr_score:.6f}<br>
        Initial Weight: {initial_weight:.6f}<br>
        Hash: {hash_id}
        """

        net.add_node(
            node_name,
            label=label[:30] if len(label) > 30 else label,
            title=title,
            size=size,
            color={"background": color, "border": border_color, "highlight": {"background": "#00ff9d", "border": "#00ba7c"}},
            shape=shape,
            borderWidth=border_width,
            borderWidthSelected=5
        )

    # Get top passage keys from the top_passages data
    top_passage_keys = set()
    for p in scores_data.get("top_passages", [])[:5]:
        # Find the passage key by matching content
        for passage_key in hipporag.passage_node_keys:
            try:
                content = hipporag.chunk_embedding_store.get_row(passage_key).get("content", "")
                if p["content"][:50] in content:
                    top_passage_keys.add(passage_key)
                    break
            except:
                pass

    # Combine all highlighted node keys
    highlighted_node_keys = query_entity_keys | connected_chunk_keys | top_passage_keys

    # Add edges with highlighting for relevant connections
    for e in hipporag.graph.es:
        source = hipporag.graph.vs[e.source]['name']
        target = hipporag.graph.vs[e.target]['name']
        source_hash = hipporag.graph.vs[e.source]['hash_id'] if 'hash_id' in hipporag.graph.vs.attributes() else ''
        target_hash = hipporag.graph.vs[e.target]['hash_id'] if 'hash_id' in hipporag.graph.vs.attributes() else ''
        weight = e['weight'] if 'weight' in hipporag.graph.es.attributes() else 1

        # Check if this edge connects query entities to chunks
        source_is_query_entity = source_hash in query_entity_keys
        target_is_query_entity = target_hash in query_entity_keys
        source_is_connected_chunk = source_hash in connected_chunk_keys
        target_is_connected_chunk = target_hash in connected_chunk_keys
        source_is_top_passage = source_hash in top_passage_keys
        target_is_top_passage = target_hash in top_passage_keys

        # Determine edge style based on relevance
        if (source_is_query_entity and target_is_connected_chunk) or (target_is_query_entity and source_is_connected_chunk):
            # Direct edge from query entity to its chunk - GOLD highlight
            edge_color = "#00ba7c"
            edge_width = 4
            edge_dashes = False
        elif (source_is_query_entity or target_is_query_entity) and (source_is_top_passage or target_is_top_passage):
            # Edge involving query entity and top passage - GREEN highlight
            edge_color = "#00ba7c"
            edge_width = 3
            edge_dashes = False
        elif source_is_query_entity or target_is_query_entity:
            # Edge connected to query entity - CYAN highlight
            edge_color = "#1d9bf0"
            edge_width = 2.5
            edge_dashes = False
        elif source_is_top_passage or target_is_top_passage:
            # Edge connected to top passage - PURPLE highlight
            edge_color = "#794bc4"
            edge_width = 2
            edge_dashes = False
        elif source_hash in highlighted_node_keys or target_hash in highlighted_node_keys:
            # Edge connected to any highlighted node - lighter highlight
            edge_color = "#536471"
            edge_width = 1.5
            edge_dashes = False
        else:
            # Default edge - dim gray
            edge_color = "#2a3540"
            edge_width = 0.5
            edge_dashes = False

        net.add_edge(
            source,
            target,
            value=weight,
            title=f"Weight: {weight:.4f}",
            color=edge_color,
            width=edge_width,
            dashes=edge_dashes
        )

    # Count node types for stats
    has_hash_id = 'hash_id' in hipporag.graph.vs.attributes()
    entity_count = sum(1 for v in hipporag.graph.vs if has_hash_id and v['hash_id'].startswith('entity'))
    passage_count = sum(1 for v in hipporag.graph.vs if has_hash_id and v['hash_id'].startswith('chunk'))

    # Professional HTML content
    html_content = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        * {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; box-sizing: border-box; }}
        html, body {{ margin: 0; padding: 0; width: 100vw; height: 100vh; overflow: hidden; background: #0f1419; }}
        .card {{ width: 100vw !important; height: 100vh !important; margin: 0 !important; padding: 0 !important; border: none !important; }}
        #mynetwork {{ width: 100vw !important; height: 100vh !important; position: fixed !important; top: 0 !important; left: 0 !important; border: none !important; }}
        #loadingBar {{ display: none !important; }}

        .panel {{
            position: fixed;
            background: linear-gradient(145deg, rgba(22, 32, 42, 0.98), rgba(15, 20, 25, 0.98));
            border: 1px solid #38444d;
            border-radius: 16px;
            color: #e7e9ea;
            z-index: 1000;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
            backdrop-filter: blur(10px);
        }}

        .panel h3 {{ margin: 0 0 12px 0; font-weight: 600; font-size: 15px; }}
        .panel p {{ margin: 6px 0; font-size: 13px; line-height: 1.5; }}
        .panel hr {{ border: none; border-top: 1px solid #38444d; margin: 12px 0; }}

        .legend-item {{
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 4px 0;
        }}
        .legend-dot {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            flex-shrink: 0;
        }}
        .legend-box {{
            width: 14px;
            height: 10px;
            border-radius: 3px;
            flex-shrink: 0;
        }}

        .stat-card {{
            background: rgba(29, 155, 240, 0.1);
            border: 1px solid rgba(29, 155, 240, 0.2);
            border-radius: 8px;
            padding: 8px 12px;
            margin: 8px 0;
        }}
        .stat-value {{ font-size: 20px; font-weight: 700; color: #1d9bf0; }}
        .stat-label {{ font-size: 11px; color: #71767b; text-transform: uppercase; letter-spacing: 0.5px; }}

        .passage-item {{
            background: rgba(255, 255, 255, 0.03);
            border-radius: 8px;
            padding: 10px;
            margin: 8px 0;
            border-left: 3px solid #1d9bf0;
        }}
        .passage-rank {{ color: #1d9bf0; font-weight: 600; }}
        .passage-score {{ color: #71767b; font-size: 11px; }}
        .passage-text {{ color: #e7e9ea; font-size: 12px; margin-top: 4px; line-height: 1.4; }}

        .fact-item {{
            font-size: 12px;
            color: #71767b;
            padding: 4px 0;
        }}
        .fact-predicate {{ color: #1d9bf0; }}

        #chunkModal {{
            display: none;
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0, 0, 0, 0.85);
            z-index: 2000;
            backdrop-filter: blur(5px);
        }}
        .modal-content {{
            position: relative;
            margin: 5% auto;
            padding: 24px;
            width: 80%;
            max-width: 700px;
            background: linear-gradient(145deg, #1e2732, #16202a);
            border: 1px solid #38444d;
            border-radius: 16px;
            color: #e7e9ea;
            box-shadow: 0 16px 48px rgba(0, 0, 0, 0.5);
        }}
        .modal-close {{
            position: absolute;
            top: 16px;
            right: 20px;
            font-size: 24px;
            cursor: pointer;
            color: #71767b;
            transition: color 0.2s;
        }}
        .modal-close:hover {{ color: #f4212e; }}
        #chunkContent {{
            white-space: pre-wrap;
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 13px;
            background: #0f1419;
            padding: 16px;
            border-radius: 8px;
            max-height: 55vh;
            overflow-y: auto;
            line-height: 1.6;
            border: 1px solid #38444d;
        }}
    </style>

    <!-- Modal for viewing chunk content -->
    <div id="chunkModal">
        <div class="modal-content">
            <span id="closeModal" class="modal-close">&times;</span>
            <h3 style="color: #1d9bf0; margin-bottom: 16px;">📄 Document Content</h3>
            <div id="chunkContent"></div>
        </div>
    </div>

    <!-- Left Panel: Query Info & Legend -->
    <div class="panel" style="top: 20px; left: 20px; width: 320px; padding: 20px; max-height: calc(100vh - 80px); overflow-y: auto;">
        <h3 style="color: #1d9bf0; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;">Query Analysis</h3>
        <p style="font-size: 14px; color: #e7e9ea; background: rgba(29, 155, 240, 0.1); padding: 12px; border-radius: 8px; border-left: 3px solid #1d9bf0;">"{query}"</p>

        <div style="display: flex; gap: 10px; margin: 16px 0;">
            <div class="stat-card" style="flex: 1; text-align: center;">
                <div class="stat-value">{entity_count}</div>
                <div class="stat-label">Entities</div>
            </div>
            <div class="stat-card" style="flex: 1; text-align: center;">
                <div class="stat-value">{passage_count}</div>
                <div class="stat-label">Passages</div>
            </div>
            <div class="stat-card" style="flex: 1; text-align: center;">
                <div class="stat-value">{len(hipporag.graph.es)}</div>
                <div class="stat-label">Edges</div>
            </div>
        </div>

        <p style="font-size: 12px;"><b>Matched Entities:</b></p>
        <p style="color: #00ba7c; font-size: 13px;">{', '.join(query_entities) if query_entities else 'None found'}</p>

        <hr>
        <p style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: #71767b;">Entity Relevance</p>
        <div class="legend-item"><div class="legend-dot" style="background: #00ba7c;"></div><span>Query Entity (seed)</span></div>
        <div class="legend-item"><div class="legend-dot" style="background: #ff00ff;"></div><span>Top 5%</span></div>
        <div class="legend-item"><div class="legend-dot" style="background: #f4212e;"></div><span>Top 5-15%</span></div>
        <div class="legend-item"><div class="legend-dot" style="background: #ff7a00;"></div><span>Top 15-30%</span></div>
        <div class="legend-item"><div class="legend-dot" style="background: #ffd400;"></div><span>Top 30-50%</span></div>
        <div class="legend-item"><div class="legend-dot" style="background: #536471;"></div><span>Bottom 50%</span></div>

        <hr>
        <p style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: #71767b;">Passage Relevance</p>
        <div class="legend-item"><div class="legend-box" style="background: #00ba7c; border: 2px solid #00ff9d;"></div><span>Connected to Query Entity</span></div>
        <div class="legend-item"><div class="legend-box" style="background: #f91880;"></div><span>Top 10%</span></div>
        <div class="legend-item"><div class="legend-box" style="background: #794bc4;"></div><span>Top 10-25%</span></div>
        <div class="legend-item"><div class="legend-box" style="background: #1d9bf0;"></div><span>Top 25-50%</span></div>
        <div class="legend-item"><div class="legend-box" style="background: #536471;"></div><span>Bottom 50%</span></div>

        <hr>
        <p style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: #71767b;">Edge Highlighting</p>
        <div class="legend-item"><div style="width: 24px; height: 4px; background: #00ba7c; border-radius: 2px;"></div><span>Entity → Chunk (direct)</span></div>
        <div class="legend-item"><div style="width: 24px; height: 3px; background: #1d9bf0; border-radius: 2px;"></div><span>Query entity connections</span></div>
        <div class="legend-item"><div style="width: 24px; height: 2px; background: #794bc4; border-radius: 2px;"></div><span>Top passage connections</span></div>
        <div class="legend-item"><div style="width: 24px; height: 1px; background: #2a3540; border-radius: 2px;"></div><span>Other edges</span></div>

        <p style="font-size: 11px; color: #71767b; margin-top: 8px;">💡 Click passage nodes to view content</p>
    </div>

    <!-- Right Panel: Retrieved Results -->
    <div class="panel" style="top: 20px; right: 20px; width: 340px; padding: 20px; max-height: calc(100vh - 80px); overflow-y: auto;">
        <h3 style="color: #1d9bf0; font-size: 11px; text-transform: uppercase; letter-spacing: 1px;">Top Retrieved Passages</h3>
    """

    for p in scores_data["top_passages"][:5]:
        html_content += f'''
        <div class="passage-item">
            <span class="passage-rank">#{p['rank']}</span>
            <span class="passage-score">Score: {p['score']:.6f}</span>
            <div class="passage-text">{p['content'][:120]}...</div>
        </div>'''

    html_content += """
        <hr>
        <h3 style="color: #1d9bf0; font-size: 11px; text-transform: uppercase; letter-spacing: 1px;">Matched Facts</h3>
    """

    for f in scores_data["top_facts"][:5]:
        html_content += f'''<div class="fact-item">({f['subject']}, <span class="fact-predicate">{f['predicate']}</span>, {f['object']})</div>'''

    if not scores_data["top_facts"]:
        html_content += '<p style="color: #71767b; font-size: 12px;">No facts matched</p>'

    html_content += "</div>"

    # JavaScript for chunk content storage and modal
    import json
    chunk_json = json.dumps(chunk_contents, ensure_ascii=False)

    html_content += f"""
    <script>
    var chunkContents = {chunk_json};

    // Close modal on click
    document.getElementById('closeModal').onclick = function() {{
        document.getElementById('chunkModal').style.display = 'none';
    }};

    // Close modal on outside click
    document.getElementById('chunkModal').onclick = function(e) {{
        if (e.target === this) {{
            this.style.display = 'none';
        }}
    }};

    // Close on Escape key
    document.onkeydown = function(e) {{
        if (e.key === 'Escape') {{
            document.getElementById('chunkModal').style.display = 'none';
        }}
    }};
    </script>
    """

    # Save and customize HTML
    net.save_graph(output_path)

    # Add legend to the HTML
    with open(output_path, 'r', encoding='utf-8') as f:
        html = f.read()

    html = html.replace('<body>', f'<body>{html_content}')

    # Add network click handler and physics stabilization
    network_click_handler = """
    <script>
    // Wait for network to be ready, then add handlers
    function setupNetwork() {
        if (typeof network !== 'undefined') {
            // DISABLE PHYSICS after stabilization - makes graph stable
            network.on('stabilizationIterationsDone', function() {
                console.log('Stabilization complete - disabling physics');
                network.setOptions({ physics: { enabled: false } });
            });

            // Also disable after stabilized event
            network.on('stabilized', function() {
                console.log('Graph stabilized - disabling physics');
                network.setOptions({ physics: { enabled: false } });
            });

            // Click handler for viewing chunk content
            network.on('click', function(params) {
                if (params.nodes.length > 0) {
                    var nodeId = params.nodes[0];
                    if (chunkContents[nodeId]) {
                        document.getElementById('chunkContent').innerText = chunkContents[nodeId];
                        document.getElementById('chunkModal').style.display = 'block';
                    }
                }
            });

            // Double-click to re-enable physics temporarily (for repositioning)
            network.on('doubleClick', function(params) {
                if (params.nodes.length === 0) {
                    console.log('Re-enabling physics for 3 seconds...');
                    network.setOptions({ physics: { enabled: true } });
                    setTimeout(function() {
                        network.setOptions({ physics: { enabled: false } });
                        console.log('Physics disabled again');
                    }, 3000);
                }
            });

            console.log('Network handlers added');
        } else {
            setTimeout(setupNetwork, 100);
        }
    }
    // Start checking after DOM loads
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', setupNetwork);
    } else {
        setTimeout(setupNetwork, 500);
    }
    </script>
    """

    html = html.replace('</body>', f'{network_click_handler}</body>')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"Query visualization saved to: {output_path}")
    return output_path


def visualize_query(query: str = None):
    """Main function to visualize query relevance on knowledge graph."""
    from src.hipporag import HippoRAG

    print("Loading HippoRAG...")
    hipporag = HippoRAG(
        save_dir='outputs',
        llm_model_name='qwen3-next:80b-a3b-instruct-q4_K_M',
        embedding_model_name='Transformers/intfloat/multilingual-e5-large'
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
