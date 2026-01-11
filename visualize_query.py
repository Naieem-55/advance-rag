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


def create_query_visualization(hipporag, query: str, output_path: str = "outputs/query_graph.html", max_nodes: int = 200) -> str:
    """
    Create a focused visualization showing only query-relevant nodes:
    - Query entities (extracted from matched facts)
    - Documents/passages connected to those entities
    - Top retrieved passages
    """
    import time
    start_time = time.time()

    # Get relevance scores
    print(f"  [viz] Getting relevance scores...")
    scores_data = get_query_relevance_scores(hipporag, query)

    if "error" in scores_data:
        print(f"Error: {scores_data['error']}")
        return None

    ppr_scores = scores_data.get("ppr_scores", {})
    query_entities = set(scores_data.get("query_entities", []))
    top_passages = scores_data.get("top_passages", [])
    top_facts = scores_data.get("top_facts", [])
    retrieval_method = scores_data.get("retrieval_method", "hybrid_ppr_dpr")
    use_dpr_only = retrieval_method == "dpr_only"

    if use_dpr_only:
        print(f"  [viz] Warning: {scores_data.get('warning', 'No facts matched')}")

    # Build focused subgraph with only relevant nodes
    from src.hipporag.utils.misc_utils import compute_mdhash_id

    top_node_names = set()
    query_entity_keys = set()
    connected_chunk_keys = set()

    # 1. Add query entities and their connected chunks
    print(f"  [viz] Finding query entities and connected documents...")

    # Helper to safely get vertex attribute
    def get_hash_id(vertex):
        return vertex['hash_id'] if 'hash_id' in hipporag.graph.vs.attributes() else ''

    for qe in query_entities:
        qe_key = compute_mdhash_id(content=qe, prefix="entity-")
        query_entity_keys.add(qe_key)

        # Find the entity node
        for v in hipporag.graph.vs:
            if get_hash_id(v) == qe_key:
                top_node_names.add(v['name'])
                break

        # Get chunks connected to this entity
        chunk_ids = hipporag.ent_node_to_chunk_ids.get(qe_key, set())
        connected_chunk_keys.update(chunk_ids)

    # 2. Add connected chunk nodes
    for v in hipporag.graph.vs:
        if get_hash_id(v) in connected_chunk_keys:
            top_node_names.add(v['name'])

    # 3. Add top retrieved passages (find by content match)
    print(f"  [viz] Adding top {len(top_passages)} retrieved passages...")
    top_passage_keys = set()
    for p in top_passages[:10]:
        for passage_key in hipporag.passage_node_keys:
            try:
                content = hipporag.chunk_embedding_store.get_row(passage_key).get("content", "")
                if p["content"][:50] in content:
                    top_passage_keys.add(passage_key)
                    # Find node name
                    for v in hipporag.graph.vs:
                        if get_hash_id(v) == passage_key:
                            top_node_names.add(v['name'])
                            break
                    break
            except:
                pass

    # 4. Add entities from top facts that might not be in query_entities
    for fact in top_facts[:10]:
        for phrase in [fact.get('subject', ''), fact.get('object', '')]:
            if phrase:
                phrase_key = compute_mdhash_id(content=phrase.lower(), prefix="entity-")
                for v in hipporag.graph.vs:
                    if get_hash_id(v) == phrase_key:
                        top_node_names.add(v['name'])
                        # Also get chunks for these entities
                        chunk_ids = hipporag.ent_node_to_chunk_ids.get(phrase_key, set())
                        for cv in hipporag.graph.vs:
                            if get_hash_id(cv) in chunk_ids:
                                top_node_names.add(cv['name'])
                        break

    print(f"  [viz] Selected {len(top_node_names)} relevant nodes for visualization")

    # Create network with CDN resources to avoid 404 errors
    net = Network(
        height="100vh",
        width="100vw",
        bgcolor="#0a0e14",
        font_color="#e6e6e6",
        cdn_resources='in_line'  # Embed all JS/CSS to avoid 404 errors
    )

    # Professional physics settings - optimized for large graphs
    net.set_options('''
    {
        "nodes": {
            "font": {
                "size": 11,
                "face": "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
                "color": "#ffffff",
                "strokeWidth": 2,
                "strokeColor": "#0a0e14"
            },
            "borderWidth": 1,
            "borderWidthSelected": 3,
            "shadow": {
                "enabled": true,
                "color": "rgba(0,0,0,0.3)",
                "size": 8,
                "x": 0,
                "y": 2
            }
        },
        "edges": {
            "color": {"color": "rgba(100,100,120,0.3)", "highlight": "#6366f1", "hover": "#818cf8"},
            "smooth": {"type": "continuous", "roundness": 0.3},
            "width": 0.5,
            "selectionWidth": 2
        },
        "physics": {
            "enabled": true,
            "barnesHut": {
                "gravitationalConstant": -8000,
                "centralGravity": 0.15,
                "springLength": 120,
                "springConstant": 0.02,
                "damping": 0.5,
                "avoidOverlap": 0.5
            },
            "solver": "barnesHut",
            "stabilization": {
                "enabled": true,
                "iterations": 150,
                "updateInterval": 25,
                "fit": true
            },
            "maxVelocity": 30,
            "minVelocity": 0.75
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "hideEdgesOnDrag": true,
            "hideEdgesOnZoom": true,
            "zoomView": true,
            "dragView": true,
            "keyboard": {"enabled": true}
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

    # Simple, clear color scheme for focused visualization
    def get_color_config(is_query_entity=False, is_connected_chunk=False, is_top_passage=False, is_entity=True):
        """Returns color config dict for vis.js nodes"""
        if is_query_entity:
            # Bright green for query entities (seed nodes)
            return {
                'background': '#10b981',
                'border': '#059669',
                'highlight': {'background': '#34d399', 'border': '#10b981'}
            }
        elif is_connected_chunk:
            # Blue for chunks directly connected to query entities
            return {
                'background': '#3b82f6',
                'border': '#2563eb',
                'highlight': {'background': '#60a5fa', 'border': '#3b82f6'}
            }
        elif is_top_passage:
            # Purple for top retrieved passages
            return {
                'background': '#8b5cf6',
                'border': '#7c3aed',
                'highlight': {'background': '#a78bfa', 'border': '#8b5cf6'}
            }
        elif is_entity:
            # Indigo for other entities (from facts)
            return {
                'background': '#6366f1',
                'border': '#4f46e5',
                'highlight': {'background': '#818cf8', 'border': '#6366f1'}
            }
        else:
            # Gray for other nodes
            return {
                'background': '#6b7280',
                'border': '#4b5563',
                'highlight': {'background': '#9ca3af', 'border': '#6b7280'}
            }

    # Note: query_entity_keys and connected_chunk_keys already computed above

    # Build index of included nodes for edge filtering
    included_node_indices = set()

    # Add nodes (only those in top_node_names for performance)
    print(f"  [viz] Adding {len(top_node_names)} nodes...")
    node_count = 0
    for v in hipporag.graph.vs:
        node_name = v['name']

        # Skip nodes not in the filtered set
        if node_name not in top_node_names:
            continue

        included_node_indices.add(v.index)
        node_count += 1

        hash_id = v['hash_id'] if 'hash_id' in hipporag.graph.vs.attributes() else ''

        score_data = ppr_scores.get(node_name, {})
        ppr_score = score_data.get('ppr_score', 0)
        initial_weight = score_data.get('initial_weight', 0)

        is_entity = hash_id.startswith('entity')
        is_passage = hash_id.startswith('chunk')
        is_query_entity = hash_id in query_entity_keys
        is_connected_chunk = hash_id in connected_chunk_keys
        is_top_passage = hash_id in top_passage_keys

        # Determine node appearance based on type
        if is_passage:
            shape = "box"
            # Show passage content preview and store full content
            try:
                content = hipporag.chunk_embedding_store.get_row(hash_id)["content"]
                label = content[:50] + "..." if len(content) > 50 else content
                chunk_contents[node_name] = content.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '')
            except:
                label = "Document"
                chunk_contents[node_name] = "Content not available"

            # Size based on relevance
            if is_connected_chunk:
                size = 35  # Largest - directly connected to query entity
                node_type_label = "Connected Document"
                badge_color = "#3b82f6"
            elif is_top_passage:
                size = 30  # Large - top retrieved
                node_type_label = "Retrieved Document"
                badge_color = "#8b5cf6"
            else:
                size = 25
                node_type_label = "Document"
                badge_color = "#6b7280"
        else:
            shape = "dot"
            label = node_name

            if is_query_entity:
                size = 40  # Largest - query seed entity
                node_type_label = "Query Entity"
                badge_color = "#10b981"
            else:
                size = 30  # Related entity from facts
                node_type_label = "Related Entity"
                badge_color = "#6366f1"

        # Get color config
        color_config = get_color_config(
            is_query_entity=is_query_entity,
            is_connected_chunk=is_connected_chunk,
            is_top_passage=is_top_passage,
            is_entity=is_entity
        )

        # Border width
        border_width = 3 if (is_query_entity or is_connected_chunk) else 2

        # Tooltip
        title = f'''<div style="font-family: system-ui; padding: 10px; max-width: 350px;">
            <div style="font-weight: 600; font-size: 13px; color: #1e293b; margin-bottom: 8px;">{label[:80]}{"..." if len(label) > 80 else ""}</div>
            <div style="margin-bottom: 8px;">
                <span style="background: {badge_color}22; color: {badge_color}; padding: 3px 10px; border-radius: 4px; font-size: 11px; font-weight: 600;">{node_type_label.upper()}</span>
            </div>
            <div style="font-size: 11px; color: #64748b; line-height: 1.5;">
                <div><b>PPR Score:</b> {ppr_score:.6f}</div>
            </div>
        </div>'''

        net.add_node(
            node_name,
            label=label[:28] if len(label) > 28 else label,
            title=title,
            size=size,
            color=color_config,
            shape=shape,
            borderWidth=border_width,
            borderWidthSelected=4
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

    # Add edges with highlighting for relevant connections (only between included nodes)
    print(f"  [viz] Adding edges between {node_count} nodes...")
    edge_count = 0
    for e in hipporag.graph.es:
        # Skip edges not between included nodes
        if e.source not in included_node_indices or e.target not in included_node_indices:
            continue

        edge_count += 1
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

        # Determine edge style based on relevance (matching professional design)
        if (source_is_query_entity and target_is_connected_chunk) or (target_is_query_entity and source_is_connected_chunk):
            # Direct edge from query entity to its chunk - Emerald highlight
            edge_color = "#10b981"
            edge_width = 3
        elif (source_is_query_entity or target_is_query_entity) and (source_is_top_passage or target_is_top_passage):
            # Edge involving query entity and top passage - Green highlight
            edge_color = "#34d399"
            edge_width = 2.5
        elif source_is_query_entity or target_is_query_entity:
            # Edge connected to query entity - Indigo highlight
            edge_color = "#6366f1"
            edge_width = 2
        elif source_is_top_passage or target_is_top_passage:
            # Edge connected to top passage - Purple highlight
            edge_color = "#8b5cf6"
            edge_width = 1.5
        elif source_hash in highlighted_node_keys or target_hash in highlighted_node_keys:
            # Edge connected to any highlighted node - lighter highlight
            edge_color = "rgba(100,100,120,0.5)"
            edge_width = 1
        else:
            # Default edge - very dim
            edge_color = "rgba(100,100,120,0.2)"
            edge_width = 0.3

        # Professional edge tooltip
        edge_title = f'''<div style="font-family: system-ui; padding: 6px; font-size: 11px;">
            <div style="color: #64748b;">Edge Weight</div>
            <div style="font-weight: 500; color: #1e293b;">{weight:.4f}</div>
        </div>'''

        net.add_edge(
            source,
            target,
            value=weight,
            title=edge_title,
            color=edge_color,
            width=edge_width
        )

    print(f"  [viz] Added {edge_count} edges")

    # Count node types for stats
    total_nodes = len(hipporag.graph.vs)
    total_edges = len(hipporag.graph.es)
    shown_nodes = node_count
    shown_edges = edge_count
    shown_passages = len(chunk_contents)

    # Professional HTML content (matching main KG visualization design)
    html_content = f"""
    <style>
        :root {{
            --bg-primary: #0a0e14;
            --bg-secondary: #111827;
            --bg-tertiary: #1f2937;
            --border-color: #374151;
            --text-primary: #f9fafb;
            --text-secondary: #9ca3af;
            --text-muted: #6b7280;
            --accent-indigo: #6366f1;
            --accent-pink: #ec4899;
            --accent-emerald: #10b981;
            --accent-purple: #8b5cf6;
            --accent-amber: #f59e0b;
            --accent-red: #ef4444;
        }}
        * {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; box-sizing: border-box; margin: 0; padding: 0; }}
        html, body {{ width: 100vw; height: 100vh; overflow: hidden; background: var(--bg-primary); }}
        .card {{ width: 100vw !important; height: 100vh !important; margin: 0 !important; padding: 0 !important; border: none !important; background: transparent !important; }}
        #mynetwork {{ width: 100vw !important; height: 100vh !important; position: fixed !important; top: 0 !important; left: 0 !important; border: none !important; background: var(--bg-primary) !important; }}
        #loadingBar {{ display: none !important; }}

        /* Loading overlay */
        .loading-overlay {{
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: var(--bg-primary);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            z-index: 2000;
            transition: opacity 0.5s;
        }}
        .loading-overlay.hidden {{
            opacity: 0;
            pointer-events: none;
        }}
        .loading-spinner {{
            width: 40px;
            height: 40px;
            border: 3px solid var(--border-color);
            border-top-color: var(--accent-indigo);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }}
        .loading-text {{
            margin-top: 16px;
            color: var(--text-secondary);
            font-size: 13px;
        }}
        @keyframes spin {{
            to {{ transform: rotate(360deg); }}
        }}

        /* Panel styling */
        .panel {{
            position: fixed;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            color: var(--text-primary);
            z-index: 1000;
            box-shadow: 0 4px 24px rgba(0, 0, 0, 0.5);
            overflow: hidden;
        }}
        .panel-header {{
            padding: 14px 18px;
            border-bottom: 1px solid var(--border-color);
            background: linear-gradient(135deg, var(--bg-tertiary), var(--bg-secondary));
        }}
        .panel-title {{
            font-size: 13px;
            font-weight: 600;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .panel-body {{
            padding: 14px 18px;
            max-height: calc(100vh - 160px);
            overflow-y: auto;
        }}
        .panel-body::-webkit-scrollbar {{
            width: 6px;
        }}
        .panel-body::-webkit-scrollbar-track {{
            background: var(--bg-tertiary);
        }}
        .panel-body::-webkit-scrollbar-thumb {{
            background: var(--border-color);
            border-radius: 3px;
        }}

        /* Query box */
        .query-box {{
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-left: 3px solid var(--accent-indigo);
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 14px;
            font-size: 13px;
            line-height: 1.5;
            color: var(--text-primary);
        }}

        /* Stats grid */
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 8px;
            margin-bottom: 14px;
        }}
        .stat-card {{
            background: var(--bg-tertiary);
            border-radius: 8px;
            padding: 10px 8px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 18px;
            font-weight: 700;
            line-height: 1.2;
        }}
        .stat-card.entities .stat-value {{ color: var(--accent-indigo); }}
        .stat-card.passages .stat-value {{ color: var(--accent-pink); }}
        .stat-card.edges .stat-value {{ color: var(--accent-emerald); }}
        .stat-label {{
            font-size: 9px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-top: 2px;
        }}

        /* Matched entities */
        .entities-box {{
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.2);
            border-radius: 8px;
            padding: 10px 12px;
            margin-bottom: 14px;
        }}
        .entities-label {{
            font-size: 10px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 6px;
        }}
        .entities-list {{
            font-size: 12px;
            color: var(--accent-emerald);
            line-height: 1.5;
        }}

        /* Section divider */
        .section-title {{
            font-size: 10px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin: 14px 0 8px 0;
            padding-top: 10px;
            border-top: 1px solid var(--border-color);
        }}

        /* Legend items */
        .legend-grid {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 4px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 5px 8px;
            background: var(--bg-tertiary);
            border-radius: 6px;
            font-size: 11px;
            color: var(--text-secondary);
        }}
        .legend-dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            flex-shrink: 0;
        }}
        .legend-box {{
            width: 12px;
            height: 10px;
            border-radius: 3px;
            flex-shrink: 0;
        }}
        .legend-line {{
            width: 20px;
            height: 3px;
            border-radius: 2px;
            flex-shrink: 0;
        }}

        /* Passage items */
        .passage-item {{
            background: var(--bg-tertiary);
            border-radius: 8px;
            padding: 10px 12px;
            margin: 8px 0;
            border-left: 3px solid var(--accent-indigo);
        }}
        .passage-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
        }}
        .passage-rank {{
            font-size: 11px;
            font-weight: 600;
            color: var(--accent-indigo);
            background: rgba(99, 102, 241, 0.15);
            padding: 2px 8px;
            border-radius: 4px;
        }}
        .passage-score {{
            font-size: 10px;
            color: var(--text-muted);
        }}
        .passage-text {{
            font-size: 11px;
            color: var(--text-secondary);
            line-height: 1.5;
        }}

        /* Fact items */
        .fact-item {{
            font-size: 11px;
            color: var(--text-secondary);
            padding: 6px 8px;
            background: var(--bg-tertiary);
            border-radius: 6px;
            margin: 4px 0;
        }}
        .fact-subject {{ color: var(--accent-indigo); }}
        .fact-predicate {{ color: var(--accent-amber); }}
        .fact-object {{ color: var(--accent-emerald); }}

        /* Modal */
        #chunkModal {{
            display: none;
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0, 0, 0, 0.85);
            z-index: 3000;
            backdrop-filter: blur(5px);
        }}
        .modal-content {{
            position: relative;
            margin: 5% auto;
            padding: 0;
            width: 85%;
            max-width: 750px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            color: var(--text-primary);
            box-shadow: 0 16px 48px rgba(0, 0, 0, 0.5);
            overflow: hidden;
        }}
        .modal-header {{
            padding: 16px 20px;
            border-bottom: 1px solid var(--border-color);
            background: var(--bg-tertiary);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .modal-title {{
            font-size: 14px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .modal-close {{
            background: none;
            border: none;
            color: var(--text-muted);
            cursor: pointer;
            font-size: 20px;
            padding: 4px;
            line-height: 1;
            transition: color 0.2s;
        }}
        .modal-close:hover {{ color: var(--accent-red); }}
        #chunkContent {{
            white-space: pre-wrap;
            font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace;
            font-size: 12px;
            background: var(--bg-primary);
            padding: 20px;
            max-height: 60vh;
            overflow-y: auto;
            line-height: 1.7;
            color: var(--text-secondary);
        }}

        /* Zoom controls */
        .zoom-controls {{
            position: fixed;
            bottom: 20px;
            right: 20px;
            display: flex;
            flex-direction: column;
            gap: 4px;
            z-index: 1000;
        }}
        .zoom-btn {{
            width: 36px;
            height: 36px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            color: var(--text-secondary);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
            transition: all 0.2s;
        }}
        .zoom-btn:hover {{
            background: var(--bg-tertiary);
            color: var(--text-primary);
            border-color: var(--accent-indigo);
        }}

        /* Tip */
        .tip {{
            font-size: 10px;
            color: var(--text-muted);
            margin-top: 12px;
            padding: 8px;
            background: var(--bg-tertiary);
            border-radius: 6px;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
    </style>

    <!-- Loading overlay -->
    <div class="loading-overlay" id="loadingOverlay">
        <div class="loading-spinner"></div>
        <div class="loading-text">Analyzing query relevance...</div>
    </div>

    <!-- Modal for viewing chunk content -->
    <div id="chunkModal">
        <div class="modal-content">
            <div class="modal-header">
                <div class="modal-title">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                        <polyline points="14 2 14 8 20 8"/>
                        <line x1="16" y1="13" x2="8" y2="13"/>
                        <line x1="16" y1="17" x2="8" y2="17"/>
                    </svg>
                    Document Content
                </div>
                <button class="modal-close" id="closeModal">&times;</button>
            </div>
            <div id="chunkContent"></div>
        </div>
    </div>

    <!-- Left Panel: Query Info & Legend -->
    <div class="panel" style="top: 16px; left: 16px; width: 280px;">
        <div class="panel-header">
            <div class="panel-title">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="11" cy="11" r="8"/>
                    <path d="M21 21l-4.35-4.35"/>
                </svg>
                Query Analysis
            </div>
        </div>
        <div class="panel-body">
            <div class="query-box">"{query}"</div>

            <div class="stats-grid">
                <div class="stat-card entities">
                    <div class="stat-value">{shown_nodes:,}</div>
                    <div class="stat-label">Nodes Shown</div>
                </div>
                <div class="stat-card passages">
                    <div class="stat-value">{shown_passages:,}</div>
                    <div class="stat-label">Passages</div>
                </div>
                <div class="stat-card edges">
                    <div class="stat-value">{shown_edges:,}</div>
                    <div class="stat-label">Edges</div>
                </div>
            </div>
            <div style="font-size: 10px; color: var(--text-muted); margin-bottom: 12px; text-align: center;">
                Showing top relevant nodes from {total_nodes:,} total ({total_edges:,} edges)
            </div>

            <div class="entities-box">
                <div class="entities-label">Matched Entities</div>
                <div class="entities-list">{', '.join(query_entities) if query_entities else 'None found in knowledge graph'}</div>
            </div>

            <div class="section-title">Node Types</div>
            <div class="legend-grid">
                <div class="legend-item"><div class="legend-dot" style="background: #10b981;"></div><span>Query Entity (seed)</span></div>
                <div class="legend-item"><div class="legend-dot" style="background: #6366f1;"></div><span>Related Entity</span></div>
                <div class="legend-item"><div class="legend-box" style="background: #3b82f6;"></div><span>Connected Document</span></div>
                <div class="legend-item"><div class="legend-box" style="background: #8b5cf6;"></div><span>Retrieved Document</span></div>
            </div>

            <div class="section-title">Edge Highlighting</div>
            <div class="legend-grid">
                <div class="legend-item"><div class="legend-line" style="background: #10b981;"></div><span>Entity to chunk (direct)</span></div>
                <div class="legend-item"><div class="legend-line" style="background: #6366f1;"></div><span>Query entity connections</span></div>
                <div class="legend-item"><div class="legend-line" style="background: #8b5cf6;"></div><span>Top passage connections</span></div>
            </div>

            <div class="tip">Click passage nodes to view full content</div>
        </div>
    </div>

    <!-- Right Panel: Retrieved Results -->
    <div class="panel" style="top: 16px; right: 16px; width: 280px;">
        <div class="panel-header">
            <div class="panel-title">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
                </svg>
                Retrieved Passages
            </div>
        </div>
        <div class="panel-body">
    """

    for p in scores_data["top_passages"][:5]:
        html_content += f'''
            <div class="passage-item">
                <div class="passage-header">
                    <span class="passage-rank">#{p['rank']}</span>
                    <span class="passage-score">Score: {p['score']:.4f}</span>
                </div>
                <div class="passage-text">{p['content'][:150]}...</div>
            </div>'''

    html_content += """
            <div class="section-title">Matched Facts</div>
    """

    for f in scores_data["top_facts"][:5]:
        html_content += f'''<div class="fact-item">(<span class="fact-subject">{f['subject']}</span>, <span class="fact-predicate">{f['predicate']}</span>, <span class="fact-object">{f['object']}</span>)</div>'''

    if not scores_data["top_facts"]:
        html_content += '<div class="fact-item" style="color: var(--text-muted);">No facts matched in knowledge graph</div>'

    html_content += """
        </div>
    </div>

    <!-- Zoom controls -->
    <div class="zoom-controls">
        <button class="zoom-btn" onclick="zoomIn()" title="Zoom In">+</button>
        <button class="zoom-btn" onclick="zoomOut()" title="Zoom Out">-</button>
        <button class="zoom-btn" onclick="fitGraph()" title="Fit All">&#8689;</button>
        <button class="zoom-btn" onclick="togglePanels()" title="Toggle Panels" id="toggleBtn">&#9776;</button>
    </div>
    """

    # JavaScript for chunk content storage, modal, zoom controls, and loading overlay
    import json
    chunk_json = json.dumps(chunk_contents, ensure_ascii=False)

    html_content += f"""
    <script>
    var chunkContents = {chunk_json};

    // Zoom controls
    function zoomIn() {{
        if (typeof network !== 'undefined') {{
            var scale = network.getScale() * 1.3;
            network.moveTo({{ scale: scale, animation: {{ duration: 300 }} }});
        }}
    }}
    function zoomOut() {{
        if (typeof network !== 'undefined') {{
            var scale = network.getScale() / 1.3;
            network.moveTo({{ scale: scale, animation: {{ duration: 300 }} }});
        }}
    }}
    function fitGraph() {{
        if (typeof network !== 'undefined') {{
            network.fit({{ animation: {{ duration: 500, easingFunction: 'easeInOutQuad' }} }});
        }}
    }}

    // Toggle panels for full graph view
    var panelsVisible = true;
    function togglePanels() {{
        var panels = document.querySelectorAll('.panel');
        panelsVisible = !panelsVisible;
        panels.forEach(function(p) {{
            p.style.display = panelsVisible ? 'block' : 'none';
        }});
        document.getElementById('toggleBtn').style.background = panelsVisible ? '' : '#6366f1';
    }}

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
    print(f"  [viz] Generating HTML file...")
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
            // Hide loading overlay and disable physics after stabilization
            network.on('stabilizationIterationsDone', function() {
                network.setOptions({ physics: { enabled: false } });
                document.getElementById('loadingOverlay').classList.add('hidden');
            });

            network.on('stabilized', function() {
                network.setOptions({ physics: { enabled: false } });
                document.getElementById('loadingOverlay').classList.add('hidden');
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

            // Double-click to focus on node or re-enable physics
            network.on('doubleClick', function(params) {
                if (params.nodes.length > 0) {
                    // Focus on the clicked node
                    network.focus(params.nodes[0], {
                        scale: 1.5,
                        animation: { duration: 500, easingFunction: 'easeInOutQuad' }
                    });
                } else {
                    // Re-enable physics temporarily for repositioning
                    network.setOptions({ physics: { enabled: true } });
                    setTimeout(function() {
                        network.setOptions({ physics: { enabled: false } });
                    }, 3000);
                }
            });

            // Timeout to hide loading if stabilization takes too long
            setTimeout(function() {
                document.getElementById('loadingOverlay').classList.add('hidden');
            }, 8000);

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

    elapsed = time.time() - start_time
    print(f"  [viz] Done! {node_count} nodes, {edge_count} edges in {elapsed:.1f}s")
    print(f"  [viz] Saved to: {output_path}")
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
