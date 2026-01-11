"""
HippoRAG Knowledge Graph - Interactive Web Visualization
Opens an interactive graph visualization in your browser
"""

import pickle
import json
import os
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading

def load_graph(save_dir='outputs', llm_model=None, embedding_model=None):
    """Load the igraph object from pickle file. Auto-detects model directories if not specified."""

    # Auto-detect model directory if not specified
    if llm_model is None or embedding_model is None:
        if os.path.exists(save_dir):
            print(f"Scanning {save_dir} for graph.pickle...")
            for item in os.listdir(save_dir):
                item_path = os.path.join(save_dir, item)
                if os.path.isdir(item_path):
                    graph_file = os.path.join(item_path, "graph.pickle")
                    print(f"  Checking: {item_path}")
                    if os.path.exists(graph_file):
                        print(f"  Found graph at: {graph_file}")
                        with open(graph_file, 'rb') as f:
                            return pickle.load(f)
        print(f"No graph found in {save_dir}")
        return None

    graph_path = os.path.join(save_dir, f"{llm_model}_{embedding_model}", "graph.pickle")

    if not os.path.exists(graph_path):
        print(f"Graph not found at: {graph_path}")
        return None

    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)

    return graph

def load_openie_results(save_dir='outputs', llm_model=None):
    """Load OpenIE extraction results. Auto-detects if llm_model not specified."""

    # Auto-detect openie file if not specified
    if llm_model is None:
        if os.path.exists(save_dir):
            for item in os.listdir(save_dir):
                if item.startswith('openie_results_ner_') and item.endswith('.json'):
                    openie_path = os.path.join(save_dir, item)
                    print(f"Auto-detected OpenIE results at: {openie_path}")
                    with open(openie_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
        return None

    openie_path = os.path.join(save_dir, f"openie_results_ner_{llm_model}.json")

    if not os.path.exists(openie_path):
        return None

    with open(openie_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    return results

def create_pyvis_visualization(graph, output_file='outputs/knowledge_graph_interactive.html'):
    """Create interactive visualization using Pyvis."""
    try:
        from pyvis.network import Network
    except ImportError:
        print("Pyvis not installed. Installing now...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'pyvis'])
        from pyvis.network import Network

    # Create Pyvis network with CDN resources to avoid 404 errors
    net = Network(
        height='100vh',
        width='100vw',
        bgcolor='#0a0e14',
        font_color='#e6e6e6',
        directed=True,
        notebook=False,
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
            "color": {"color": "rgba(100,100,120,0.4)", "highlight": "#6366f1", "hover": "#818cf8"},
            "smooth": {"type": "continuous", "roundness": 0.3},
            "arrows": {"to": {"enabled": true, "scaleFactor": 0.3}},
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
            "keyboard": {"enabled": true},
            "multiselect": true,
            "navigationButtons": false
        }
    }
    ''')

    # Professional color palette with gradients
    color_config = {
        'entity': {
            'background': '#6366f1',  # Indigo
            'border': '#4f46e5',
            'highlight': {'background': '#818cf8', 'border': '#6366f1'}
        },
        'chunk': {
            'background': '#ec4899',  # Pink
            'border': '#db2777',
            'highlight': {'background': '#f472b6', 'border': '#ec4899'}
        },
        'passage': {
            'background': '#ec4899',
            'border': '#db2777',
            'highlight': {'background': '#f472b6', 'border': '#ec4899'}
        },
        'document': {
            'background': '#f59e0b',  # Amber
            'border': '#d97706',
            'highlight': {'background': '#fbbf24', 'border': '#f59e0b'}
        },
        'default': {
            'background': '#10b981',  # Emerald
            'border': '#059669',
            'highlight': {'background': '#34d399', 'border': '#10b981'}
        }
    }

    # Add nodes with professional styling
    for v in graph.vs:
        node_id = v.index

        # Get node content/label
        content = str(v['content']) if 'content' in graph.vs.attributes() else ''
        name = v['name'] if 'name' in graph.vs.attributes() else str(node_id)
        hash_id = v['hash_id'] if 'hash_id' in graph.vs.attributes() else ''

        # Determine node type from hash_id prefix
        node_type = hash_id.split('-')[0] if '-' in hash_id else 'default'

        # Create label (truncate if too long)
        if content and len(content) > 0:
            label = content[:25] + '...' if len(content) > 25 else content
        else:
            label = name[:25] + '...' if len(name) > 25 else name

        # Skip empty nodes - show type instead
        if not label or label.strip() == '':
            label = f"{node_type}_{node_id}"

        # Get color config
        colors = color_config.get(node_type, color_config['default'])

        # Node size: chunks are larger, entities sized by connections
        degree = graph.degree(node_id)
        if node_type == 'chunk':
            size = 30  # Larger size for passage/chunk nodes
            shape = 'box'
        else:
            size = 12 + min(degree * 2, 20)
            shape = 'dot'

        # Professional hover tooltip with styled HTML
        title = f'''<div style="font-family: system-ui; padding: 8px; max-width: 300px;">
            <div style="font-weight: 600; font-size: 13px; color: #1e293b; margin-bottom: 6px;">{label}</div>
            <div style="font-size: 11px; color: #64748b;">
                <span style="background: {colors['background']}22; color: {colors['background']}; padding: 2px 6px; border-radius: 4px; font-weight: 500;">{node_type.upper()}</span>
                <span style="margin-left: 8px;">{degree} connections</span>
            </div>'''
        if content and len(content) > 25:
            title += f'<div style="margin-top: 8px; font-size: 11px; color: #475569; line-height: 1.4; border-top: 1px solid #e2e8f0; padding-top: 8px;">{content[:300]}{"..." if len(content) > 300 else ""}</div>'
        title += '</div>'

        net.add_node(
            node_id,
            label=label,
            title=title,
            color=colors,
            size=size,
            shape=shape,
            borderWidth=1,
            borderWidthSelected=3
        )

    # Add edges with subtle styling
    for e in graph.es:
        source = e.source
        target = e.target

        # Edge label/title
        edge_type = e['type'] if 'type' in graph.es.attributes() else ''
        weight = e['weight'] if 'weight' in graph.es.attributes() else 1.0

        # Professional edge tooltip
        title = f'''<div style="font-family: system-ui; padding: 6px; font-size: 11px;">
            <div style="color: #64748b;">Relationship</div>
            <div style="font-weight: 500; color: #1e293b;">{edge_type if edge_type else "connected"}</div>
            <div style="color: #94a3b8; margin-top: 4px;">Weight: {weight:.4f}</div>
        </div>'''

        net.add_edge(
            source,
            target,
            title=title,
            width=0.3 + float(weight) * 1.5 if weight else 0.5
        )

    # Save to HTML
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    net.save_graph(output_file)

    # Count node types
    has_hash_id = 'hash_id' in graph.vs.attributes()
    entity_count = sum(1 for v in graph.vs if has_hash_id and v['hash_id'].startswith('entity'))
    chunk_count = sum(1 for v in graph.vs if has_hash_id and v['hash_id'].startswith('chunk'))

    # Add custom title and legend to HTML
    with open(output_file, 'r', encoding='utf-8') as f:
        html_content = f.read()

    legend_html = f'''
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
            --accent-amber: #f59e0b;
        }}
        * {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; box-sizing: border-box; margin: 0; padding: 0; }}
        html, body {{ width: 100vw; height: 100vh; overflow: hidden; background: var(--bg-primary); }}
        .card {{ width: 100vw !important; height: 100vh !important; margin: 0 !important; padding: 0 !important; border: none !important; background: transparent !important; }}
        #mynetwork {{ width: 100vw !important; height: 100vh !important; position: fixed !important; top: 0 !important; left: 0 !important; border: none !important; background: var(--bg-primary) !important; }}
        #loadingBar {{ display: none !important; }}

        /* Main control panel */
        .kg-panel {{
            position: fixed;
            top: 20px;
            left: 20px;
            width: 320px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            color: var(--text-primary);
            z-index: 1000;
            box-shadow: 0 4px 24px rgba(0, 0, 0, 0.5);
            overflow: hidden;
        }}
        .kg-header {{
            padding: 16px 20px;
            border-bottom: 1px solid var(--border-color);
            background: linear-gradient(135deg, var(--bg-tertiary), var(--bg-secondary));
        }}
        .kg-title {{
            font-size: 16px;
            font-weight: 600;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .kg-subtitle {{
            font-size: 11px;
            color: var(--text-muted);
            margin-top: 4px;
        }}
        .kg-body {{
            padding: 16px 20px;
        }}

        /* Search box */
        .search-container {{
            position: relative;
            margin-bottom: 16px;
        }}
        .search-input {{
            width: 100%;
            padding: 10px 12px 10px 36px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            color: var(--text-primary);
            font-size: 13px;
            outline: none;
            transition: border-color 0.2s, box-shadow 0.2s;
        }}
        .search-input:focus {{
            border-color: var(--accent-indigo);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15);
        }}
        .search-input::placeholder {{
            color: var(--text-muted);
        }}
        .search-icon {{
            position: absolute;
            left: 12px;
            top: 50%;
            transform: translateY(-50%);
            color: var(--text-muted);
            font-size: 14px;
        }}
        .search-results {{
            font-size: 11px;
            color: var(--text-muted);
            margin-top: 8px;
            display: none;
        }}
        .search-results.active {{
            display: block;
        }}

        /* Stats grid */
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 8px;
            margin-bottom: 16px;
        }}
        .stat-card {{
            background: var(--bg-tertiary);
            border-radius: 8px;
            padding: 12px 8px;
            text-align: center;
            border: 1px solid transparent;
            transition: border-color 0.2s;
        }}
        .stat-card:hover {{
            border-color: var(--border-color);
        }}
        .stat-value {{
            font-size: 20px;
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
            margin-top: 4px;
        }}

        /* Section divider */
        .section-title {{
            font-size: 10px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin: 12px 0 8px 0;
        }}

        /* Legend items */
        .legend-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 6px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 6px 8px;
            background: var(--bg-tertiary);
            border-radius: 6px;
            font-size: 11px;
            color: var(--text-secondary);
            cursor: pointer;
            transition: background 0.2s;
            border: 1px solid transparent;
        }}
        .legend-item:hover {{
            background: var(--bg-primary);
            border-color: var(--border-color);
        }}
        .legend-item.active {{
            border-color: var(--accent-indigo);
            background: rgba(99, 102, 241, 0.1);
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

        /* Controls section */
        .controls-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 4px;
            margin-top: 8px;
        }}
        .control-item {{
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 10px;
            color: var(--text-muted);
            padding: 4px 0;
        }}
        .kbd {{
            background: var(--bg-primary);
            border: 1px solid var(--border-color);
            padding: 2px 5px;
            border-radius: 4px;
            font-size: 9px;
            color: var(--text-secondary);
            font-family: monospace;
        }}

        /* Loading overlay */
        .loading-overlay {{
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
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
        .loading-subtext {{
            margin-top: 4px;
            color: var(--text-muted);
            font-size: 11px;
        }}
        @keyframes spin {{
            to {{ transform: rotate(360deg); }}
        }}

        /* Node info panel */
        .node-info {{
            position: fixed;
            bottom: 20px;
            left: 20px;
            right: 20px;
            max-width: 600px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 16px 20px;
            z-index: 1000;
            display: none;
            box-shadow: 0 4px 24px rgba(0, 0, 0, 0.5);
        }}
        .node-info.active {{
            display: block;
        }}
        .node-info-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 12px;
        }}
        .node-info-title {{
            font-size: 15px;
            font-weight: 600;
            color: var(--text-primary);
        }}
        .node-info-badge {{
            font-size: 10px;
            padding: 3px 8px;
            border-radius: 4px;
            text-transform: uppercase;
            font-weight: 600;
        }}
        .node-info-badge.entity {{
            background: rgba(99, 102, 241, 0.15);
            color: var(--accent-indigo);
        }}
        .node-info-badge.chunk {{
            background: rgba(236, 72, 153, 0.15);
            color: var(--accent-pink);
        }}
        .node-info-content {{
            font-size: 12px;
            color: var(--text-secondary);
            line-height: 1.6;
            max-height: 120px;
            overflow-y: auto;
        }}
        .node-info-close {{
            position: absolute;
            top: 12px;
            right: 12px;
            background: none;
            border: none;
            color: var(--text-muted);
            cursor: pointer;
            font-size: 16px;
            padding: 4px;
            line-height: 1;
        }}
        .node-info-close:hover {{
            color: var(--text-primary);
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
    </style>

    <!-- Loading overlay -->
    <div class="loading-overlay" id="loadingOverlay">
        <div class="loading-spinner"></div>
        <div class="loading-text">Initializing Knowledge Graph</div>
        <div class="loading-subtext">{graph.vcount():,} nodes, {graph.ecount():,} edges</div>
    </div>

    <!-- Main control panel -->
    <div class="kg-panel">
        <div class="kg-header">
            <div class="kg-title">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="3"/>
                    <circle cx="19" cy="5" r="2"/>
                    <circle cx="5" cy="19" r="2"/>
                    <circle cx="19" cy="19" r="2"/>
                    <circle cx="5" cy="5" r="2"/>
                    <line x1="12" y1="9" x2="12" y2="3"/>
                    <line x1="12" y1="15" x2="12" y2="21"/>
                    <line x1="9" y1="12" x2="3" y2="12"/>
                    <line x1="15" y1="12" x2="21" y2="12"/>
                </svg>
                HippoRAG Knowledge Graph
            </div>
            <div class="kg-subtitle">Interactive visualization explorer</div>
        </div>
        <div class="kg-body">
            <!-- Search -->
            <div class="search-container">
                <span class="search-icon">&#128269;</span>
                <input type="text" class="search-input" id="searchInput" placeholder="Search nodes...">
                <div class="search-results" id="searchResults"></div>
            </div>

            <!-- Stats -->
            <div class="stats-grid">
                <div class="stat-card entities">
                    <div class="stat-value">{entity_count:,}</div>
                    <div class="stat-label">Entities</div>
                </div>
                <div class="stat-card passages">
                    <div class="stat-value">{chunk_count:,}</div>
                    <div class="stat-label">Passages</div>
                </div>
                <div class="stat-card edges">
                    <div class="stat-value">{graph.ecount():,}</div>
                    <div class="stat-label">Edges</div>
                </div>
            </div>

            <!-- Legend -->
            <div class="section-title">Node Types</div>
            <div class="legend-grid">
                <div class="legend-item active" data-type="entity" onclick="toggleFilter('entity')">
                    <div class="legend-dot" style="background: #6366f1;"></div>
                    <span>Entities</span>
                </div>
                <div class="legend-item active" data-type="chunk" onclick="toggleFilter('chunk')">
                    <div class="legend-box" style="background: #ec4899;"></div>
                    <span>Passages</span>
                </div>
            </div>

            <!-- Controls -->
            <div class="section-title">Controls</div>
            <div class="controls-grid">
                <div class="control-item"><span class="kbd">Scroll</span> Zoom</div>
                <div class="control-item"><span class="kbd">Drag</span> Pan</div>
                <div class="control-item"><span class="kbd">Click</span> Select</div>
                <div class="control-item"><span class="kbd">DblClick</span> Focus</div>
            </div>
        </div>
    </div>

    <!-- Zoom controls -->
    <div class="zoom-controls">
        <button class="zoom-btn" onclick="zoomIn()" title="Zoom In">+</button>
        <button class="zoom-btn" onclick="zoomOut()" title="Zoom Out">-</button>
        <button class="zoom-btn" onclick="fitGraph()" title="Fit All">&#8689;</button>
    </div>

    <!-- Node info panel -->
    <div class="node-info" id="nodeInfo">
        <button class="node-info-close" onclick="closeNodeInfo()">&times;</button>
        <div class="node-info-header">
            <div class="node-info-title" id="nodeInfoTitle"></div>
            <div class="node-info-badge" id="nodeInfoBadge"></div>
        </div>
        <div class="node-info-content" id="nodeInfoContent"></div>
    </div>
    '''

    # JavaScript for full interactivity
    stabilization_script = '''
    <script>
    // Store node data for search
    var nodeData = {};
    var activeFilters = { entity: true, chunk: true };

    function setupStableNetwork() {
        if (typeof network !== 'undefined' && typeof nodes !== 'undefined') {
            // Store node data for search
            nodes.forEach(function(node) {
                nodeData[node.id] = {
                    label: node.label || '',
                    title: node.title || '',
                    type: (node.title && node.title.includes('ENTITY')) ? 'entity' : 'chunk'
                };
            });

            // Hide loading overlay after stabilization
            network.on('stabilizationIterationsDone', function() {
                network.setOptions({ physics: { enabled: false } });
                document.getElementById('loadingOverlay').classList.add('hidden');
            });
            network.on('stabilized', function() {
                network.setOptions({ physics: { enabled: false } });
                document.getElementById('loadingOverlay').classList.add('hidden');
            });

            // Show node info on click
            network.on('click', function(params) {
                if (params.nodes.length > 0) {
                    var nodeId = params.nodes[0];
                    var node = nodes.get(nodeId);
                    if (node) {
                        showNodeInfo(node);
                    }
                } else {
                    closeNodeInfo();
                }
            });

            // Double-click to focus on node
            network.on('doubleClick', function(params) {
                if (params.nodes.length > 0) {
                    network.focus(params.nodes[0], {
                        scale: 1.5,
                        animation: { duration: 500, easingFunction: 'easeInOutQuad' }
                    });
                } else {
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
            setTimeout(setupStableNetwork, 100);
        }
    }

    // Show node info panel
    function showNodeInfo(node) {
        var panel = document.getElementById('nodeInfo');
        var titleEl = document.getElementById('nodeInfoTitle');
        var badgeEl = document.getElementById('nodeInfoBadge');
        var contentEl = document.getElementById('nodeInfoContent');

        var isEntity = node.title && node.title.includes('ENTITY');
        titleEl.textContent = node.label || 'Unknown';
        badgeEl.textContent = isEntity ? 'Entity' : 'Passage';
        badgeEl.className = 'node-info-badge ' + (isEntity ? 'entity' : 'chunk');

        // Extract content from title HTML
        var content = '';
        if (node.title) {
            var match = node.title.match(/padding-top: 8px;">([^<]+)/);
            if (match) {
                content = match[1];
            }
        }
        contentEl.textContent = content || 'No additional content available.';

        panel.classList.add('active');
    }

    // Close node info panel
    function closeNodeInfo() {
        document.getElementById('nodeInfo').classList.remove('active');
    }

    // Zoom controls
    function zoomIn() {
        if (typeof network !== 'undefined') {
            var scale = network.getScale() * 1.3;
            network.moveTo({ scale: scale, animation: { duration: 300 } });
        }
    }
    function zoomOut() {
        if (typeof network !== 'undefined') {
            var scale = network.getScale() / 1.3;
            network.moveTo({ scale: scale, animation: { duration: 300 } });
        }
    }
    function fitGraph() {
        if (typeof network !== 'undefined') {
            network.fit({ animation: { duration: 500, easingFunction: 'easeInOutQuad' } });
        }
    }

    // Search functionality
    var searchTimeout;
    document.addEventListener('DOMContentLoaded', function() {
        var searchInput = document.getElementById('searchInput');
        var searchResults = document.getElementById('searchResults');

        if (searchInput) {
            searchInput.addEventListener('input', function(e) {
                clearTimeout(searchTimeout);
                var query = e.target.value.toLowerCase().trim();

                if (query.length < 2) {
                    searchResults.classList.remove('active');
                    searchResults.textContent = '';
                    // Reset all nodes
                    if (typeof nodes !== 'undefined') {
                        nodes.forEach(function(node) {
                            nodes.update({ id: node.id, opacity: 1 });
                        });
                    }
                    return;
                }

                searchTimeout = setTimeout(function() {
                    var matches = [];
                    if (typeof nodes !== 'undefined') {
                        nodes.forEach(function(node) {
                            var label = (node.label || '').toLowerCase();
                            if (label.includes(query)) {
                                matches.push(node.id);
                                nodes.update({ id: node.id, opacity: 1 });
                            } else {
                                nodes.update({ id: node.id, opacity: 0.1 });
                            }
                        });
                    }

                    searchResults.classList.add('active');
                    if (matches.length > 0) {
                        searchResults.textContent = 'Found ' + matches.length + ' matching nodes';
                        // Focus on first match
                        if (typeof network !== 'undefined' && matches.length <= 50) {
                            network.fit({ nodes: matches, animation: { duration: 500 } });
                        }
                    } else {
                        searchResults.textContent = 'No matches found';
                    }
                }, 300);
            });

            // Clear search on Escape
            searchInput.addEventListener('keydown', function(e) {
                if (e.key === 'Escape') {
                    searchInput.value = '';
                    searchInput.dispatchEvent(new Event('input'));
                }
            });
        }
    });

    // Toggle filter (placeholder for future enhancement)
    function toggleFilter(type) {
        var item = document.querySelector('.legend-item[data-type="' + type + '"]');
        if (item) {
            item.classList.toggle('active');
            activeFilters[type] = item.classList.contains('active');
        }
    }

    // Initialize
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', setupStableNetwork);
    } else {
        setTimeout(setupStableNetwork, 500);
    }
    </script>
    '''

    # Insert legend after body tag
    html_content = html_content.replace('<body>', f'<body>{legend_html}')
    # Insert stabilization script before closing body
    html_content = html_content.replace('</body>', f'{stabilization_script}</body>')

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return output_file

def create_triples_visualization(openie_results, output_file='outputs/triples_graph.html'):
    """Create a separate visualization for extracted triples."""
    try:
        from pyvis.network import Network
    except ImportError:
        import subprocess
        subprocess.check_call(['pip', 'install', 'pyvis'])
        from pyvis.network import Network

    if not openie_results:
        return None

    docs_list = openie_results.get('docs', []) if isinstance(openie_results, dict) else openie_results

    net = Network(
        height='100vh',
        width='100vw',
        bgcolor='#0a0e14',
        font_color='#e6e6e6',
        directed=True,
        notebook=False,
        cdn_resources='in_line'  # Embed all JS/CSS to avoid 404 errors
    )

    net.set_options('''
    {
        "nodes": {
            "font": {
                "size": 12,
                "face": "system-ui, -apple-system, BlinkMacSystemFont, sans-serif",
                "color": "#ffffff",
                "strokeWidth": 2,
                "strokeColor": "#0a0e14"
            },
            "borderWidth": 1,
            "shadow": {
                "enabled": true,
                "color": "rgba(0,0,0,0.3)",
                "size": 8
            }
        },
        "edges": {
            "font": {"size": 10, "face": "system-ui", "color": "#9ca3af", "strokeWidth": 0},
            "smooth": {"type": "curvedCW", "roundness": 0.15},
            "arrows": {"to": {"enabled": true, "scaleFactor": 0.4}},
            "color": {"color": "rgba(100,100,120,0.5)", "highlight": "#6366f1"}
        },
        "physics": {
            "enabled": true,
            "barnesHut": {
                "gravitationalConstant": -5000,
                "centralGravity": 0.2,
                "springLength": 100,
                "springConstant": 0.03,
                "damping": 0.5
            },
            "solver": "barnesHut",
            "stabilization": {
                "enabled": true,
                "iterations": 200,
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
            "hideEdgesOnZoom": true
        }
    }
    ''')

    # Track unique entities
    entities = {}
    entity_id = 0

    # Process all documents
    for doc in docs_list:
        triples = doc.get('extracted_triples', [])

        for triple in triples:
            if isinstance(triple, (list, tuple)) and len(triple) >= 3:
                subj, rel, obj = triple[0], triple[1], triple[2]
            elif isinstance(triple, dict):
                subj = triple.get('subject', triple.get('head', ''))
                rel = triple.get('relation', triple.get('predicate', ''))
                obj = triple.get('object', triple.get('tail', ''))
            else:
                continue

            # Skip empty values
            if not subj or not obj:
                continue

            # Add subject node
            if subj not in entities:
                entities[subj] = entity_id
                net.add_node(
                    entity_id,
                    label=subj[:22] + '...' if len(subj) > 22 else subj,
                    title=f'''<div style="font-family: system-ui; padding: 8px;">
                        <div style="font-weight: 600; color: #1e293b;">{subj}</div>
                        <div style="font-size: 11px; color: #6366f1; margin-top: 4px;">Subject Entity</div>
                    </div>''',
                    color={'background': '#6366f1', 'border': '#4f46e5', 'highlight': {'background': '#818cf8', 'border': '#6366f1'}},
                    size=18
                )
                entity_id += 1

            # Add object node
            if obj not in entities:
                entities[obj] = entity_id
                net.add_node(
                    entity_id,
                    label=obj[:22] + '...' if len(obj) > 22 else obj,
                    title=f'''<div style="font-family: system-ui; padding: 8px;">
                        <div style="font-weight: 600; color: #1e293b;">{obj}</div>
                        <div style="font-size: 11px; color: #10b981; margin-top: 4px;">Object Entity</div>
                    </div>''',
                    color={'background': '#10b981', 'border': '#059669', 'highlight': {'background': '#34d399', 'border': '#10b981'}},
                    size=18
                )
                entity_id += 1

            # Add edge with relation label
            net.add_edge(
                entities[subj],
                entities[obj],
                label=rel[:18] + '...' if len(rel) > 18 else rel,
                title=f'''<div style="font-family: system-ui; padding: 6px; font-size: 11px;">
                    <div style="color: #64748b;">Relation</div>
                    <div style="font-weight: 500; color: #1e293b;">{rel}</div>
                </div>''',
                width=1.5
            )

    # Save
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    net.save_graph(output_file)

    # Count entities for stats
    triple_count = sum(len(doc.get('extracted_triples', [])) for doc in docs_list)

    # Add legend
    with open(output_file, 'r', encoding='utf-8') as f:
        html_content = f.read()

    legend_html = f'''
    <style>
        :root {{
            --bg-primary: #0a0e14;
            --bg-secondary: #111827;
            --bg-tertiary: #1f2937;
            --border-color: #374151;
            --text-primary: #f9fafb;
            --text-secondary: #9ca3af;
            --text-muted: #6b7280;
        }}
        * {{ font-family: system-ui, -apple-system, sans-serif; box-sizing: border-box; margin: 0; padding: 0; }}
        html, body {{ width: 100vw; height: 100vh; overflow: hidden; background: var(--bg-primary); }}
        .card {{ width: 100vw !important; height: 100vh !important; margin: 0 !important; padding: 0 !important; border: none !important; background: transparent !important; }}
        #mynetwork {{ width: 100vw !important; height: 100vh !important; position: fixed !important; top: 0 !important; left: 0 !important; border: none !important; background: var(--bg-primary) !important; }}
        #loadingBar {{ display: none !important; }}

        .triples-panel {{
            position: fixed;
            top: 20px;
            left: 20px;
            width: 280px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            color: var(--text-primary);
            z-index: 1000;
            box-shadow: 0 4px 24px rgba(0, 0, 0, 0.5);
            overflow: hidden;
        }}
        .triples-header {{
            padding: 16px 20px;
            border-bottom: 1px solid var(--border-color);
            background: linear-gradient(135deg, var(--bg-tertiary), var(--bg-secondary));
        }}
        .triples-title {{
            font-size: 15px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .triples-subtitle {{
            font-size: 11px;
            color: var(--text-muted);
            margin-top: 4px;
        }}
        .triples-body {{
            padding: 16px 20px;
        }}
        .stats-row {{
            display: flex;
            gap: 8px;
            margin-bottom: 16px;
        }}
        .stat-item {{
            flex: 1;
            background: var(--bg-tertiary);
            border-radius: 8px;
            padding: 10px 8px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 18px;
            font-weight: 700;
            color: #6366f1;
        }}
        .stat-label {{
            font-size: 9px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-top: 2px;
        }}
        .section-title {{
            font-size: 10px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin: 12px 0 8px 0;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px;
            background: var(--bg-tertiary);
            border-radius: 6px;
            font-size: 12px;
            color: var(--text-secondary);
            margin-bottom: 6px;
        }}
        .legend-dot {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            flex-shrink: 0;
        }}
        .legend-line {{
            width: 20px;
            height: 2px;
            flex-shrink: 0;
            background: #64748b;
        }}
    </style>

    <div class="triples-panel">
        <div class="triples-header">
            <div class="triples-title">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M12 2L2 7l10 5 10-5-10-5z"/>
                    <path d="M2 17l10 5 10-5"/>
                    <path d="M2 12l10 5 10-5"/>
                </svg>
                Knowledge Triples
            </div>
            <div class="triples-subtitle">Subject - Relation - Object</div>
        </div>
        <div class="triples-body">
            <div class="stats-row">
                <div class="stat-item">
                    <div class="stat-value">{len(entities):,}</div>
                    <div class="stat-label">Entities</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" style="color: #10b981;">{triple_count:,}</div>
                    <div class="stat-label">Triples</div>
                </div>
            </div>

            <div class="section-title">Legend</div>
            <div class="legend-item">
                <div class="legend-dot" style="background: #6366f1;"></div>
                <span>Subject Entities</span>
            </div>
            <div class="legend-item">
                <div class="legend-dot" style="background: #10b981;"></div>
                <span>Object Entities</span>
            </div>
            <div class="legend-item">
                <div class="legend-line"></div>
                <span>Relations (labeled edges)</span>
            </div>
        </div>
    </div>
    '''

    # JavaScript to disable physics after stabilization
    triples_stabilization_script = '''
    <script>
    function setupStableNetwork() {
        if (typeof network !== 'undefined') {
            network.on('stabilizationIterationsDone', function() {
                network.setOptions({ physics: { enabled: false } });
            });
            network.on('stabilized', function() {
                network.setOptions({ physics: { enabled: false } });
            });
            network.on('doubleClick', function(params) {
                if (params.nodes.length === 0) {
                    network.setOptions({ physics: { enabled: true } });
                    setTimeout(function() {
                        network.setOptions({ physics: { enabled: false } });
                    }, 3000);
                }
            });
        } else {
            setTimeout(setupStableNetwork, 100);
        }
    }
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', setupStableNetwork);
    } else {
        setTimeout(setupStableNetwork, 500);
    }
    </script>
    '''

    html_content = html_content.replace('<body>', f'<body>{legend_html}')
    html_content = html_content.replace('</body>', f'{triples_stabilization_script}</body>')

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return output_file

def start_server_and_open(html_file, port=8888):
    """Start a simple HTTP server and open the visualization in browser."""
    import functools

    # Get absolute path
    abs_html_file = os.path.abspath(html_file)
    html_dir = os.path.dirname(abs_html_file)
    html_filename = os.path.basename(abs_html_file)

    print(f"\nServing from directory: {html_dir}")
    print(f"HTML file: {html_filename}")

    # Create a custom handler that serves from the correct directory
    handler = functools.partial(SimpleHTTPRequestHandler, directory=html_dir)

    # Create server
    httpd = HTTPServer(('localhost', port), handler)

    # Start server in background thread
    server_thread = threading.Thread(target=httpd.serve_forever)
    server_thread.daemon = True
    server_thread.start()

    url = f'http://localhost:{port}/{html_filename}'
    print(f"\nServer started at: {url}")
    print("Press Ctrl+C to stop the server\n")

    # Open in browser
    webbrowser.open(url)

    return httpd

def main():
    print("="*60)
    print("  HippoRAG Knowledge Graph - Web Visualization")
    print("="*60)

    # Load graph
    print("\nLoading knowledge graph...")
    graph = load_graph()

    if graph is None:
        print("\nNo graph found. Please run test_hipporag.py first.")
        return

    print(f"Graph loaded: {graph.vcount()} nodes, {graph.ecount()} edges")

    # Load OpenIE results
    print("\nLoading OpenIE results...")
    openie_results = load_openie_results()

    # Create main graph visualization
    print("\nCreating interactive knowledge graph...")
    kg_file = create_pyvis_visualization(graph, 'outputs/knowledge_graph_interactive.html')
    print(f"Created: {kg_file}")

    # Create triples visualization
    if openie_results:
        print("\nCreating triples visualization...")
        triples_file = create_triples_visualization(openie_results, 'outputs/triples_graph.html')
        print(f"Created: {triples_file}")

    # Start server and open in browser
    print("\n" + "="*60)
    print("  Starting web server...")
    print("="*60)

    try:
        httpd = start_server_and_open(kg_file, port=8888)

        print("\nVisualization files created:")
        print(f"  1. Knowledge Graph: outputs/knowledge_graph_interactive.html")
        if openie_results:
            print(f"  2. Triples Graph:   outputs/triples_graph.html")

        print(f"\nOpen in browser:")
        print(f"  - http://localhost:8888/knowledge_graph_interactive.html")
        if openie_results:
            print(f"  - http://localhost:8888/triples_graph.html")

        print("\nPress Ctrl+C to stop the server...")

        # Keep server running
        while True:
            pass

    except KeyboardInterrupt:
        print("\n\nServer stopped.")
    except Exception as e:
        print(f"\nError starting server: {e}")
        print(f"\nYou can still open the HTML files directly:")
        print(f"  - {os.path.abspath(kg_file)}")

if __name__ == "__main__":
    main()
