import re
import math
import urllib.parse as _url
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import networkx as nx
import pandas as pd
from dash import html, dcc, Input, Output, State, MATCH

from indra_variants.app.config import DATA_DIR, PORT, DEBUG

cyto.load_extra_layouts()

TSV_RE = re.compile(r"^(?P<prot>.+)_variant_effects_with_clinvar_with_domains\.tsv$", re.I)
PROTS = sorted(TSV_RE.match(p.name).group("prot")
               for p in Path(DATA_DIR).iterdir()
               if TSV_RE.match(p.name))

def format_star_rating(star_val):
    review_map = {
        4.0: "practice guideline",
        3.0: "expert panel",
        2.0: "multiple submitters, no conflicts",
        1.0: "single submitter",
        0.0: "no assertion criteria provided" # Updated for clarity
    }
    try:
        s_val = float(star_val)
        num_stars = int(s_val)
        review_text = review_map.get(s_val, "review status not specified")
        
        if num_stars > 0:
            return f"{'★ ' * num_stars} ({review_text})"
        else:
            return f"({review_text})"
            
    except (ValueError, TypeError):
        return "(no review info)"
    
# ----------------------Build Graph--------------------------–
def build_elements(prot: str):
    df_path = Path(DATA_DIR) / f"{prot}_variant_effects_with_clinvar_with_domains.tsv"
    df = pd.read_csv(df_path, sep="\t").fillna('')

    G = nx.MultiDiGraph()
    G.add_node(prot)
    
    all_domains = set()
    for features_str in df['DomainFeature']:
        if features_str:
            domains = [d.strip() for d in features_str.split(";")]
            all_domains.update(domains)

    all_domains.discard('')
    all_domains.discard('CHAIN')
    domain_nodes = all_domains

    for domain in domain_nodes:
        G.add_node(domain)
        G.add_edge(prot, domain, relation="has_domain")

    for _, row in df.iterrows():
        var = row["variant_info"]

        all_conditions = []
        for i in range(1, 11):
            disease = row.get(f"disease_{i}", "")
            # Check if disease field is meaningful
            if disease and 'not provided' not in disease.lower():
                all_conditions.append(disease)
 
        clinvar_data = None
        if all_conditions:
            clinvar_data = {
                "pathogenicity": row.get("significance_1", "N/A"),
                "review": format_star_rating(row.get("star_1", 0.0)),
                "conditions": "; ".join(all_conditions)
            }
        features_str = row["DomainFeature"]
        notes_str = row["DomainNote"]

        features = [f.strip() for f in features_str.split(';')] if features_str else []
        notes = [n.strip() for n in notes_str.split(';')] if notes_str else []

        has_specific_domain = False

        for domain, note in zip(features, notes):
            if domain and domain != 'CHAIN':
                G.add_edge(domain, var, relation="DV", note=note, clinvar_data=clinvar_data)
                has_specific_domain = True

        if not has_specific_domain:
            G.add_edge(prot, var, relation="PV", note="No specific domain annotated", pmid="", clinvar_data=clinvar_data)

        src = var
        for seg in row["chain"].split(" -[")[1:]:
            if "]->" not in seg:
                continue
            rel, tgt = seg.split("]->")
            G.add_edge(src.strip(), tgt.strip(),
                       relation=rel.strip(), pmid=row["pmid"])
            src = tgt.strip()

    variants = set(df["variant_info"])
    endpoints = set(df["biological_process/disease"])
    freq = df["biological_process/disease"].value_counts().to_dict()

    def get_layer(n):
        if n == prot:
            return 0
        if n in domain_nodes:
            return 1
        if n in variants:
            return 2
        if n in endpoints:
            return 4
        return 3

    # pos = {prot: (0, 0)}
    # for L in range(1, 5):
    #     ring = [n for n in G if get_layer(n) == L]
    #     if not ring:
    #         continue
    #     for i, n in enumerate(ring):
    #         r = 280 * (L + 2)
    #         ang = 2 * math.pi * i / len(ring)
    #         pos[n] = (r * math.cos(ang), r * math.sin(ang))
    
    node_sizes = {}
    for n in G.nodes():
        layer = get_layer(n)
        size = 50.0
        if layer == 0: size = 80.0
        elif layer == 1: size = 60.0
        elif layer == 4: size = 50.0 + 10.0 * freq.get(n, 1)
        node_sizes[n] = max(size, 40.0)

    # parmaeters
    MIN_RADII = { 1: 300, 2: 600, 3: 1000, 4: 1500 }
    MAX_RADII = { 1: 1200, 2: 1800, 3: 3000, 4: 5000 }
    padding = 50
    TOLERANCE_PER_LAYER = { 1: 0.0, 2: 0.2, 3: 0.4, 4: 0.5 }
    ANGULAR_PADDING_FACTOR = 0.1 
    CROWDING_THRESHOLD = math.radians(5) # each weight unit should get at least this many radians

    node_placements = {}; parent_map = {prot: None}
    for u, v in nx.bfs_edges(G, source=prot):
        if v not in parent_map: parent_map[v] = u
    children_map = {n: [] for n in G.nodes()}
    for child, parent in parent_map.items():
        if parent is not None: children_map[parent].append(child)
    memo_descendants = {}
    def count_descendants(node):
        if node in memo_descendants: return memo_descendants[node]
        count = len(children_map.get(node, []));
        for child in children_map.get(node, []): count += count_descendants(child)
        memo_descendants[node] = count
        return count
    def place_nodes_recursively(node, start_angle, sweep_angle):
        children = sorted(children_map.get(node, []))
        if not children: return
        num_gaps = len(children) -1 
        if sweep_angle < 0.01 or num_gaps <= 0:
            padding_per_gap = 0.0; effective_sweep = sweep_angle
        else:
            total_padding = sweep_angle * ANGULAR_PADDING_FACTOR
            padding_per_gap = total_padding / num_gaps
            effective_sweep = sweep_angle - total_padding
        total_weight = sum(count_descendants(c) + 1 for c in children)
        if total_weight == 0: return
        current_angle = start_angle
        for child in children:
            weight = count_descendants(child) + 1
            child_sweep = (weight / total_weight) * effective_sweep
            child_angle = current_angle + child_sweep / 2
            node_placements[child] = {'layer': get_layer(child), 'angle': child_angle}
            place_nodes_recursively(child, current_angle, child_sweep)
            current_angle += child_sweep + padding_per_gap

    
    # address for domain and no-domain children separately
    domain_roots = []; no_domain_roots = []
    for child in children_map.get(prot, []):
        if get_layer(child) == 1: domain_roots.append(child)
        else: no_domain_roots.append(child)
    domain_roots.sort(); no_domain_roots.sort()
    
    total_domain_weight = sum(count_descendants(n) + 1 for n in domain_roots)
    total_no_domain_weight = sum(count_descendants(n) + 1 for n in no_domain_roots)

    use_proportional_split = False
    # check if domain side is too crowded
    if total_domain_weight > 0 and (math.pi / total_domain_weight) < CROWDING_THRESHOLD:
        use_proportional_split = True
    # check if no-domain side is too crowded
    if total_no_domain_weight > 0 and (math.pi / total_no_domain_weight) < CROWDING_THRESHOLD:
        use_proportional_split = True

    # use different angle allocation strategies based on crowding
    if use_proportional_split:
        # ues "proportional split" mode (for dense, complex graphs)
        total_graph_weight = total_domain_weight + total_no_domain_weight
        domain_side_sweep = (total_domain_weight / total_graph_weight) * 2 * math.pi if total_graph_weight > 0 else 0
        no_domain_side_sweep = (total_no_domain_weight / total_graph_weight) * 2 * math.pi if total_graph_weight > 0 else 0
        
        domain_start_angle = -math.pi / 2
        no_domain_start_angle = domain_start_angle + domain_side_sweep
    else:
        domain_side_sweep = math.pi
        no_domain_side_sweep = math.pi
        domain_start_angle = -math.pi / 2
        no_domain_start_angle = math.pi / 2

    # calculate placements for domain and no-domain children
    num_domains = len(domain_roots)
    if num_domains > 0:
        if not use_proportional_split and num_domains == 1:
            node_placements[domain_roots[0]] = {'layer': 1, 'angle': 0.0}
            place_nodes_recursively(domain_roots[0], -math.pi / 4, math.pi / 2)
        else:
            angle_per_sector = domain_side_sweep / num_domains
            for i, domain_node in enumerate(domain_roots):
                sector_start = domain_start_angle + i * angle_per_sector
                angle = sector_start + angle_per_sector / 2
                node_placements[domain_node] = {'layer': 1, 'angle': angle}
                place_nodes_recursively(domain_node, sector_start, angle_per_sector)

    num_no_domain = len(no_domain_roots)
    if num_no_domain > 0:
        angle_per_sector = no_domain_side_sweep / num_no_domain
        for i, variant_node in enumerate(no_domain_roots):
            sector_start = no_domain_start_angle + i * angle_per_sector
            angle = sector_start + angle_per_sector / 2
            layer = get_layer(variant_node)
            node_placements[variant_node] = {'layer': layer, 'angle': angle}
            place_nodes_recursively(variant_node, sector_start, angle_per_sector)


    # dynamic radius calculation
    dynamic_radii = {0: 0}
    nodes_by_layer = {L: [] for L in range(1, 5)}
    for node, placement in node_placements.items():
        if placement['layer'] in nodes_by_layer:
            nodes_by_layer[placement['layer']].append(node)

    for L in range(1, 5):
        min_radial_radius = 0
        prev_layer_radius = dynamic_radii.get(L - 1, dynamic_radii.get(0))
        prev_layer_nodes = nodes_by_layer.get(L - 1, [prot] if L==1 else [])
        if prev_layer_nodes:
            max_size_prev = max(node_sizes[n] for n in prev_layer_nodes)
            max_size_curr = max((node_sizes[n] for n in nodes_by_layer[L]), default=0)
            min_radial_radius = prev_layer_radius + max_size_prev / 2 + max_size_curr / 2 + padding

        min_circumferential_radius = 0
        layer_nodes = nodes_by_layer[L]
        if len(layer_nodes) > 1:
            sorted_nodes = sorted(layer_nodes, key=lambda n: node_placements[n]['angle'])
            for i in range(len(sorted_nodes)):
                n1 = sorted_nodes[i]; n2 = sorted_nodes[(i + 1) % len(sorted_nodes)]
                angle1 = node_placements[n1]['angle']; angle2 = node_placements[n2]['angle']
                delta_angle = abs(angle2 - angle1)
                if delta_angle > math.pi: delta_angle = 2 * math.pi - delta_angle
                if delta_angle > 1e-9:
                    
                    # use distance tolerance to relax radius requirement
                    tolerance = TOLERANCE_PER_LAYER.get(L, 0.0) 
                    required_dist = (node_sizes[n1] / 2 + node_sizes[n2] / 2 + padding) * (1 - tolerance)
                    
                    required_radius = required_dist / (2 * math.sin(delta_angle / 2))
                    if required_radius > min_circumferential_radius:
                        min_circumferential_radius = required_radius
        
        ideal_radius = max(min_radial_radius, min_circumferential_radius)
        min_r = MIN_RADII.get(L, ideal_radius)
        max_r = MAX_RADII.get(L, ideal_radius)
        dynamic_radii[L] = max(min_r, min(ideal_radius, max_r))

    # calculate final positions based on dynamic radii
    pos = {prot: (0, 0)}
    for node, placement in node_placements.items():
        layer = placement['layer']
        angle = placement['angle']
        radius = dynamic_radii.get(layer)
        if radius is not None:
            pos[node] = (radius * math.cos(angle), radius * math.sin(angle))
            
    # end here ----------------------------------------

    rel_types = sorted({d['relation'] for _, _, d in G.edges(data=True)
                        if d['relation'] not in {'PV', 'DV', 'has_domain'}})
    palette = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6"]
    rel_color = {r: palette[i % len(palette)] for i, r in enumerate(rel_types)}

    els = []
    for n, (x, y) in pos.items():
        layer = get_layer(n)
        if layer == 0:
            size = 80
        elif layer == 1:
            size = 60
        elif layer == 4:
            size =  50 + 10 * freq.get(n, 1)
        else:
            size = 50
            
        label = n

        node_el = {
            "data": {"id": n, "label": label, "real": n},
            "classes": f"L{layer}",
            "style": {"width": size, "height": size}
        }

        if n in pos:
            node_el["position"] = {"x": pos[n][0], "y": pos[n][1]}
        els.append(node_el)

    for u, v, d in G.edges(data=True):
        relation = d.get('relation', '')
        if relation == 'PV': cls = 'edge-PV'
        elif relation == 'DV': cls = 'edge-DV'
        elif relation == 'has_domain': cls = 'edge-has_domain'
        else: cls = f"edge-{relation}"

        src4indra = prot if u in variants else u
        edge_data={
            "id" :f"{u}->{v}_{d.get('pmid', '')}_{d.get('note', '')}",
            "source": u,
            "target": v,
            "rel": relation,
            "src4indra": src4indra
        }
        if 'pmid' in d and d['pmid']:
            edge_data['pmid'] = d['pmid']
        if 'note' in d and d['note']:
            edge_data['note'] = d['note']
        if 'clinvar_data' in d and d['clinvar_data']:
            edge_data['clinvar_data'] = d['clinvar_data']

        els.append({
            "data": edge_data,
            "classes": cls
        })

    edge_set = {(u, v) for u, v in G.edges()}
    legend_rels = ['Gene to Domain', 'Domain to Variant', 'Gene to Variant'] + rel_types
    legend_colors = {
        'Gene to Domain': '#7f8c8d',
        'Domain to Variant': '#2c3e50',
        'Gene to Variant': '#d5cbc9',
        **rel_color
    }

    return els, legend_rels, legend_colors, list(edge_set)


# ------------------------Dash App------------------------–
app = dash.Dash(__name__,
                suppress_callback_exceptions=True,
                external_stylesheets=[dbc.themes.FLATLY])
# Set the server for deployment, see https://dash.plotly.com/deployment
server = app.server
app.layout = html.Div([dcc.Location(id="url"), html.Div(id="page")])


# ---Homepage---
def homepage():
    prot_options = [{'label': p, 'value': p} for p in PROTS]

    search_card = html.Div(
        [
            html.H1("Protein Variant Network Explorer",
                    style={'marginTop': 0, 'marginBottom': 12}),
            html.P("Type a protein/gene name below (auto-suggest enabled) "
                   "or browse alphabetically.",
                   style={'fontSize': 18, 'margin': '0 0 20px 0'}),
            dcc.Dropdown(id='prot-search', options=prot_options,
                         placeholder="search protein / gene …",
                         style={'fontSize': 18}, clearable=True,
                         searchable=True),
            dbc.Button("Search", id='submit-prot', n_clicks=0,
                       color="primary", style={'marginTop': 18,
                                               'fontSize': 18}),
        ],
        style={'maxWidth': 880, 'margin': '40px auto',
               'background': '#f8f9fa', 'padding': '32px 48px',
               'borderRadius': 8, 'boxShadow': '0 0 8px rgba(0,0,0,0.15)',
               'fontFamily': 'Arial, sans-serif'}
    )

    directory = html.Div(id='prot-directory',
                         style={'maxWidth': 880, 'margin': '0 auto',
                                'fontFamily': 'Arial, sans-serif'})

    footer = html.Div(
        [
            html.Span("Developed by the "),
            html.A("Gyori Lab", href="https://gyorilab.github.io",
                   target="_blank"),
            html.Span(" at Northeastern University"),
            html.Br(),
            html.Span("INDRA Variant is funded under DARPA ASKEM / "
                      "ARPA-H BDF (HR00112220036)")
        ],
        style={'background': '#f1f1f1', 'padding': '10px 24px',
               'textAlign': 'center', 'fontSize': 14,
               'fontFamily': 'Arial, sans-serif', 'marginTop': 40}
    )

    return html.Div([search_card, directory, footer])


# ---Network Page---
def network_page(prot: str):
    els, legend_rels, legend_colors, edge_set = build_elements(prot)

    def rel_style(r, c):
        return {'selector': f'.edge-{r}',
                'style': {'line-color': c, 'target-arrow-color': c,
                          'target-arrow-shape': 'triangle',
                          'curve-style': 'bezier', 'width': 2}}

    sidebar = html.Div(
        id={'type': 'edge-info', 'prot': prot},
        children=[
            html.Div("Domain & Variant Information", 
                    style={'fontSize': 18, 'fontWeight': 'bold', 
                           'marginBottom': 15, 'color': '#2c3e50',
                           'borderBottom': '2px solid #ecf0f1',
                           'paddingBottom': 10}),
            html.Div("Click on an edge to see detailed information about domains, variants, and clinical data.",
                    style={'color': '#7f8c8d', 'fontSize': 14, 'lineHeight': '1.4'})
        ],
        style={
            'position': 'fixed',
            'left': 0,
            'top': 0,
            'width': 350,
            'height': '100vh',
            'background': 'linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)',
            'padding': 20,
            'boxShadow': '2px 0 10px rgba(0,0,0,0.1)',
            'borderRight': '1px solid #dee2e6',
            'fontSize': 16,
            'fontFamily': 'Arial, sans-serif',
            'zIndex': 1000,
            'overflowY': 'auto'
        }
    )

    main_content = html.Div([
        html.Div([
            dcc.Link("← Home", href="/", 
                    style={'color': '#0366d6', 'textDecoration': 'none', 
                           'fontSize': 16, 'fontWeight': 'bold'}),
            html.H3(f"{prot} Variant Network", 
                   style={'textAlign': 'center', 'margin': '10px 0',
                          'color': '#2c3e50'}),
            html.P("Tip: click the central protein/gene to clear all highlights.",
                   style={'textAlign': 'center', 'marginTop': 0,
                          'marginBottom': 15, 'color': '#666',
                          'fontFamily': 'Arial, sans-serif', 'fontSize': 14})
        ], style={'padding': '10px 20px', 'background': '#ffffff',
                  'borderBottom': '1px solid #dee2e6'}),

        dcc.Store(id={'type': 'store-els',  'prot': prot},  data=els),
        dcc.Store(id={'type': 'store-edges', 'prot': prot},  data=edge_set),
        dcc.Store(id={'type': 'store-root', 'prot': prot},  data=prot),

        cyto.Cytoscape(
            id={'type': 'cy-net', 'prot': prot},
            elements=els, 
            layout={'name': 'preset'},
            style={'width': '100%', 'height': 'calc(100vh - 120px)'},
            stylesheet=[
                {'selector': 'node', 'style': {
                    'shape': 'ellipse', 'background-opacity': 0.5,
                    'font-size': 38, 'font-weight': 'bold',
                    'label': 'data(label)',
                    'text-valign': 'center',
                    'text-halign': 'center'}},
                # L0: root protein
                {'selector': '.L0',
                 'style': {'background-color': '#aacdd7',
                           'color': '#004466',
                           'label': 'data(real)'}},
                # L1: domain
                {'selector': '.L1', 
                 'style': {'background-color':"#8bb6b3",
                           'color': '#125652'}},
                # L2: variant
                {'selector': '.L2',
                 'style': {'background-color': '#a492bb',
                           'color': '#573d82'}},
                # L3: intermediate node 
                {'selector': '.L3',
                 'style': {'background-color': '#cce9b6',
                           'color': '#3f6330'}},
                # L4: endpoint
                {'selector': '.L4',
                 'style': {'background-color': '#fabf77',
                           'color': '#b05e04',
                           'label': 'data(real)'}},
                # Edges
                {'selector': '.edge-PV', 
                 'style': {'line-color': '#d5cbc9', 
                           'target-arrow-shape': 'triangle', 
                           'width': 2}},
                # Gene -> Domain
                {'selector': '.edge-has_domain', 
                 'style': {'line-color': '#7f8c8d', 
                           'target-arrow-shape': 'none', 
                           'width': 3, 
                           'line-style': 'dotted'}},
                # Domain -> Variant
                {'selector': '.edge-DV', 
                 'style': {'line-color': '#2c3e50', 
                           'target-arrow-shape': 'triangle', 
                           'width': 1.5}},
                
                *[rel_style(r, c) for r, c in legend_colors.items() if r not in ['Gene to Domain', 'Domain to Variant', 'Gene to Variant']],
                {'selector': '.faded', 'style': {'opacity': 0.15}}
            ]),

        html.Div([
            html.H4("Legend",
                    style={'margin': 0, 'fontSize': 16,
                           'fontWeight': 'bold',
                           'fontFamily': 'Arial, sans-serif',
                           'color': '#2c3e50'}),
            html.Ul([
                html.Li([html.Span('→',
                                   style={'color': legend_colors.get(r, '#d5cbc9'),
                                          'marginRight': 8,
                                          'fontSize': 16}), r],
                        style={'fontSize': 14, 'listStyle': 'none',
                               'margin': '2px 0'})
                for r in legend_rels
            ], style={'paddingLeft': 0, 'margin': '8px 0 0 0'})
        ], style={'position': 'absolute', 'top': 20, 'right': 20,
                  'background': 'rgba(255,255,255,0.95)',
                  'padding': '12px 16px',
                  'borderRadius': 8,
                  'boxShadow': '0 2px 8px rgba(0,0,0,0.15)',
                  'fontFamily': 'Arial, sans-serif',
                  'maxHeight': '70vh',
                  'overflowY': 'auto'})
        
    ], style={
        'marginLeft': 350,
        'position': 'relative',
        'height': '100vh'
    })

    return html.Div([sidebar, main_content])


# ----
@app.callback(Output("page", "children"), Input("url", "pathname"))
def router(path):
    if path in (None, "/"):
        return homepage()
    if path.startswith("/protein/"):
        prot = path.split("/")[2]
        if prot in PROTS:
            return network_page(prot)
    return html.H3("404 – Not found")

@app.callback(
    Output({'type': 'edge-info', 'prot': MATCH}, 'children'),
    Input({'type': 'cy-net', 'prot': MATCH}, 'tapEdgeData'),
    prevent_initial_call=True)
def show_edge_info(edge):
    if not edge:
        return [
            html.Div("Domain & Variant Information", 
                    style={'fontSize': 18, 'fontWeight': 'bold', 
                           'marginBottom': 15, 'color': '#2c3e50',
                           'borderBottom': '2px solid #ecf0f1',
                           'paddingBottom': 10}),
            html.Div("Click on an edge to see detailed information about domains, variants, and clinical data.",
                    style={'color': '#7f8c8d', 'fontSize': 14, 'lineHeight': '1.4'})
        ]

    content = []

    content.append(
        html.Div("Edge Information", 
                style={'fontSize': 18, 'fontWeight': 'bold', 
                       'marginBottom': 15, 'color': '#2c3e50',
                       'borderBottom': '2px solid #ecf0f1',
                       'paddingBottom': 10})
    )

    rel = edge.get('rel', 'N/A')
    source = edge.get('source', 'N/A')
    target = edge.get('target', 'N/A')

    content.append(
        html.Div([
            html.Div("Relationship", 
                    style={'fontSize': 14, 'fontWeight': 'bold', 
                        'color': '#34495e', 'marginBottom': 5}),
            html.Div(f"{source} → {target}", 
                    style={'fontSize': 14, 'color': '#2c3e50', 
                        'marginBottom': 8, 'fontWeight': 'bold'}),
            *( [] if rel in ['DV', 'PV', 'has_domain'] else [
                html.Div(f"Type: {rel}", 
                    style={'fontSize': 14, 'color': '#7f8c8d'})
            ])
        ], style={
            'background': '#ffffff',
            'padding': 12,
            'borderRadius': 6,
            'boxShadow': '0 1px 3px rgba(0,0,0,0.1)',
            'marginBottom': 15,
            'border': '1px solid #dee2e6'
        })
    )

    if rel == 'DV':
        if 'note' in edge and edge['note']:
            content.append(
                html.Div([
                    html.Div("Domain Description", 
                            style={'fontSize': 14, 'fontWeight': 'bold', 
                                   'color': '#34495e', 'marginBottom': 8}),
                    html.Div(edge['note'], 
                            style={'fontSize': 14, 'color': '#2c3e50',
                                   'lineHeight': '1.4'})
                ], style={
                    'background': '#ffffff',
                    'padding': 12,
                    'borderRadius': 6,
                    'boxShadow': '0 1px 3px rgba(0,0,0,0.1)',
                    'marginBottom': 15,
                    'border': '1px solid #dee2e6'
                })
            )
    else:
        if 'note' in edge and edge['note']:
            content.append(
                html.Div([
                    html.Div("Description", 
                            style={'fontSize': 14, 'fontWeight': 'bold', 
                                   'color': '#34495e', 'marginBottom': 8}),
                    html.Div(edge['note'], 
                            style={'fontSize': 14, 'color': '#2c3e50',
                                   'lineHeight': '1.4'})
                ], style={
                    'background': '#ffffff',
                    'padding': 12,
                    'borderRadius': 6,
                    'boxShadow': '0 1px 3px rgba(0,0,0,0.1)',
                    'marginBottom': 15,
                    'border': '1px solid #dee2e6'
                })
            )

    if 'clinvar_data' in edge and edge['clinvar_data']:
        data = edge['clinvar_data']
        content.append(
            html.Div([
                html.Div("ClinVar Information", 
                        style={'fontSize': 14, 'fontWeight': 'bold', 
                               'color': '#34495e', 'marginBottom': 10}),
                html.Div([
                    html.Div([
                        html.Span("Pathogenicity: ", style={'fontWeight': 'bold'}),
                        html.Span(data.get('pathogenicity', 'N/A'))
                    ], style={'marginBottom': 6}),
                    html.Div([
                        html.Span("Review Status: ", style={'fontWeight': 'bold'}),
                        html.Span(data.get('review', 'N/A'))
                    ], style={'marginBottom': 6}),
                    html.Div([
                        html.Span("Associated Condition: ", style={'fontWeight': 'bold'}),
                        html.Div(data.get('conditions', 'N/A'),
                                style={'marginTop': 4, 'fontStyle': 'italic'})
                    ])
                ], style={'fontSize': 13, 'color': '#2c3e50', 'lineHeight': '1.4'})
            ], style={
                'background': '#ffffff',
                'padding': 12,
                'borderRadius': 6,
                'boxShadow': '0 1px 3px rgba(0,0,0,0.1)',
                'marginBottom': 15,
                'border': '1px solid #dee2e6'
            })
        )

    if 'pmid' in edge and edge['pmid']:
        pmid = edge['pmid']
        content.append(
            html.Div([
                html.Div("External Resources", 
                        style={'fontSize': 14, 'fontWeight': 'bold', 
                               'color': '#34495e', 'marginBottom': 10}),
                html.Div([
                    html.Div(f"PubMed ID: {pmid}", 
                            style={'fontSize': 13, 'color': '#7f8c8d', 'marginBottom': 8}),
                    html.A("🔗 View in PubMed",
                          href=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                          target="_blank", 
                          style={'display': 'block', 'marginBottom': 8,
                                 'color': '#0366d6', 'textDecoration': 'none',
                                 'fontSize': 13, 'fontWeight': 'bold'}),
                    
                    html.A("🔗 View in INDRA",
                          href=(f"https://discovery.indra.bio/search/"
                               f"?agent={_url.quote_plus(edge['src4indra'])}&other_agent={_url.quote_plus(edge['target'])}"
                               "&agent_role=subject&other_role=object"),
                          target="_blank",
                          style={'display': 'block', 'color': '#0366d6', 
                                 'textDecoration': 'none', 'fontSize': 13,
                                 'fontWeight': 'bold'})
                ])
            ], style={
                'background': '#ffffff',
                'padding': 12,
                'borderRadius': 6,
                'boxShadow': '0 1px 3px rgba(0,0,0,0.1)',
                'marginBottom': 15,
                'border': '1px solid #dee2e6'
            })
        )

    return content

# ---------------------- highlight callback ---------------------------------
@app.callback(
    Output({'type': 'cy-net', 'prot': MATCH}, 'elements'),
    Input({'type': 'cy-net', 'prot': MATCH}, 'tapNodeData'),
    [State({'type': 'store-els',   'prot': MATCH}, 'data'),
     State({'type': 'store-edges', 'prot': MATCH}, 'data'),
     State({'type': 'store-root',  'prot': MATCH}, 'data')],
    prevent_initial_call=True)
def highlight(node, elements, edge_set, root_prot):
    if not node:
        return elements

    if node['id'] == root_prot:
        for el in elements:
            el['classes'] = el['classes'].replace(' faded', '')
        return elements

    edge_set = {tuple(e) for e in edge_set}
    sel = node['id']
    keep_nodes = {sel}
    keep_edges = set()

    stack = [sel]
    while stack:
        cur = stack.pop()
        for s, t in edge_set:
            if s == cur and (s, t) not in keep_edges:
                keep_edges.add((s, t))
                keep_nodes.add(t)
                stack.append(t)

    stack = [sel]
    while stack:
        cur = stack.pop()
        for s, t in edge_set:
            if t == cur and (s, t) not in keep_edges:
                keep_edges.add((s, t))
                keep_nodes.add(s)
                stack.append(s)

    for el in elements:
        if 'source' in el['data']:  # edge
            keep = ((el['data']['source'], el['data']['target']) in keep_edges
                    or el['data']['rel'] == 'PV')
        else:  # node
            keep = el['data']['id'] in keep_nodes

        if keep:
            el['classes'] = el['classes'].replace(' faded', '')
        else:
            if 'faded' not in el['classes']:
                el['classes'] += ' faded'

    return elements


# ---------------------- Search -------------------------------
@app.callback(Output('prot-directory', 'children'),
              Input('prot-search', 'value'))
def filter_directory(query):
    query = (query or "").strip().lower()
    blocks = []
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        group = [p for p in PROTS if p.startswith(letter)]
        if query:
            group = [p for p in group if query in p.lower()]
        if not group:
            continue
        blocks.append(
            html.Details([
                html.Summary(f"{letter} ({len(group)})",
                             style={'cursor': 'pointer', 'fontSize': 20}),
                html.Ul([
                    html.Li(html.A(p, href=f"/protein/{p}",
                                   style={'textDecoration': 'none',
                                          'color': '#0366d6',
                                          'fontWeight': 'bold'
                                          if query and query in p.lower()
                                          else 'normal'}))
                    for p in group
                ], style={'columnCount': 3, 'listStyle': 'none',
                          'padding': 0, 'margin': '6px 0'})
            ], open=bool(query))
        )
    return blocks


@app.callback(Output('url', 'href', allow_duplicate=True),
              Input('submit-prot', 'n_clicks'),
              State('prot-search', 'value'),
              prevent_initial_call=True)
def jump_to_protein(_, value):
    return f"/protein/{value}" if value else dash.no_update


# --------------------Run App----------------------------–
if __name__ == "__main__":
    app.run(debug=DEBUG, port=PORT, host="0.0.0.0")
