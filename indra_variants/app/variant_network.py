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

from config import DATA_DIR, PORT, DEBUG

cyto.load_extra_layouts()

TSV_RE = re.compile(r"^(?P<prot>.+)_variant_effects\.tsv$", re.I)
PROTS = sorted(TSV_RE.match(p.name).group("prot")
               for p in Path(DATA_DIR).iterdir()
               if TSV_RE.match(p.name))


# ----------------------Build Graph--------------------------–
def build_elements(prot: str):
    df_path = Path(DATA_DIR) / f"{prot}_variant_effects.tsv"
    df = pd.read_csv(df_path, sep="\t")

    G = nx.MultiDiGraph()
    G.add_node(prot)

    for _, row in df.iterrows():
        var = row["variant_info"]
        G.add_edge(prot, var, relation="PV", pmid="")
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
        if n in variants:
            return 1
        if n in endpoints:
            return 3
        return 2

    pos = {prot: (0, 0)}
    for L in range(1, 4):
        ring = [n for n in G if get_layer(n) == L]
        for i, n in enumerate(ring):
            r = 280 * (L + 2)
            ang = 2 * math.pi * i / len(ring)
            pos[n] = (r * math.cos(ang), r * math.sin(ang))

    rel_types = sorted({d['relation'] for _, _, d in G.edges(data=True)
                        if d['relation'] != 'PV'})
    palette = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6"]
    rel_color = {r: palette[i % len(palette)] for i, r in enumerate(rel_types)}

    els = []
    for n, (x, y) in pos.items():
        layer = get_layer(n)
        size = 60 if layer != 3 else 50 + 10 * freq.get(n, 1)
        label = '' if layer == 3 else n
        els.append({
            "data": {"id": n, "label": label, "real": n},
            "position": {"x": x, "y": y},
            "classes": f"L{layer}",
            "style": {"width": size, "height": size}
        })

    for u, v, d in G.edges(data=True):
        cls = 'edge-PV' if d['relation'] == 'PV' else f"edge-{d['relation']}"
        src4indra = prot if u in variants else u
        els.append({
            "data": {"id": f"{u}->{v}_{d['pmid']}",
                     "source": u, "target": v,
                     "pmid": d["pmid"], "rel": d["relation"],
                     "src4indra": src4indra},
            "classes": cls
        })

    edge_set = {(u, v) for u, v in G.edges()}
    return els, rel_types, rel_color, list(edge_set)


# ------------------------Dash App------------------------–
app = dash.Dash(__name__,
                suppress_callback_exceptions=True,
                external_stylesheets=[dbc.themes.FLATLY])
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
    els, rel_types, rel_color, edge_set = build_elements(prot)

    def rel_style(r, c):
        return {'selector': f'.edge-{r}',
                'style': {'line-color': c, 'target-arrow-color': c,
                          'target-arrow-shape': 'triangle',
                          'curve-style': 'bezier', 'width': 2}}

    return html.Div([
        dcc.Link("← Home", href="/"), html.Br(),
        html.H3(f"{prot} Variant Network", style={'textAlign': 'center'}),
        html.P("Tip: click the central protein/gene to clear all highlights.",
               style={'textAlign': 'center', 'marginTop': -6,
                      'marginBottom': 12, 'color': '#666',
                      'fontFamily': 'Arial, sans-serif'}),

        dcc.Store(id={'type': 'store-els',  'prot': prot},  data=els),
        dcc.Store(id={'type': 'store-edges', 'prot': prot},  data=edge_set),
        dcc.Store(id={'type': 'store-root', 'prot': prot},  data=prot),

        cyto.Cytoscape(
            id={'type': 'cy-net', 'prot': prot},
            elements=els, layout={'name': 'preset'},
            # Set width and height to fill the viewport
            style={'width': '100%', 'height': '100vh'},
            stylesheet=[
                {'selector': 'node', 'style': {
                    'shape': 'ellipse', 'background-opacity': 0.5,
                    'font-size': 38, 'font-weight': 'bold',
                    'label': 'data(label)',
                    'text-valign': 'center',
                    'text-halign': 'center'}},
                {'selector': '.L0',
                 'style': {'background-color': '#aacdd7',
                           'color': '#004466',
                           'label': 'data(real)'}},
                {'selector': '.L1',
                 'style': {'background-color': '#a492bb',
                           'color': '#573d82'}},
                {'selector': '.L2',
                 'style': {'background-color': '#cce9b6',
                           'color': '#3f6330'}},
                {'selector': '.L3',
                 'style': {'background-color': '#fabf77',
                           'color': '#b05e04',
                           'label': 'data(real)'}},
                {'selector': '.edge-PV',
                 'style': {'line-color': '#d5cbc9',
                           'target-arrow-color': '#d5cbc9',
                           'target-arrow-shape': 'triangle',
                           'curve-style': 'bezier',
                           'width': 2}},
                *[rel_style(r, c) for r, c in rel_color.items()],
                {'selector': '.faded', 'style': {'opacity': 0.15}}
            ]),

        # ---------- Legend ----------
        html.Div([
            html.H4("Legend",
                    style={'margin': 0, 'fontSize': 18,
                           'fontWeight': 'normal',
                           'fontFamily': 'Arial, sans-serif'}),
            html.Ul([
                html.Li([html.Span('→',
                                   style={'color': rel_color.get(r, '#d5cbc9'),
                                          'marginRight': 8,
                                          'fontSize': 18}), r],
                        style={'fontSize': 18, 'listStyle': 'none',
                               'margin': '1px 0'})
                for r in (['Gene to Variant'] + rel_types)
            ], style={'paddingLeft': 0})
        ], style={'position': 'absolute', 'top': 95, 'right': 28,
                  'background': 'rgba(255,255,255,0.85)',
                  'padding': '8px 12px',
                  'borderRadius': 6,
                  'boxShadow': '0 0 4px rgba(0,0,0,0.3)',
                  'fontFamily': 'Arial, sans-serif'}),

        # ---------- Edge-info ----------
        html.Div(id={'type': 'edge-info', 'prot': prot},
                 style={'position': 'absolute', 'top': 95, 'right': 250,
                        'minWidth': 150, 'background': 'rgba(255,255,255,0.9)',
                        'padding': '10px 14px', 'borderRadius': 6,
                        'boxShadow': '0 0 6px rgba(0,0,0,0.25)',
                        'fontSize': 18, 'fontFamily': 'Arial, sans-serif',
                        'zIndex': 999})
    ])


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


# ---------------------- edge info callback ------------------------------
@app.callback(
    Output({'type': 'edge-info', 'prot': MATCH}, 'children'),
    Input({'type': 'cy-net', 'prot': MATCH}, 'tapEdgeData'),
    prevent_initial_call=True)
def show_edge_info(edge):
    if not (edge and edge.get('pmid')):
        return ""

    pmid = edge['pmid']
    rel = edge.get('rel', 'N/A')

    pubmed_link = html.A("🔗 PubMed",
                         href=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                         target="_blank", style={'marginRight': 8})

    src = _url.quote_plus(edge['src4indra'])
    tgt = _url.quote_plus(edge['target'])
    indra_url = (f"https://discovery.indra.bio/search/"
                 f"?agent={src}&other_agent={tgt}"
                 "&agent_role=subject&other_role=object")
    indra_link = html.A("🔗 INDRA", href=indra_url, target="_blank")

    return html.Div([
        f"PMID: {pmid}", html.Br(),
        f"Relationship: {rel}", html.Br(),
        pubmed_link, indra_link
    ])


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
