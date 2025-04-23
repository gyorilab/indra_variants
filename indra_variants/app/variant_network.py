import os, re, math, pandas as pd, networkx as nx, dash
from dash import html, dcc, Input, Output, State, MATCH, ALL
import dash_cytoscape as cyto

from config import DATA_DIR, PORT, DEBUG

cyto.load_extra_layouts()


TSV_RE   = re.compile(r"^(?P<prot>.+)_variant_effects\.tsv$", re.I)
PROTS = sorted(
    TSV_RE.match(p.name).group("prot")
    for p in DATA_DIR.iterdir() if TSV_RE.match(p.name)
)


# ---------- Bulid Graph --------------------
def build_elements(prot: str):
    df_path = DATA_DIR / f"{prot}_variant_effects.tsv"
    df = pd.read_csv(df_path, sep="\t")
    G = nx.MultiDiGraph();  G.add_node(prot)

    for _, row in df.iterrows():
        var = row["variant_info"]
        G.add_edge(prot, var, relation="PV", pmid="")
        src = var
        for seg in row["chain"].split(" -[")[1:]:
            if "]->" not in seg: continue
            rel, tgt = seg.split("]->")
            G.add_edge(src.strip(), tgt.strip(),
                       relation=rel.strip(), pmid=row["pmid"])
            src = tgt.strip()

    variants  = set(df["variant_info"])
    endpoints = set(df["biological_process/disease"])
    freq      = df["biological_process/disease"].value_counts().to_dict()
    layer = lambda n: 0 if n==prot else 1 if n in variants else 3 if n in endpoints else 2

    # --- ring layout
    pos = {prot:(0,0)}
    for L in range(1,4):
        ring=[n for n in G if layer(n)==L]
        for i,n in enumerate(ring):
            r   = 280*(L+2)
            ang = 2*math.pi*i/len(ring)
            pos[n]=(r*math.cos(ang), r*math.sin(ang))

    # --- elements
    rel_types = sorted({d['relation'] for *_,d in G.edges(data=True) if d['relation']!='PV'})
    palette = ["#e74c3c","#2ecc71","#3498db","#f39c12","#9b59b6"]
    rel_color = {r:palette[i%len(palette)] for i,r in enumerate(rel_types)}

    els=[]
    for n,(x,y) in pos.items():
        L=layer(n)
        size = 60 if L!=3 else 50+10*freq.get(n,1)
        label= '' if L==3 else n
        els.append({"data":{"id":n,"label":label,"real":n},
                    "position":{"x":x,"y":y},
                    "classes":f"L{L}",
                    "style":{"width":size,"height":size}})
    for u,v,d in G.edges(data=True):
        cls='edge-PV' if d['relation']=='PV' else f"edge-{d['relation']}"
        els.append({"data":{"id":f"{u}->{v}_{d['pmid']}",
                            "source":u,"target":v,
                            "pmid":d["pmid"],"rel":d["relation"]},
                    "classes":cls})
    edge_set={(u,v) for u,v in G.edges()}
    return els, rel_types, rel_color, list(edge_set)   

# ---------- Dash app --------------------------------------------
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.layout = html.Div([dcc.Location(id="url"), html.Div(id="page")])

# ---------- Home Page -----------------------------------------------
def homepage():
    blocks=[]
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        group=[p for p in PROTS if p.startswith(letter)]
        if not group: continue
        blocks.append(html.Details([
            html.Summary(f"{letter} ({len(group)})"),
            html.Ul([html.Li(html.A(p, href=f"/protein/{p}")) for p in group],
                    style={'columnCount':3,'listStyle':'none','padding':0,'margin':'4px 0'})
        ]))
    return html.Div(blocks, style={'margin':'40px','fontFamily':'sans-serif'})

# ---------- Network Page----------------------------------------------
def network_page(prot:str):
    els, rel_types, rel_color, edge_set = build_elements(prot)

    def style_for_rel(r,c):
        return {'selector':f'.edge-{r}','style':{
                   'line-color':c,'target-arrow-color':c,
                   'target-arrow-shape':'triangle',
                   'curve-style':'bezier','width':2}}

    unique_id = f"net-{prot}"          
    return html.Div([
        dcc.Link("← Home", href="/"), html.Br(),
        html.H3(f"{prot} Variant Network", style={'textAlign':'center'}),
        html.Button("RESET", id={'type':'reset-btn','prot':prot}, n_clicks=0),
        dcc.Store(id={'type':'store-els','prot':prot},  data=els),
        dcc.Store(id={'type':'store-edges','prot':prot},data=edge_set),
        cyto.Cytoscape(
            id={'type':'cy-net','prot':prot},
            elements=els, layout={'name':'preset'},
            style={'width':'100%','height':'860px'},
            stylesheet=[
                {'selector':'node','style':{
                    'shape':'ellipse','background-opacity':0.5,
                    'font-size':38,'font-weight':'bold',
                    'label':'data(label)','text-valign':'center','text-halign':'center'}},
                {'selector':'.L0','style':{'background-color':'#aacdd7','color':'#004466','label':'data(real)'}},
                {'selector':'.L1','style':{'background-color':'#a492bb','color':'#573d82'}},
                {'selector':'.L2','style':{'background-color':'#cce9b6','color':'#3f6330'}},
                {'selector':'.L3','style':{'background-color':'#fabf77','color':'#b05e04','label':'data(real)'}},
                {'selector':'.edge-PV','style':{'line-color':'#d5cbc9','target-arrow-color':'#d5cbc9',
                                                'target-arrow-shape':'triangle','curve-style':'bezier','width':2}},
                *[style_for_rel(r,c) for r,c in rel_color.items()],
                {'selector':'.faded','style':{'opacity':0.15}}
            ]),
        # legend
        html.Div([
            html.H4("Legend",style={'margin':0,'fontSize':18,'fontWeight':'bold'}),
            html.Ul([
                html.Li([html.Span('-->',style={'color':rel_color.get(r,'#d5cbc9'),
                                              'marginRight':'8px','fontSize':20}), r],
                        style={'fontSize':18,'listStyle':'none','margin':'2px 0','fontWeight':'bold'})
                for r in (['PV']+rel_types)
            ])
        ], style={'position':'absolute','top':'75px','right':'28px','background':'rgba(255,255,255,0.85)',
                  'padding':'8px 12px','borderRadius':'6px','boxShadow':'0 0 4px rgba(0,0,0,0.3)'}),
        # info box
        html.Div(id={'type':'edge-info','prot':prot},
            style={'position':'absolute','top':'75px', 'right':'280px','minWidth':'150px',
                   'background':'rgba(255,255,255,0.9)','padding':'10px 14px',
                   'borderRadius':'6px','boxShadow':'0 0 6px rgba(0,0,0,0.25)',
                   'fontSize':18,'fontWeight':'bold','zIndex':999})
    ])


@app.callback(Output("page","children"), Input("url","pathname"))
def serve_page(path):
    if path in (None,"/"): return homepage()
    if path.startswith("/protein/"):
        prot = path.split("/")[2]
        if prot in PROTS: return network_page(prot)
    return html.H3("404 – Not found")

# ---------- Edge-info callback (pattern-matching) ----------------
@app.callback(
    Output({'type':'edge-info','prot':MATCH}, 'children'),
    [Input({'type':'cy-net','prot':MATCH}, 'tapEdgeData'),
     Input({'type':'reset-btn','prot':MATCH},  'n_clicks')],
    prevent_initial_call=True
)
def show_info(edge, _):
    if edge and edge.get('pmid'):
        pmid = edge['pmid']
        rel = edge.get('rel', 'N/A')
        link = html.A("🔗 PubMed", href=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                      target="_blank", style={'marginLeft': '10px'})
        return html.Div([
            html.Div(f"PMID: {pmid}"),
            html.Div(f"Relation: {rel}"),
            html.Div(link)
        ])
    return ""

# ---------- Highlight callback  ---------------------------------
@app.callback(
    Output({'type':'cy-net','prot':MATCH}, 'elements'),
    [Input({'type':'cy-net','prot':MATCH}, 'tapNodeData'),
     Input({'type':'reset-btn','prot':MATCH}, 'n_clicks')],
    [State({'type':'store-els','prot':MATCH},  'data'),
     State({'type':'store-edges','prot':MATCH},'data')],
    prevent_initial_call=True
)
def highlight(node, n_reset, elements, edge_set):
    # turn to set(tuple) 
    edge_set = set(tuple(e) for e in edge_set)

    # RESET
    if dash.callback_context.triggered and \
       dash.callback_context.triggered[0]['prop_id'].endswith('reset-btn.n_clicks'):
        for el in elements:
            el['classes'] = el['classes'].replace(' faded','')
        return elements

    if not node:  # Click blank
        return elements

    sel = node['id']
    keep_nodes={sel}; keep_edges=set()

    stack=[sel]
    while stack:
        cur=stack.pop()
        for s,t in edge_set:
            if s==cur and (s,t) not in keep_edges:
                keep_edges.add((s,t)); keep_nodes.add(t); stack.append(t)
    stack=[sel]
    while stack:
        cur=stack.pop()
        for s,t in edge_set:
            if t==cur and (s,t) not in keep_edges:
                keep_edges.add((s,t)); keep_nodes.add(s); stack.append(s)

    updated=[]
    for el in elements:
        if 'source' in el['data']:
            fade=((el['data']['source'],el['data']['target']) not in keep_edges
                  and el['data']['rel']!='PV')
            el['classes']=el['classes'].replace(' faded','')+(' faded' if fade else '')
        else:
            el['classes']=el['classes'].replace(' faded','')+('' if el['data']['id'] in keep_nodes else ' faded')
        updated.append(el)
    return updated

# ----------------------------- Run app ----------------------------------
if __name__ == "__main__":
    app.run(debug=DEBUG, port=PORT, host="0.0.0.0")
