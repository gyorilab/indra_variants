"""This module runs the network generation pipeline."""

import pandas as pd
import gzip
import csv
import tqdm
import networkx as nx
import os
from collections import defaultdict
from functools import lru_cache
from indra.assemblers.indranet import IndraNetAssembler
from indra.ontology.bio import bio_ontology
from indra.statements import stmt_from_json
from indra_cogex.util import load_stmt_json_str


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
fname = "/Users/jicijiang/code/data/processed_statements.tsv.gz"
INDEX_FILENAME = os.path.join(HERE, os.pardir, "data", "genes_to_pmids.tsv")
BATCH_WRITE_SIZE = 5

# Define statement types once for reuse
DIRECTIONAL_TYPES = {
    # Enzyme-catalyzed modifications (enzyme -> substrate)
    'Acetylation', 'Deacetylation', 'Phosphorylation', 'Dephosphorylation',
    'Methylation', 'Demethylation', 'Ubiquitination', 'Deubiquitination',
    'Sumoylation', 'Desumoylation', 'Glycosylation', 'Deglycosylation',
    'Hydroxylation', 'Dehydroxylation', 'Farnesylation', 'Defarnesylation',
    'Geranylgeranylation', 'Degeranylgeranylation', 'Myristoylation', 'Demyristoylation',
    'Palmitoylation', 'Depalmitoylation', 'Ribosylation', 'Deribosylation',
    'Autophosphorylation', 'Transphosphorylation',
    
    # Regulatory relationships (regulator -> target)
    'Activation', 'Inhibition', 'RegulateActivity', 'RegulateAmount',
    'Gap', 'Gef', 'GtpActivation',
    
    # State/amount changes (cause -> effect)
    'ActiveForm', 'Conversion', 'IncreaseAmount', 'DecreaseAmount',
    
    # Localization changes (mover -> moved or cause -> effect)
    'Migration', 'Translocation',
    
    # General modifications (modifier -> modified)
    'AddModification', 'Modification', 'RemoveModification', 'SelfModification'
}

NON_DIRECTIONAL_TYPES = {
    # Symmetric relationships
    'Association', 'Complex',
    
    # Property relationships
    'HasActivity'
}

# -----------------------------------------------------------------------------
# Step‑0: get index file
# -----------------------------------------------------------------------------
print("[INFO] Loading gene → PMID index…")
index_df = pd.read_csv(INDEX_FILENAME, sep="\t")

gene_index = {
    row["gene"].strip(): [p.strip() for p in str(row["pmids"]).split(",") if p.strip()]
    for _, row in index_df.iterrows()
}
needed_pmids = {pmid for pmids in gene_index.values() for pmid in pmids}
print(f"[INFO] Indexed {len(gene_index)} proteins, {len(needed_pmids)} unique PMIDs.")

# -----------------------------------------------------------------------------
# Step‑1: one-time scanning .gz → pmid_to_jsonstrs
# -----------------------------------------------------------------------------
print("[INFO] Scanning statements file（one-time I/O）…")
pmid_to_jsonstrs = defaultdict(list)


with gzip.open(fname, "rt") as fh:
    reader = csv.reader(fh, delimiter="\t")
    for _, stmt_json_str in tqdm.tqdm(reader, desc="Indexing statements"):
        try:
            stmt_json = load_stmt_json_str(stmt_json_str)
        except Exception:
            continue
        for ev in stmt_json.get("evidence", []):
            pmid = str(ev.get("text_refs", {}).get("PMID") or ev.get("pmid") or "")
            if pmid and pmid in needed_pmids:
                pmid_to_jsonstrs[pmid].append(stmt_json_str)
                break
print(f"[INFO] Captured statements for {len(pmid_to_jsonstrs)} / {len(needed_pmids)} PMIDs.")


# -----------------------------------------------------------------------------
# Cache functions: bio_ontology.get_type & get_edge_relation
# -----------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _cached_get_type(ns: str, ident: str):
    return bio_ontology.get_type(ns, ident)


@lru_cache(maxsize=100_000)
def _cached_edge_relation(stmts_id: int, u: str, v: str):
    """Cache edge relation lookup with proper error handling"""
    try:
        return get_edge_relation_from_statements(_statements_registry[stmts_id], u, v) or "?"
    except KeyError:
        # Statements not in registry, possibly cleaned up
        return "?"

# Registry tables
_graph_registry = {}
_statements_registry = {}


# -----------------------------------------------------------------------------
# Helper functions for edge relation and path validation
# -----------------------------------------------------------------------------
def get_edge_relation_from_statements(stmts, source, target):
    """
    Get edge relation from statements using the same logic as validation
    This ensures consistency between validation and chain construction
    """
    # **SAME LOGIC AS VALIDATION**: Store statements by agent pairs
    agent_pair_statements = defaultdict(lambda: {'directional': set(), 'non_directional': set()})
    
    # Process each statement and categorize by agent pairs
    for stmt in stmts:
        agent_names = [a.name for a in stmt.real_agent_list() if a and a.name]
        stmt_type = type(stmt).__name__
        
        if len(agent_names) >= 2:
            # For each pair of agents in the statement
            for i in range(len(agent_names)):
                for j in range(i + 1, len(agent_names)):  # Only consider unique pairs
                    agent1, agent2 = agent_names[i], agent_names[j]
                    # Create ordered pair key for consistent lookup
                    pair_key = tuple(sorted([agent1, agent2]))
                    
                    if stmt_type in DIRECTIONAL_TYPES:
                        # Store the actual direction: (subject, object)
                        direction = (agent1, agent2)  # i -> j (original order from statement)
                        agent_pair_statements[pair_key]['directional'].add((direction, stmt_type))
                    elif stmt_type in NON_DIRECTIONAL_TYPES:
                        agent_pair_statements[pair_key]['non_directional'].add(stmt_type)
                    else:
                        # Unknown statement type - treat as directional
                        direction = (agent1, agent2)
                        agent_pair_statements[pair_key]['directional'].add((direction, stmt_type))
    
    # Find the relation for the specific direction
    pair_key = tuple(sorted([source, target]))
    edge_direction = (source, target)
    
    if pair_key not in agent_pair_statements:
        return 'unknown'
    
    statements = agent_pair_statements[pair_key]
    
    # Check if there's a directional statement supporting this exact direction
    for direction, stmt_type in statements['directional']:
        if direction == edge_direction:
            return stmt_type
    
    # If no directional support, return first non-directional statement type
    if statements['non_directional']:
        return next(iter(statements['non_directional']))
    
    return 'unknown'


def get_edge_relation(g, source, target):
    """Get edge relation from graph, handling both MultiGraph and regular Graph"""
    # This is kept for backward compatibility, but we should use get_edge_relation_from_statements
    edge_data = g.get_edge_data(source, target)
    if edge_data is None:
        return 'unknown'
    
    if g.is_multigraph():
        # For MultiGraph, get all edge types
        edge_types = []
        for edge_key, data in edge_data.items():
            edge_types.append(data.get('stmt_type', 'unknown'))
        
        # Return the first directional type found, otherwise return the first type
        for edge_type in edge_types:
            if edge_type in DIRECTIONAL_TYPES:
                return edge_type
        return edge_types[0] if edge_types else 'unknown'
    else:
        # For regular Graph
        return edge_data.get('stmt_type', 'unknown')


def _validate_path_against_statements(path_nodes, stmts, start_protein):
    """
    Validate that each step in the path corresponds to actual INDRA statements
    This prevents incorrect reverse directions by properly handling mixed relationship types
    """
    if len(path_nodes) < 2:
        return True
    
    # **CORRECT APPROACH**: Store statements by agent pairs, not by directed edges
    # Key: (agent1, agent2) - unordered pair, Value: {directional_stmts, non_directional_stmts}
    agent_pair_statements = defaultdict(lambda: {'directional': set(), 'non_directional': set()})
    
    # Process each statement and categorize by agent pairs
    for stmt in stmts:
        agent_names = [a.name for a in stmt.real_agent_list() if a and a.name]
        stmt_type = type(stmt).__name__
        
        if len(agent_names) >= 2:
            # For each pair of agents in the statement
            for i in range(len(agent_names)):
                for j in range(i + 1, len(agent_names)):  # Only consider unique pairs
                    agent1, agent2 = agent_names[i], agent_names[j]
                    # Create ordered pair key for consistent lookup
                    pair_key = tuple(sorted([agent1, agent2]))
                    
                    if stmt_type in DIRECTIONAL_TYPES:
                        # Store the actual direction: (subject, object)
                        direction = (agent1, agent2)  # i -> j (original order from statement)
                        agent_pair_statements[pair_key]['directional'].add((direction, stmt_type))
                    elif stmt_type in NON_DIRECTIONAL_TYPES:
                        agent_pair_statements[pair_key]['non_directional'].add(stmt_type)
                    else:
                        # Unknown statement type - treat as directional
                        print(f"[WARNING] Unknown statement type: {stmt_type}, treating as directional")
                        direction = (agent1, agent2)
                        agent_pair_statements[pair_key]['directional'].add((direction, stmt_type))
    
    # **CORRECT VALIDATION**: Check each step in the path
    for i in range(len(path_nodes) - 1):
        current_node = path_nodes[i]
        next_node = path_nodes[i + 1]
        
        # Create ordered pair key for lookup
        pair_key = tuple(sorted([current_node, next_node]))
        edge_direction = (current_node, next_node)
        
        if pair_key not in agent_pair_statements:
            print(f"[DEBUG] No statements found for pair: {current_node} <-> {next_node}")
            return False
        
        statements = agent_pair_statements[pair_key]
        edge_supported = False
        
        # Check if there's a directional statement supporting this exact direction
        for direction, stmt_type in statements['directional']:
            if direction == edge_direction:
                edge_supported = True
                break
        
        # If no directional support, check for non-directional statements
        if not edge_supported and statements['non_directional']:
            edge_supported = True
        
        if not edge_supported:
            print(f"[DEBUG] Invalid edge direction: {current_node} -> {next_node}")
            print(f"[DEBUG] Available directional statements: {statements['directional']}")
            print(f"[DEBUG] Available non-directional statements: {statements['non_directional']}")
            return False
    
    return True


# -----------------------------------------------------------------------------
# Process (protein, pmid) one by one
# -----------------------------------------------------------------------------
def process_protein_pmid(protein: str, pmid: str, raw_json_list: list[str]):
    """Process a single protein-PMID pair to extract variant-to-endpoint pathways"""
    if not raw_json_list:
        return []
    
    # Parse statements from JSON
    try:
        stmts = [stmt_from_json(load_stmt_json_str(s)) for s in raw_json_list]
    except Exception:
        return []
    
    # Build graph from statements
    try:
        G = IndraNetAssembler(stmts).make_model()
        
        # **CRITICAL FIX**: Ensure graph is directed for proper pathway directionality
        if not isinstance(G, nx.DiGraph):
            G = G.to_directed()
            
    except Exception:
        return []
    
    # Check if protein exists in graph
    if protein not in G.nodes:
        return []

    # !!FIXED: Register both graph and statements for consistent processing
    G_id = id(G)
    stmts_id = id(stmts)
    _graph_registry[G_id] = G
    _statements_registry[stmts_id] = stmts

    # Extract variant information
    variant_set = {
        f"{m.residue_from}{m.position}{m.residue_to}"
        for s in stmts for a in s.real_agent_list() 
        if a and a.name == protein and a.mutations
        for m in a.mutations 
        if m.residue_from and m.position and m.residue_to
    } or {protein}

    # Identify endpoint nodes (biological processes and diseases)
    endpoint_nodes, name_to_go = set(), {}
    for s in stmts:
        for ag in s.real_agent_list():
            if ag and ag.db_refs:
                for ns, ident in ag.db_refs.items():
                    if _cached_get_type(ns, ident) in {"biological_process", "disease"}:
                        endpoint_nodes.add(ag.name)
                        if ns == "GO":
                            name_to_go[ag.name] = ident
    
    if not endpoint_nodes:
        return []

    # !!FIXED: Use directed graph for shortest path calculation with strict validation
    try:
        all_paths = nx.single_source_shortest_path(G, protein)
    except (nx.NodeNotFound, Exception):
        return []

    # !!DEBUG: Print graph edges for debugging
    # print(f"[DEBUG] Graph edges for {protein}: {list(G.edges(data=True))}")

    # Build pathway records
    records = []
    for endpoint in endpoint_nodes:
        final_endpoint = endpoint
        
        # Handle GO ID mapping if necessary
        if endpoint not in G.nodes and endpoint in name_to_go and name_to_go[endpoint] in G.nodes:
            final_endpoint = name_to_go[endpoint]
        
        if final_endpoint not in all_paths:
            continue
            
        path_nodes = all_paths[final_endpoint]
        
        # !!CRITICAL: Validate path against original INDRA statements
        valid_path = _validate_path_against_statements(path_nodes, stmts, protein)
        
        if not valid_path:
            continue
        
        # Build pathway chain with edge relations using consistent logic
        chain_parts = [path_nodes[0]]
        for i in range(len(path_nodes) - 1):
            rel = _cached_edge_relation(stmts_id, path_nodes[i], path_nodes[i + 1])
            chain_parts.append(f"-[{rel}]-> {path_nodes[i + 1]}")
        chain_str = " ".join(chain_parts)
        
        # Create records for each variant
        for variant in variant_set:
            records.append({
                "variant_protein": protein,
                "variant_info": variant,
                "biological_process/disease": endpoint,
                "chain": chain_str,
                "pmid": pmid,
            })
    
    # Clean up the registries to prevent memory leaks
    _graph_registry.pop(G_id, None)
    _statements_registry.pop(stmts_id, None)
    return records


# -----------------------------------------------------------------------------
# Serial execution
# -----------------------------------------------------------------------------
tasks = [(p, pmid) for p, pmids in gene_index.items() for pmid in pmids]
print(f"[INFO] Processing {len(tasks)} (protein, PMID) pairs in serial mode…")

results = []
for protein, pmid in tqdm.tqdm(tasks, desc="Processing pairs"):
    results.extend(process_protein_pmid(protein, pmid, pmid_to_jsonstrs.get(pmid, [])))

if not results:
    print("[WARNING] No data extracted. Exiting.")
    exit(0)

df_all = pd.DataFrame(results)
print(f"[INFO] Total extracted rows: {len(df_all)}")

# -----------------------------------------------------------------------------
# Output: TSV files by protein
# -----------------------------------------------------------------------------
for protein, df_sub in df_all.groupby("variant_protein"):
    output_file = f"{protein}_variant_effects.tsv"
    
    # Remove existing file if present
    if os.path.exists(output_file):
        os.remove(output_file)
    
    # Write in batches
    header = True
    for start in range(0, len(df_sub), BATCH_WRITE_SIZE):
        df_sub.iloc[start:start+BATCH_WRITE_SIZE].to_csv(
            output_file, sep="\t", index=False, mode="a", header=header
        )
        header = False
    print(f"[INFO] Saved {len(df_sub)} rows → {output_file}")

print("[INFO] Pipeline finished (serial mode, on‑the‑fly edge relation).")