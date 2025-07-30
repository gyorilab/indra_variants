"""This script processes INDRA statements to create a mapping
from mutated human gene names to their associated PubMed IDs (PMIDs)
based on a local dump of the INDRA database."""
import gzip
import os
import csv
import tqdm
from collections import defaultdict
import pandas as pd
from indra.statements import stmt_from_json
from indra_cogex.util import load_stmt_json_str


HERE = os.path.dirname(os.path.abspath(__file__))

# Input - this file is a dump of INDRA statements
fname = "/Users/jicijiang/code/data/processed_statements.tsv.gz"
# Output
output_filename = os.path.join(HERE, os.pardir, "data", "genes_to_pmids.tsv")

if __name__ == "__main__":
    # Build gene-to-PMID mapping using a defaultdict of set.
    genes_to_pmids = defaultdict(set)
    counter = 0

    with gzip.open(fname, "rt") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for stmt_hash_str, stmt_json_str in tqdm.tqdm(reader,
                                                      desc="First pass"):
            counter += 1
            stmt_hash = int(stmt_hash_str)
            stmt_json = load_stmt_json_str(stmt_json_str)
            stmt = stmt_from_json(stmt_json)

            # Ensure we have a human gene with a specific mutation.
            if not len(stmt.real_agent_list()) == 2:
                continue
            first_agent = stmt.real_agent_list()[0]
            if 'HGNC' not in first_agent.db_refs:
                continue
            gene_name = first_agent.name
            if not first_agent.mutations:
                continue
            mut = first_agent.mutations[0]
            if not (mut.residue_from and mut.residue_to and mut.position):
                continue

            evidence_list = stmt_json["evidence"]
            for evidence in evidence_list:
                tr = evidence.get("text_refs", {})
                pmid = tr.get("PMID") or evidence.get("pmid")
                if pmid:
                    # Convert PMID to string for consistency.
                    genes_to_pmids[gene_name].add(str(pmid))

            if counter % 100 == 0:
                rows = []
                for gene, pmid_set in genes_to_pmids.items():
                    pmids_str = ",".join(sorted(list(pmid_set)))
                    rows.append({"gene": gene, "pmids": pmids_str})
                df = pd.DataFrame(rows)
                df.to_csv(output_filename, sep="\t", index=False)
                print(f"[INFO] Processed {counter} rows, partial "
                      f"gene-to-PMID mapping saved to {output_filename}")

    rows = []
    for gene, pmid_set in genes_to_pmids.items():
        pmids_str = ",".join(sorted(list(pmid_set)))
        rows.append({"gene": gene, "pmids": pmids_str})
    df = pd.DataFrame(rows)
    df.to_csv(output_filename, sep="\t", index=False)
    print(f"[INFO] Final gene-to-PMID mapping saved to {output_filename}")
