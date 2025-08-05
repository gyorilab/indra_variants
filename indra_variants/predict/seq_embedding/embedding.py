# Embedding by ESM-2 with sequence caching and cropping
import pandas as pd
import torch
from tqdm import tqdm
import re
import esm

print("Loading ESM-2 model...")
model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
model = model.eval().cpu()
batch_converter = alphabet.get_batch_converter()


df = pd.read_csv("training_feature_input.tsv", sep="\t")
sequences = df["sequence"]
mutations = df["variant_info"]


def parse_mutation_pos(vinfo):
    match = re.match(r"[A-Z]([0-9]+)[A-Z\*]", str(vinfo).strip())
    return int(match.group(1)) if match else None


# center crop（max:4096）
def center_crop_sequence(seq, mut_pos, max_len=4096):
    L = len(seq)
    if L <= max_len:
        return seq
    if mut_pos is None:
        return seq[:max_len]
    start = max(0, mut_pos - max_len // 2)
    end = min(L, start + max_len)
    return seq[start:end]


# embedding
def embed_sequence(seq):
    data = [("protein", " ".join(seq))]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[36], return_contacts=False)
    token_representations = results["representations"][36]
    embedding = token_representations[0, 1:-1].mean(dim=0).cpu().tolist()
    return embedding


# write cache
embedding_cache = {}
embeddings = []

print("Embedding sequences with ESM-2 ...")
for seq, vinfo in tqdm(zip(sequences, mutations), total=len(sequences)):
    if pd.isna(seq):
        embeddings.append([0.0]*2560)
        continue

    # check cache for exist sequense
    if len(seq) <= 4096:
        if seq in embedding_cache:
            emb = embedding_cache[seq]
        else:
            try:
                emb = embed_sequence(seq)
                embedding_cache[seq] = emb
            except Exception as e:
                print(f"Error on sequence: {e}")
                emb = [0.0]*2560
        embeddings.append(emb)
        continue

    # crop based on the mut_pos
    mut_pos = parse_mutation_pos(vinfo)
    cropped_seq = center_crop_sequence(seq, mut_pos)
    try:
        emb = embed_sequence(cropped_seq)
    except Exception as e:
        print(f"Error on long sequence with mutation {vinfo}: {e}")
        emb = [0.0]*2560
    embeddings.append(emb)

# save result
emb_df = pd.DataFrame(embeddings, columns=[f"esm2_{i}" for i in range(2560)])
df_out = pd.concat([df.reset_index(drop=True), emb_df], axis=1)
df_out.to_csv("variant_with_esm2.tsv", sep="\t", index=False)
print("Saved to variant_with_esm2.tsv")
