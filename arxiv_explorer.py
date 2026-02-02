# arxiv_explorer.py
import streamlit as st
import numpy as np
from helper.nearest_neighbors import build_faiss_index
from helper.data_loader import read_row_parquet

index, index_to_id = build_faiss_index(n_rows=300_000, chunk_size=100_000)

test_row = read_row_parquet(
    "./data/processed/compressed_embeddings.parquet",
    ["id", "abstract", "embedding", "title"],
    [("id", "=", "0806.4067")]
)

k = 6
similarities, neighbour_idxs = index.search(
    np.array(test_row["embedding"]).reshape(1, -1), k
)
neighbour_ids = [index_to_id[idx] for idx in neighbour_idxs[0].tolist()]

neighbour_rows = []
for neigh_id in neighbour_ids:
    row = read_row_parquet(
        "./data/processed/compressed_embeddings.parquet",
        ["id", "title", "abstract", "embedding"],
        [("id", "=", neigh_id)]
    )
    neighbour_rows.append(row)

st.title("ArXiv Paper Explorer")

st.subheader(f"Query Paper (ID: {test_row['id']})")
st.markdown(f"**Title:** {test_row['title']}")
st.markdown(f"**Abstract:** {test_row['abstract']}")
st.markdown("---")

st.subheader("Similar Papers")
for i, row in enumerate(neighbour_rows):
    if row["id"] == test_row["id"]:
        continue
    with st.expander(f"{i+1}. {row['title']} (ID: {row['id']})"):
        st.markdown(f"**Title:** {row['title']}")
        st.markdown(f"**Abstract:** {row['abstract']}")
        st.markdown(f"**Embedding shape:** {np.array(row['embedding']).shape}")

scores = similarities[0].tolist()
scores = [round(s, 4) for s in scores]
st.subheader("Similarity Scores")
st.table(
    {"Paper": [f"{row['id']}" for row in neighbour_rows if row["id"] != test_row["id"]],
     "Score": scores}
)