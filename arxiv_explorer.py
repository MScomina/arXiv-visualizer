import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from helper.data_loader import fetch_rows
from helper.nearest_neighbors import build_faiss_index
from pyvis.network import Network

DATASET_PATH = "./data/processed/compressed_embeddings.sqlite3"

@st.cache_resource(show_spinner=False)
def get_index():
    index, index_to_id = build_faiss_index(n_rows=300_000, chunk_size=100_000)
    return index, index_to_id

@st.cache_data()
def fetch_data(field : str, value : str, _conn : sqlite3.Connection):
    return fetch_rows(
        conn=conn,
        search_substring=value,
        target_col=field,
        select_cols=["id", "title", "abstract", "embedding"]
        )

conn = sqlite3.connect(DATASET_PATH, check_same_thread=False)
index, index_to_id = get_index()

st.title("ArXiv Explorer")

search_field = st.selectbox(
    "Choose field to search",
    options=["ID", "Title", "Abstract"],
    index=0
)

search_value = st.text_input(
    f"Enter {search_field} to search for"
)

if st.button("Search"):
    if not search_value:
        st.warning("Please enter a value to search.")
    else:
        try:
            results = fetch_data(search_field.lower(), search_value, conn)

            if len(results[search_field.lower()]) == 0:
                raise ValueError("No matches found for the given query.")

            st.write(f"**Found {len(results[search_field.lower()])} result(s)**")
            for k in range(len(results[search_field.lower()])):
                st.markdown(f"**ID:** {results["id"][k]}")
                st.markdown(f"**Title:** {results["title"][k]}")
                st.markdown(f"**Abstract:** {results["abstract"][k]}")
                st.markdown("---")

        except ValueError as exc:
            st.error(exc)