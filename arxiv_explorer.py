import streamlit as st
import pandas as pd
import numpy as np
from numpy import linalg as la
import sqlite3
import json
import streamlit.components.v1 as components
from helper.data_loader import fetch_rows
from helper.nearest_neighbors import build_faiss_index
from pyvis.network import Network

DATASET_PATH = "./data/processed/compressed_embeddings.sqlite3"
N_NEIGHBORS = 10

stss = st.session_state

@st.cache_resource(show_spinner=False)
def get_index():
    index, index_to_id = build_faiss_index(n_rows=3_000_000, chunk_size=100_000)
    return index, index_to_id

@st.cache_resource(show_spinner=False)
def get_sql_connection():
    return sqlite3.connect(DATASET_PATH, check_same_thread=False)

@st.cache_data(max_entries=5000)
def fetch_data(field: str, value: str | tuple[str], _conn: sqlite3.Connection, cols : tuple[str, ...] = ("id", "title", "abstract", "embedding")):
    return fetch_rows(
        conn=_conn,
        search_substring=value,
        target_col=field,
        select_cols=list(cols)
    )

conn = get_sql_connection()
index, index_to_id = get_index()

st.set_page_config(
    page_title="My ArXiv Explorer",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("ArXiv Explorer")

search_field = st.selectbox(
    label="Search by",
    options=("ID", "Title"),
    index=0
)

search_value = st.text_input(
    label="Enter search value",
    placeholder=f"Type a {search_field.lower()} here"
)

# Search logic
if st.button("Search"):
    if not search_value:
        st.warning("Please enter a value to search.")
    else:
        db_field = search_field.lower()
        results = fetch_data(
            field=db_field,
            value=search_value,
            _conn=conn,
            cols=("id", "title", "abstract", "embedding")
        )

        result_rows = results

        stss.search_results = result_rows

# Selection logic
if stss.get("search_results", False):
    result_rows = stss["search_results"]

    if result_rows and len(result_rows["id"]) != 0:
        st.success(f"Found {len(result_rows["id"])} matching entries.")

        options = [
            f"{id_} - {title[:60]}{'...' if len(title) > 60 else ''}"
            for id_, title in zip(result_rows["id"], result_rows["title"])
        ]

        selected_option = st.selectbox(
            label="Select one entry to view",
            options=options,
            placeholder="-- Select an entry --",
            index=None,
            accept_new_options=False
        )

        if selected_option:
            selected_paper_id = selected_option.split(" - ", 1)[0]
            selected_index = 0

            while selected_paper_id != stss.search_results["id"][selected_index]:
                selected_index += 1

            stss.selected_paper = dict()
            stss.selected_paper["id"] = selected_paper_id
            stss.selected_paper["title"] = stss.search_results["title"][selected_index]
            stss.selected_paper["abstract"] = stss.search_results["abstract"][selected_index]
            stss.selected_paper["embedding"] = stss.search_results["embedding"][selected_index]


# Network graph creation logic
if stss.get("selected_paper", False):
    if st.button("Find nearest papers"):

        stss.network = Network(height="400px", width="100%", bgcolor="#222222", font_color="white")
        stss.network.add_node(stss.selected_paper["id"], label=stss.selected_paper["id"], title=stss.selected_paper["title"],
                                shape="dot", color="#FF4136", physics=False)

        np_embedding = np.array(json.loads(stss.selected_paper["embedding"]), dtype=np.float16)
        np_embedding = np_embedding/la.norm(np_embedding)
        first_similarity, first_indices = index.search(np_embedding.reshape(1,-1), N_NEIGHBORS+1)

        first_similarity = np.delete(first_similarity, 0, axis=1)
        first_indices = np.delete(first_indices, 0, axis=1)
        first_indices = np.sort(first_indices, axis=1)

        first_neighbors = tuple(index_to_id[index] for index in first_indices[0])
        f_neighbors_data = fetch_data(
            field="id",
            value=first_neighbors,
            _conn=conn,
            cols=("id", "title", "abstract", "embedding")
        )
        for k in range(N_NEIGHBORS):
            stss.network.add_node(f_neighbors_data["id"][k], label=f_neighbors_data["id"][k], title=f_neighbors_data["title"][k],
                                    shape="dot", color="#3641FF", physics=False)
            stss.network.add_edge(stss.selected_paper["id"], f_neighbors_data["id"][k], value=first_similarity[0][k].item(), color="#3641FF")

        first_np_embeddings = np.array([json.loads(f_neighbors_data["embedding"][l]) for l in range(N_NEIGHBORS)], dtype=np.float16)
        first_np_embeddings = first_np_embeddings/la.norm(first_np_embeddings, axis=1, keepdims=True)

        second_similarity, second_indices = index.search(first_np_embeddings, N_NEIGHBORS+1)

        second_similarity = second_similarity[:, 1:]
        second_indices   = second_indices[:, 1:]

        second_neighbors = tuple(index_to_id[index] for index in second_indices.flatten())

        s_neighbors_data = fetch_data(
            field="id",
            value=second_neighbors,
            _conn=conn,
            cols=("id", "title", "abstract")
        )

        for id_, title in zip(s_neighbors_data["id"], s_neighbors_data["title"]):
            if id_ not in first_neighbors and id_ != stss.selected_paper["id"]:
                stss.network.add_node(id_, label=id_, title=title, shape="dot", color="#36FF41", physics=False)

        for i, first_id in enumerate(first_neighbors):
            for j in range(N_NEIGHBORS):
                second_id = index_to_id[second_indices[i, j]]
                weight = second_similarity[i, j].item()
                if first_id == second_id:
                    print(i, j)
                    print(first_id)

                stss.network.add_edge(
                    first_id,
                    second_id,
                    value=weight,
                    color="#36FF41"
                )
                

# Network graph display logic
if stss.get("network", False):
    network_html = stss.network.generate_html()
    components.html(network_html, height=450, width='100%')