![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
# Project: ArXiv Visualizer
Project for the course "Progettazione del Software e dei Sistemi Informativi" - UniTS.

## Description
The project consists of an arXiv paper visualizer, where papers' abstracts are first converted into a semantic embedding, compressed through a trained Autoencoder and then used for k-NN to find the closest papers.
<br><br>
Course catalogue: [Available here](https://units.coursecatalogue.cineca.it/corsi/2022/10507/insegnamenti/2022/117492/2016/5?annoOrdinamento=2016&coorte=2022)

## How to run
In order to run this project, you have to fetch the arXiv metadata file available [on Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv) (or the [arXiv bulk-data page](https://info.arxiv.org/help/bulk_data.html)).
### Python setup
After downloading the file and setting up the .env available in the repo, you can setup your favorite local environment (through .venv or conda) and install the required libraries in requirements.txt:
```sh
pip install -r requirements.txt
```
Then you have to generate the embeddings, train the autoencoder and compress the embeddings by running the following scripts in this order:
```sh
# embedding_script.py -> training_ae_script.py -> compress_script.py
python3 ./scripts/embedding_script.py
python3 ./scripts/training_ae_script.py
python3 ./scripts/compress_script.py
```
You can then run the actual arXiv visualizer by running it through Streamlit:
```sh
streamlit run arxiv_explorer.py
```