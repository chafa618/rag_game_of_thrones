from sentence_transformers import SentenceTransformer
import spacy
import json
from annoy import AnnoyIndex
import os

model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')


def get_embeddings(text):
    return model.encode(text)


with open('juego_de_tronos_chunks_300.json', 'r', encoding='utf-8') as json_file:
    json_data = json.load(json_file)


embedding_size = 768
index_name = "index_juego_de_tronos_chunk_300.ann"

chunk_id_mapping = {}
for chunk in json_data:
    chunk_id_mapping[chunk["chunk_id"]] = chunk



if not os.path.exists(index_name):
    index = AnnoyIndex(embedding_size, 'angular')

    for idx, chunk in chunk_id_mapping.items():

        v = get_embeddings(chunk["content"])

        index.add_item(idx, v)

    index.build(10)
    index.save(index_name)