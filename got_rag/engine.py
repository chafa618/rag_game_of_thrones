import annoy
from sentence_transformers import SentenceTransformer
import json

model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')


def get_embeddings(text):
    return model.encode(text)

def load_data(path):
    with open(path, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)
        
    mapping = make_mapping(json_data)
    return json_data, mapping

def make_mapping(json_data):
    chunk_id_mapping = {}
    for chunk in json_data:
        chunk_id_mapping[chunk["chunk_id"]] = chunk
    return chunk_id_mapping

def load_index(path, emb_size):
    index = annoy.AnnoyIndex(emb_size, 'angular')
    index.load(path)
    return index

def process_query(query, index, chunk_id_mapping):
    embedding_pregunta = get_embeddings(query)
    ids_potenciales_respuestas = index.get_nns_by_vector(embedding_pregunta, 5)
    potenciales_respuestas = [chunk_id_mapping[idx] for idx in ids_potenciales_respuestas]
    texto_potencial = [chunk for chunk in potenciales_respuestas]
    return texto_potencial, potenciales_respuestas

def run(query):
    _, chunk_id_mapping = load_data('juego_de_tronos_chunks_300.json')
    index = load_index('index_juego_de_tronos_chunk_300.ann', 768)
    return process_query(query, index, chunk_id_mapping)

#_, chunk_id_mapping = load_data('juego_de_tronos_chunks_300.json')
#index = load_index('index_juego_de_tronos_chunk_300.ann', 768)



if __name__ == '__main__':
    while True:
        query = input("Introduce tu pregunta. Escribe /bye para salir: ")
        if query == "/bye":
            break
        texto_potencial, _ = run(query)
        print('RS: ', _)
        print("Texto potencial:")
        for texto in texto_potencial:
            print(texto)
        print("\n")


