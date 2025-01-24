import logging
from sentence_transformers import SentenceTransformer
import spacy
import json
import annoy
import os
import openai
from dotenv import load_dotenv

load_dotenv()


def get_embeddings(model, text):
    return model.encode(text)

def get_openai_embeddings(text):
    client = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY'])

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    #print(response)
    return response.data[0].embedding

def create_index_openai(chunks_file, index_name):
    
    logging.info(f"Loading chunks from {chunks_file}")
    
    with open(chunks_file, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)

    embedding_size = 1536  # Updated embedding size for text-embedding-ada-002

    chunk_id_mapping = {}
    for chunk in json_data:
        chunk_id_mapping[chunk["chunk_id"]] = chunk

    if not os.path.exists(index_name):
        
        logging.info(f"Creating index {index_name}")
        index = annoy.AnnoyIndex(embedding_size, 'angular')
        
        for chunk_id, chunk in chunk_id_mapping.items():
            embedding = get_openai_embeddings(chunk["preprocess_content"])
            index.add_item(chunk_id, embedding)
        
        index.build(10)
        index.save(index_name)
        logging.info(f"Index {index_name} created and saved.")
    else:
        logging.info(f"Index {index_name} already exists.")

def create_index(model, chunks_file, index_name):
    
    logging.info(f"Loading chunks from {chunks_file}")
    
    with open(chunks_file, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)

    embedding_size = 768
    #index_name = "index_juego_de_tronos_chunk_300.ann"

    chunk_id_mapping = {}
    for chunk in json_data:
        chunk_id_mapping[chunk["chunk_id"]] = chunk

    if not os.path.exists(index_name):
        
        logging.info(f"Creating index {index_name}")
        
        index = annoy.AnnoyIndex(embedding_size, 'angular')

        for idx, chunk in chunk_id_mapping.items():

            v = get_embeddings(model, chunk["preprocess_content"])

            index.add_item(idx, v)

        index.build(10)
        index.save(index_name)

    logging.info(f"Index created and saved as {index_name}")
    
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

def postprocess_candidates(potenciales_respuestas, distances):
    filter_respuestas = []
    for candidate, dist in zip(potenciales_respuestas, distances):
        if dist <= .89:
            continue
        filter_respuestas.append(candidate)
    return filter_respuestas

def get_rag_candidates(embeddings_model, query, index, chunk_id_mapping):
    embedding_pregunta = get_embeddings(embeddings_model, query)
    ids_potenciales_respuestas, distances = index.get_nns_by_vector(embedding_pregunta, 10, include_distances=True)
    potenciales_respuestas = [chunk_id_mapping[idx] for idx in ids_potenciales_respuestas]
    postprocess_respuestas = postprocess_candidates(potenciales_respuestas, distances)
    return postprocess_respuestas

def get_rag_candidates_openai(query, index, chunk_id_mapping):
    openai_embeddings = get_openai_embeddings(query)
    ids_potenciales_respuestas, distances = index.get_nns_by_vector(openai_embeddings, 5, include_distances=True)
    potenciales_respuestas = [chunk_id_mapping[idx] for idx in ids_potenciales_respuestas]
    postprocess_respuestas = postprocess_candidates(potenciales_respuestas, distances)
    return postprocess_respuestas


if __name__ == "__main__":
    #logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Loading model...")
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    logging.info("Model loaded")
    create_index(model, "../data/jdt_chunks_sentences_256.json", "index_juego_de_tronos_chunk_256.ann")
    #create_index(model, "../jdt_chunks_sentences_512.json", "index_juego_de_tronos_chunk_512.ann")
    #create_index(model, "../jdt_chunks_sentences_1024.json", "index_juego_de_tronos_chunk_1024.ann")
    data, mapping = load_data("../data/jdt_chunks_sentences_256.json")
    #index = load_index("index_juego_de_tronos_chunk_512.ann", 768)
    #create_index_openai("../data/jdt_chunks_sentences_512.json", "index_juego_de_tronos_chunks_512_openai.ann")
        