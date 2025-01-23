import logging
from sentence_transformers import SentenceTransformer
import spacy
import json
from annoy import AnnoyIndex
import os
import openai


def get_embeddings(model, text):
    return model.encode(text)


def get_openai_embeddings(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']

def create_index_openai(model, chunks_file, index_name):
    
    logging.info(f"Loading chunks from {chunks_file}")
    
    with open(chunks_file, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)

    embedding_size = 1536  # Updated embedding size for text-embedding-ada-002

    chunk_id_mapping = {}
    for chunk in json_data:
        chunk_id_mapping[chunk["chunk_id"]] = chunk

    if not os.path.exists(index_name):
        
        logging.info(f"Creating index {index_name}")
        index = AnnoyIndex(embedding_size, 'angular')
        
        for chunk_id, chunk in chunk_id_mapping.items():
            embedding = get_openai_embeddings(chunk["chunk"])
            index.add_item(chunk_id, embedding)
        
        index.build(10)  # You can change the number of trees
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
        
        index = AnnoyIndex(embedding_size, 'angular')

        for idx, chunk in chunk_id_mapping.items():

            v = get_embeddings(model, chunk["preprocess_content"])

            index.add_item(idx, v)

        index.build(10)
        index.save(index_name)

    logging.info(f"Index created and saved as {index_name}")
    




if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Loading model...")
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    logging.info("Model loaded")
    #create_index(model, "../data/juego_de_tronos_chunks_300.json", "index_juego_de_tronos_chunk_300.ann")
    #create_index(model, "../jdt_chunks_sentences_512.json", "index_juego_de_tronos_chunk_512.ann")
    create_index(model, "../jdt_chunks_sentences_1024.json", "index_juego_de_tronos_chunk_1024.ann")