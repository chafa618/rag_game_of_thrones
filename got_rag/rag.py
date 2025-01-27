import logging
from sentence_transformers import SentenceTransformer
import spacy
import json
import annoy
import os
import openai
from dotenv import load_dotenv
import numpy as np
import argparse
#from functools import lru_cache

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


#@lru_cache(maxsize=128)
def get_embeddings(model, text):
    return model.encode(text)

def get_openai_embeddings(text):
    client = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY'])

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding

def create_index_openai(chunks_file, index_name):
    
    logging.info(f"Loading chunks from {chunks_file}")
    
    with open(chunks_file, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)

    embedding_size = 1536

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
    ids_potenciales_respuestas, distances = index.get_nns_by_vector(embedding_pregunta, 5, include_distances=True)
    potenciales_respuestas = [chunk_id_mapping[idx] for idx in ids_potenciales_respuestas]
    postprocess_respuestas = postprocess_candidates(potenciales_respuestas, distances)
    if len(postprocess_respuestas)<=1:
        logging.warning('No candidates over threshold!')
    return potenciales_respuestas

def get_rag_candidates_openai(query, index, chunk_id_mapping):
    openai_embeddings = get_openai_embeddings(query)
    ids_potenciales_respuestas, distances = index.get_nns_by_vector(openai_embeddings, 5, include_distances=True)
    potenciales_respuestas = [chunk_id_mapping[idx] for idx in ids_potenciales_respuestas]
    postprocess_respuestas = postprocess_candidates(potenciales_respuestas, distances)
    if len(postprocess_respuestas)<=1:
        logging.warning('No candidates over threshold!')
    return potenciales_respuestas


class OpenaiEmbeddings:
    def __init__(self, index, emb_size, chunks_mapping):
        """
        Inicializa una instancia de la clase.
        Args:
            index (str): Path a la bbdd vectorial que se desea usar.
            emb_size (int): El tamaño de los embeddings.
            chunks_mapping (str): Path a archivo de mappeo con los fragmentos originales
        """
        
        self.client = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        self.index = load_index(index, emb_size)
        _, self.chunk_id_mapping = load_data(chunks_mapping)

    def get_embeddings(self, text):
        """
        Obtiene la representacional vectorial de un texto dado.
        
        Args: 
            text (str): texto a procesar. 
        
        Returns:
            embeddings: representacion vectorial del texto en cuestión. 
        """
        
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    
    def get_candidates(self, text):
        """
        Obtiene los candidatos más cercanos basados en los embeddings del texto proporcionado.

        Args:
            text (str): El texto a procesar.

        Returns:
            list: Una lista de candidatos después del postprocesamiento.

        Advertencias:
            Si el número de respuestas postprocesadas es menor o igual a 1, se registra una advertencia.
        """
        
        embeddings = self.get_embeddings(text)
        ids, distances = self.index.get_nns_by_vector(embeddings, 5, include_distances=True)
        candidates = [self.chunk_id_mapping[idx] for idx in ids]
        postprocess_respuestas = postprocess_candidates(candidates, distances)
        if len(postprocess_respuestas) <= 1:
            logging.warning('No candidates over threshold!')
        return candidates


class LocalEmbeddings:
    def __init__(self, index, emb_size, chunks_mapping):
        """
        Clase que permite obtener representaciones vectoriales
        de un texto a partir de una bbdd vectorial existente. 
        
        Args:
            index (str): Path a la base de datos vectorial.
            emb_size (int): Entero que indica el tamaño de los vectores
            chunks_mapping (str): Path a archivo de mappeo con los fragmentos originales
            del texto de Juego de tronos.
        
        """
        self.client = EmbeddingModel()
        self.index = load_index(index, emb_size)
        _, self.chunk_id_mapping = load_data(chunks_mapping)

    def get_embeddings(self, text):
        """
        Obtiene la representacional vectorial de un texto dado.
        
        Args: 
            text (str): texto a procesar. 
        
        Returns:
            embeddings: representacion vectorial del texto en cuestión. 
        """
        embeddings = self.client.get_embedding(text)
        return embeddings
    
    def get_candidates(self, text):
        """
        Obtiene los candidatos más cercanos basados en los embeddings del texto proporcionado.

        Args:
            text (str): El texto a procesar.

        Returns:
            list: Una lista de candidatos después del postprocesamiento.

        Advertencias:
            Si el número de respuestas postprocesadas es menor o igual a 1, se registra una advertencia.
        """
        embeddings = self.get_embeddings(text)
        ids, distances = self.index.get_nns_by_vector(embeddings, 5, include_distances=True)
        candidates = [self.chunk_id_mapping[idx] for idx in ids]
        postprocess_respuestas = postprocess_candidates(candidates, distances)
        if len(postprocess_respuestas) <= 1:
            logging.warning(f'No candidates over threshold!\n-DISTANCES: {distances}')
        return candidates


class EmbeddingModel:
    """
    Clase que usa un modelo SentenceTransformer para vectorizar un texto y obtener embeddings
    """
    def __init__(self, model_name="paraphrase-multilingual-mpnet-base-v2"):
        logger.info(f"Cargando modelo de embeddings {model_name}...")
        try:
            self.model = SentenceTransformer(model_name)
            logger.info("Modelo de embeddings cargado exitosamente.")
        except Exception as e:
            logger.error(f"No se pudo cargar el modelo de embeddings: {e}")
            raise e

    def get_embedding(self, text):
        """
        Devuelve el embedding de un texto.
        Args:
            text (str): Texto a procesar.
        Returns:
            np.array: Vector con el embedding.
        """
        return self.model.encode(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create an Annoy index for text embeddings.")
    parser.add_argument('--model', type=str, default='sentence-transformer', choices=['sentence-transformer', 'openai'], help='The model to use for embeddings.')
    parser.add_argument('--chunks_file', type=str, required=True, help='Path to the JSON file containing text chunks.')
    parser.add_argument('--index_name', type=str, required=True, help='Name of the index file to create.')

    args = parser.parse_args()

    if args.model == 'sentence-transformer':
        logging.info("Loading SentenceTransformer model...")
        model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        logging.info("Creating Index file Using SentenceTransformers")
        create_index(model, args.chunks_file, args.index_name)
    elif args.model == 'openai':
        logging.info("Creating Index file using OpenAi")
        create_index_openai(args.chunks_file, args.index_name)