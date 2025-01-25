import annoy
from sentence_transformers import SentenceTransformer
import json
from embeddings_es import get_rag_candidates, load_data, load_index, get_rag_candidates_openai
from chat_completions import get_answer_from_ollama, get_answer_from_openai
import logging
from concurrent.futures import ThreadPoolExecutor


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')


def run(query, index, openai_embeddings_index, chunk_id_mapping, model_type='local'):
    candidatos = get_rag_candidates(model, query, index, chunk_id_mapping)
    openai_candidatos = get_rag_candidates_openai(query, openai_embeddings_index, chunk_id_mapping)
    texts = [candidato['preprocess_content'] for candidato in candidatos]
    texts2 = [candidato['preprocess_content'] for candidato in openai_candidatos]
    if model_type == 'local':
        llm_respuesta = get_answer_from_ollama(query, texts)
        llm_respuesta2 = get_answer_from_ollama(query, texts2)
        logging.info('RESPUESTA FARLOPA', llm_respuesta2)
    else:
        llm_respuesta = get_answer_from_openai(query, texts)
    return llm_respuesta



if __name__ == '__main__':
    # Load data and index
    _, chunk_id_mapping = load_data('../data/juego_de_tronos_chunks_300.json')
    index = load_index('index_juego_de_tronos_chunk_300.ann', 768)
    index_openai = load_index('index_juego_de_tronos_chunks_512_openai.ann', 1536)
    # Configure logging to save to a file
    logging.basicConfig(filename='llm_responses.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Start interactive session
    while True:
        query = input("Introduce tu pregunta. Escribe /bye para salir: ")
        if query == "/bye":
            break
        
        logging.info(f"Received query: {query}")
        
        # Run both models concurrently
        
        with ThreadPoolExecutor() as executor:
            future_local = executor.submit(run, query, index, index_openai, chunk_id_mapping, 'local')
            future_openai = executor.submit(run, query, index, index_openai, chunk_id_mapping, 'openai')
            
            local_llm_respuesta = future_local.result()
            openai_llm_respuesta = future_openai.result()
        
        logging.info("Generated response from local model:")
        logging.info(local_llm_respuesta)
        
        logging.info("Generated response from OpenAI model:")
        logging.info(openai_llm_respuesta)