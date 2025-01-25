import logging
from typing import Optional
import openai
import ollama
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from chat_completions import get_answer_from_ollama, get_answer_from_openai, get_commons_llm_answer
from embeddings_es import get_rag_candidates, get_rag_candidates_openai, load_data, load_index
from dc_training import get_dc_cls, predict

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()

model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
dc, tfidf = get_dc_cls()

_, chunk_id_mapping = load_data('../data/jdt_chunks_sentences_512.json')
index = load_index('index_juego_de_tronos_chunks_512_openai.ann', 1536)
index_olama = load_index('index_juego_de_tronos_chunk_512.ann', 768)

class QueryRequest(BaseModel):
    query: str
    context: Optional[str]

@app.post("/openai")
async def openai_endpoint(request: QueryRequest):
    try:
        candidates = get_rag_candidates_openai(request.query, index, chunk_id_mapping)
        logging.info(f"Calling /openai: {candidates}")
        answer = get_answer_from_openai(request.query, candidates)
        return {"answer": answer}
    except Exception as e:
        logging.error(f"Error in OpenAI endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post("/local_common")
async def get_common_response(request: QueryRequest):
    try:
        answer = get_commons_llm_answer(request.query)
        return {"answer": answer}
    except Exception as e:
        logging.error(f"Error in Commons endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post("/ollama")
async def ollama_endpoint(request: QueryRequest):
    try:
        candidates = get_rag_candidates(model, request.query, index_olama, chunk_id_mapping)
        logging.info(candidates)
        answer = get_answer_from_ollama(request.query, candidates)
        return {"answer": answer}
    except Exception as e:
        logging.error(f"Error in Ollama endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")



@app.post("/classify_domain")
async def dc_endpoint(request: QueryRequest):
    print(request.query)
    try:
        dc_prediction = predict(request.query, dc, tfidf)
        return {"answer": dc_prediction}
    except Exception as e:
        logging.error(f"Error in DC endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

#Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)