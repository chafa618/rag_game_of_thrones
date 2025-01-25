import logging
from typing import Optional
from pydantic import BaseModel
from chat_completions import get_answer_from_ollama, get_answer_from_openai, get_commons_llm_answer
from embeddings_es import get_rag_candidates, load_data, load_index
from dc_training import get_dc_cls, predict
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import asyncio

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class QueryRequest(BaseModel):
    query: str
    context: Optional[str]

class ChatBot:
    def __init__(self, llm_engine):
        self.embeddings_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        self.dc, self.tfidf = get_dc_cls()
        self.llm = self.get_llm_engine(llm_engine)
        self.index = load_index('index_juego_de_tronos_chunk_512.ann', 768)
        _, self.chunk_id_mapping = load_data('../data/jdt_chunks_sentences_512.json')
        
    def classify_message(self, message: str) -> str:
        dc_prediction = predict(message, self.dc, self.tfidf)
        return dc_prediction
    
    

    async def get_response(self, query: str, context: Optional[str] = "") -> str:
        classification = self.classify_message(query)
        if classification == "got":
            candidates = await asyncio.to_thread(get_rag_candidates, self.model, query, self.index, self.chunk_id_mapping)
            return await asyncio.to_thread(get_answer_from_openai, query, context)
        else:

            logging.info(candidates)
            return await asyncio.to_thread(get_answer_from_ollama, query, candidates)

# Example usage
# chatbot = ChatBot()
# response = asyncio.run(chatbot.get_response("Who is Jon Snow?", "Context from the book..."))
# print(response)