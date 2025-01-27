import logging
from typing import Optional
from pydantic import BaseModel
from chat_completions import OllamaRAGCompletionWrapper, CommonsOllamaCompletionWrapper, OpenAiCompletionWrapper
from got_rag.rag import OpenaiEmbeddings, LocalEmbeddings
from dc_training import get_dc_cls, predict
from dotenv import load_dotenv
import asyncio

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class QueryRequest(BaseModel):
    query: str
    context: Optional[str]


class ChatBot:
    """
    Clase ChatBot para manejar la clasificación de mensajes y la generación de respuestas
    utilizando diferentes motores de LLM (Local y OpenAI).

    Atributos:
    ----------
    dc_cls : modelo de clasificación de dominio.
    tfidf_vectorizer : vectorizador TF-IDF.
    commons_handler : handler de respuestas comunes.
    index_path : path al índice de embeddings.
    chunks_mapping_path : path al archivo de mapeo de chunks.
    llm : LLM seleccionado.
    embeddings : embeddings correspondientes al motor LLM seleccionado.
    """

    def __init__(self, llm_engine, index_path, mapping_path):
        """
        Inicializa la clase ChatBot con el motor LLM deseado, la ruta al índice y la ruta al archivo de mapeo de chunks.

        Parámetros:
        -----------
        llm_engine : str
            Motor LLM a utilizar ('local' o 'openai').
        index_path : str
            path al index de embeddings.
        mapping_path : str
            path al archivo de mapeo de chunks.
        """
        self.dc_cls, self.tfidf_vectorizer = get_dc_cls()
        self.commons_handler = CommonsOllamaCompletionWrapper()
        self.index_path = index_path
        self.chunks_mapping_path = mapping_path
        self.llm, self.embeddings = self.get_llm_engine(llm_engine)

    def classify_message(self, message: str) -> str:
        """
        Clasifica el mensaje para determinar si debe ser manejado por Commons o RAG.

        Parámetros:
        -----------
        message : str
            Mensaje a clasificar.

        Returns:
        --------
        str
            Clasificación del mensaje ('commons' o 'got').
        """
        dc_prediction = predict(message.lower(), self.dc_cls, self.tfidf_vectorizer) # Need to be preprocessed
        logging.debug(f"DC: {dc_prediction}")
        return dc_prediction

    def get_llm_engine(self, llm_engine):
        """
        Obtiene el motor LLM y los embeddings correspondientes según el motor seleccionado.

        Parámetros:
        -----------
        llm_engine : str
            Motor LLM a utilizar ('local' o 'openai').

        Returns:
        --------
        tuple
            Motor LLM y embeddings correspondientes.
        """
        if llm_engine == 'local':
            engine = OllamaRAGCompletionWrapper()
            embeddings = LocalEmbeddings(self.index_path, 768, self.chunks_mapping_path)
        else:
            engine = OpenAiCompletionWrapper()
            embeddings = OpenaiEmbeddings(self.index_path, 1536, self.chunks_mapping_path)
        return engine, embeddings

    async def get_response(self, query: str) -> str:
        """
        Genera una respuesta para la consulta dada utilizando el motor LLM seleccionado.

        Parámetros:
        -----------
        query : str
            Consulta para la cual se debe generar una respuesta.

        Returns:
        --------
        str
            Respuesta generada por el motor LLM.
        """
        classification = self.classify_message(query)
        if classification == "commons":
            llm_response = self.commons_handler.get_answer(query)
        else:  # classification: got
            candidates = self.embeddings.get_candidates(query)
            logging.debug(candidates)
            llm_response = self.llm.get_answer(query, candidates)
        return llm_response


if __name__ == '__main__':
    # Ejemplo de uso
    chatbot = ChatBot('local', 'index_juego_de_tronos_chunk_512.ann', '../data/jdt_chunks_sentences_512.json')
    response = asyncio.run(chatbot.get_response("¿Quién es Jon Snow?"))
    print(response)