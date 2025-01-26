# RAG Chatbot for "A Game of Thrones" (Example Implementation)

"""
Este script implementa un chatbot que responde preguntas sobre el libro "Juego de Tronos" de George R. R. Martin,
haciendo uso de un enfoque de Recuperación de Información Asistida por Generación (RAG) con un modelo de lenguaje.

Requisitos previos:
1. Archivo PDF con el texto de "Juego de Tronos".
2. Librerías Python para manejo de PDFs, embeddings, y un modelo LLM.
3. No se usa LangChain (realiza integración manual de cada paso).

Funcionamiento:
1. Preprocesamiento y extracción de texto desde el PDF.
2. Dividir el contenido en fragmentos y generar embeddings.
3. Almacenar embeddings y textos en una estructura de datos (por ejemplo, FAISS o una DB local).
4. Chatbot:
   - Dada una consulta, recupera los fragmentos más relevantes.
   - Genera la respuesta usando un modelo LLM y el contexto de los fragmentos.
   - Si la pregunta no está relacionada con el libro, responderá que no tiene información.

NOTA:
- Este script es un ejemplo simplificado. En un escenario real, se recomienda modularizar en varios archivos.
- Se incluyen logs para documentar el proceso.
- Explica supuestos y próximos pasos.

"""

import logging
import re
import os

# ============== CONFIGURACIÓN DE LOGGING ===============
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# ============== IMPORTACIONES NECESARIAS ===============
try:
    import PyPDF2
    from transformers import AutoTokenizer, AutoModel
    import torch
    import numpy as np
    # Para almacenar los embeddings (puedes usar FAISS u otra alternativa)
    import faiss
    # Para la parte de generación de texto, usaré un modelo GPT2 local como ejemplo
    # Puedes cambiar a un modelo open-source más grande o API de OpenAI
    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
except ImportError as e:
    logger.error(f"Error importando librerías: {e}")
    raise e

# ============== 1. PARSEAR EL PDF Y DIVIDIR EN FRAGMENTOS ===============
def extract_text_from_pdf(pdf_path):
    """
    Extrae todo el texto de un PDF.
    Args:
        pdf_path (str): Ruta al archivo PDF de "Juego de Tronos".
    Returns:
        str: Texto completo del PDF.
    """
    logger.info("Extrayendo texto del PDF...")
    text = ""
    try:
        with open(pdf_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        logger.error(f"Error leyendo el archivo PDF: {e}")
        raise e
    return text


def chunk_text(text, chunk_size=1000, overlap=100):
    """
    Divide el texto en fragmentos de longitud aproximada `chunk_size`, con un solapamiento de `overlap` caracteres.
    Esto facilita la indexación y el embedding.
    Args:
        text (str): Texto completo a dividir.
        chunk_size (int): Tamaño de cada fragmento en caracteres.
        overlap (int): Número de caracteres que se solapan entre fragmentos.
    Returns:
        List[str]: Lista de fragmentos.
    """
    logger.info("Dividiendo texto en fragmentos...")
    tokens = text.split()
    fragments = []

    current_pos = 0
    while current_pos < len(tokens):
        fragment = tokens[current_pos:current_pos + chunk_size]
        fragments.append(" ".join(fragment))
        current_pos += chunk_size - overlap
        if current_pos < 0:
            break
    logger.info(f"Total de fragmentos creados: {len(fragments)}")
    return fragments


# ============== 2. GENERACIÓN DE EMBEDDINGS ===============
class EmbeddingModel:
    """
    Clase para manejar el modelo de embeddings. Puedes utilizar Sentence-Transformers u otro.
    En este ejemplo, uso un modelo de BERT genérico (distilbert-base-uncased) para ilustrar.
    """

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        logger.info(f"Cargando tokenizer y modelo de embeddings {model_name}...")
        try:
            from sentence_transformers import SentenceTransformer
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


# ============== 3. ALMACENAMIENTO DE EMBEDDINGS E ÍNDICE (FAISS) ===============

def build_faiss_index(embeddings, fragments, dimension=384):
    """
    Construye un índice FAISS para consultas rápidas de similitud.
    Args:
        embeddings (List[np.array]): Lista de vectores de embeddings.
        fragments (List[str]): Lista de fragmentos de texto.
        dimension (int): Dimensión del embedding.
    Returns:
        index (faiss.IndexFlatL2): Índice FAISS entrenado.
        fragments (List[str]): Lista de fragmentos (se almacena para referencia).
    """
    logger.info("Construyendo índice FAISS...")
    index = faiss.IndexFlatL2(dimension)
    # Convertir la lista de embeddings a un array de float32
    embeddings_array = np.array(embeddings, dtype=np.float32)
    index.add(embeddings_array)
    logger.info(f"Se agregó {embeddings_array.shape[0]} vectores al índice.")
    return index, fragments


def search_index(query_embedding, index, fragments, k=3):
    """
    Busca en el índice FAISS los k fragmentos más similares al embedding de la consulta.
    Args:
        query_embedding (np.array): Vector de embedding de la consulta.
        index (faiss.IndexFlatL2): Índice FAISS.
        fragments (List[str]): Lista de fragmentos de texto.
        k (int): Número de resultados a devolver.
    Returns:
        List[str]: Lista de los fragmentos más relevantes.
    """
    query_vector = np.array([query_embedding], dtype=np.float32)
    distances, indices = index.search(query_vector, k)
    result_fragments = []
    for idx in indices[0]:
        result_fragments.append(fragments[idx])
    return result_fragments


# ============== 4. INTERFAZ CON EL MODELO DE LENGUAJE (LLM) ===============

class LocalLLM:
    """
    Ejemplo de clase para un modelo de lenguaje local.
    Para un mejor rendimiento, se recomienda un modelo instruct-tuned (ej. GPT-NeoX, Mistral, etc.).
    """
    def __init__(self, model_name="gpt2"):
        logger.info(f"Cargando modelo local de lenguaje: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()

    def generate(self, prompt, max_length=100, temperature=0.7):
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)


# ============== 5. LOGICA DEL CHATBOT (RAG) ===============

def generate_answer(question, index, fragments, embedding_model, llm, threshold=0.3, k=3):
    """
    Genera una respuesta usando RAG.
    1. Obtener embedding de la pregunta.
    2. Buscar los k fragmentos más similares en el índice.
    3. Construir un prompt con el contexto.
    4. Generar la respuesta con el LLM.
    5. Si no se encuentra un fragmento relevante (p. ej. similitud baja), responder que no hay información.
    """
    # 1. Obtener embedding de la pregunta
    query_emb = embedding_model.get_embedding(question)

    # 2. Buscar fragmentos relevantes
    relevant_frags = search_index(query_emb, index, fragments, k=k)

    # (Opcional) Podríamos verificar la distancia para determinar relevancia.
    # Por simplicidad, asumimos que FAISS ya nos trae lo más relevante.

    # 3. Construir prompt con contexto
    context_str = "\n".join([f"Fragmento {i+1}: {frag}" for i, frag in enumerate(relevant_frags)])
    prompt = (
        f"La siguiente información proviene del libro 'Juego de Tronos':\n"
        f"{context_str}\n\n"
        f"Pregunta: {question}\n"
        "Responde en base al contexto proporcionado. Si no hay suficiente contexto, di que no cuentas con esa info.\n"
    )

    # 4. Generar respuesta
    answer = llm.generate(prompt)

    return answer


# ============== 6. SCRIPT PRINCIPAL ===============

def main(pdf_path="/path/a/juego_de_tronos.pdf"):
    """
    Flujo principal del script:
    1. Extraer texto del PDF.
    2. Dividir en fragmentos.
    3. Crear embeddings e índice.
    4. Iniciar loop de chat.
    """
    # 1. Extraer texto
    text = extract_text_from_pdf(pdf_path)

    # 2. Dividir en fragmentos
    fragments = chunk_text(text, chunk_size=300, overlap=50)

    # 3. Crear modelo de embeddings e índice
    embedding_model = EmbeddingModel()
    embeddings = [embedding_model.get_embedding(frag) for frag in fragments]
    # Dimension estimada del modelo (varía según el embedding_model)
    dimension = len(embeddings[0])
    index, stored_fragments = build_faiss_index(embeddings, fragments, dimension=dimension)

    # 4. Cargar modelo LLM (local)
    llm = LocalLLM(model_name="gpt2")

    # 5. Iniciar loop de chat
    logger.info("Iniciando chatbot RAG. Escribe 'salir' para terminar.")
    while True:
        user_input = input("\nUsuario: ")
        if user_input.lower() in ["salir", "exit", "quit"]:
            logger.info("Saliendo del chatbot.")
            break
        # Generar respuesta
        answer = generate_answer(user_input, index, stored_fragments, embedding_model, llm)
        print("Bot:", answer)


if __name__ == "__main__":
    # Reemplaza la ruta del PDF con la tuya
    pdf_path = "./juego_de_tronos.pdf"  # Ejemplo
    if not os.path.exists(pdf_path):
        logger.warning(f"El archivo PDF {pdf_path} no existe en esta ruta.")
    main(pdf_path)

# ============== DOCUMENTACIÓN, SUPUESTOS Y PRÓXIMOS PASOS ===============
"""
SUPUESTOS:
1. El archivo PDF es "Juego de Tronos" en formato texto que puede ser procesado por PyPDF2.
2. El modelo de embeddings "all-MiniLM-L6-v2" se instala con pip:
   pip install sentence-transformers
3. Para FAISS:
   pip install faiss-cpu
4. Para el modelo local GPT2:
   pip install transformers torch

PRÓXIMOS PASOS:
- Utilizar un modelo LLM instruct-tuned que entienda mejor las preguntas (e.g. GPT-Neo, Llama 2, etc.).
- Mejorar la división de fragmentos (chunking) para que respete saltos de página o capítulos.
- Almacenar los embeddings y fragmentos en una base de datos real (p. ej. SQLite, Postgres, vectordb).
- Manejar de forma más rigurosa la similitud y relevancia (por ejemplo, revisar la distancia devuelta por FAISS).
- Implementar un modo "streamlit" para visualización.
- Extender el script con un pipeline de Logging más robusto y configuraciones en archivo YAML.
"""
