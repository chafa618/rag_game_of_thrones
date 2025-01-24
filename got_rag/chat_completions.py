import logging
import openai
import ollama
from dotenv import load_dotenv
import os

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_context(context_documents):
    context_divider = "\n---\n"
    context = context_divider.join(context_documents)
    return context

def get_answer_from_openai(query, context):
    client = openai.Client(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {
                "role": "developer", "content": "Sos un sistema de RAG especializado en Juego de Tronos. Voy a proveerte de mensaje y su contexto proveniente del libro. Necesito que analices el mensaje y elabores una respuesta usando esa información. Si el mensaje y el contexto no estuvieran relacionados, solo responde que no hay relación entre ellos, sin más detalles."
            },
            {
                "role": "user",
                "content": f"Contexto:\n```{context}```\nMensaje: {query}"
            },
        ]
    )
    return response.choices[0].message.content

def get_answer_from_ollama(query, context):
    logging.info(f'Query Ollama: {query}\n{context}')
    system_prompt ="""Sos Asistente experto en Juego de Tronos en español. Recibirás una pregunta y un potencial contexto proveniente de Juego de Tronos. Vas a analizar la pregunta y los contextos, y vas a elaborar una respuesta basandote en la información proporcionada. Recorda omitir toda mencion literal a la palabra contexto. Respondé una vez que tengas la respuesta final. Si no sabes que decir, simplemente respondé que no tenes información relevante al respecto."""

    prompt = f"""Contexto: \n```{context}```\nMensaje: {query}"""
    messages = [
        {'role': 'assistant', 'content': system_prompt},
        {'role': 'user', 'content': prompt},
    ]
    generate_params = {
        'model': "mistral:7b",
        'options': ollama.Options(temperature=0.7, num_ctx=1024),
        'messages': messages,
        'stream': False
    }

    response = ollama.chat(**generate_params)
    
    logging.info({'ollama_rag_answer': response})
    return response['message']['content']


def get_commons_llm_answer(query):
    
    system_prompt = """Sos un asistente virtual. Vas a recibir un mensaje y tendrás que continuar con la conversacion con un mensaje corto. Terminarás tu mensaje animando al otro a que te pregunte algo relacionado con el libro Cancion de Fuego y Hielo de G.R.R. Martin. Responde siempre en español y usando hasta 10 palabras."""

    messages = [
        {'role': 'assistant', 'content': system_prompt},
        {'role': 'user', 'content': query},
    ]
    generate_params = {
        'model': "mistral:7b",
        'options': ollama.Options(temperature=0.0, num_ctx=1024),
        'messages': messages,
        'stream': False  # Set to True if you want real-time responses
    }

    # Get a response
    #cl = ollama.Client("chat")
    response = ollama.chat(**generate_params)
    
    logging.info({'ollama_generic_answer': response})
    return response['message']['content']