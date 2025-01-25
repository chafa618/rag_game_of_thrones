import logging
import openai
import ollama
from dotenv import load_dotenv
import os

load_dotenv()

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def preprocess_context(context_documents):
    context_divider = "\n---\n"
    context = context_divider.join(context_documents)
    return context


def get_answer_from_openai(query, context):
    context = preprocess_context([i['preprocess_content'] for i in context])
    #logging.info(f"From GetOPENAI {context}")
    client = openai.Client(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {
                "role": "developer", "content": "Sos un sistema de RAG especializado en el libro de G.R.R. Martin Cancion de hielo y fuego. Voy a proveerte de una pregunta y una serie de párrafos del libro como contexto. Usando esa información vas a redactar una respuesta acorde a la pregunta. Omitirás toda mención a la palabra contexto y a la saga de television de Juego de tronos."
            },
            {
                "role": "user",
                "content": f"Pregunta: ```{query}```\nContexto:\n```{context}```"
            },
        ]
    )
    return response.choices[0].message.content


def get_answer_from_ollama(query, context):
    logging.info(f'Query Ollama: {query}\n{context}')
    context = preprocess_context([i['preprocess_content'] for i in context])
    system_prompt = """Sos Asistente experto en Juego de Tronos en español. Recibirás una pregunta y un potencial contexto proveniente del libro de Juego de Tronos. Vas a analizar la pregunta y los contextos, y vas a elaborar una respuesta basandote en la información proporcionada. omitirás en la respuesta toda mencion literal a la palabra contexto. Respondé una vez que tengas la respuesta final. Si no sabes que decir, simplemente respondé que no tenes información relevante al respecto."""

    prompt = f"""Contexto: \n```{context}```\nMensaje: {query}"""
    
    
    messages = [
        {'role': 'assistant', 'content': system_prompt},
        {'role': 'user', 'content': prompt},
    ]
    generate_params = {
        'model': "qwen:1.8b", # "mistral:7b",
        'options': ollama.Options(temperature=0.1, num_ctx=512),
        'messages': messages,
        'stream': False
    }

    response = ollama.chat(**generate_params)

    logging.info({'ollama_rag_answer': response})
    return response['message']['content']


def get_commons_llm_answer(query):

    #system_prompt = """Sos un asistente virtual. Vas a recibir un mensaje y tendrás que continuar con la conversacion con un mensaje corto. Terminarás tu mensaje animando al otro a que te pregunte algo relacionado con el libro Cancion de Fuego y Hielo de G.R.R. Martin. Responde siempre en español y usando hasta 10 palabras."""

    system_prompt = """Sos un asistente virtual que puede responder consultas sobre Canción de Hielo y Fuego. Tu funcion en continuar la conversación. Responderás con un mensaje corto e invitarás siempre a que se te pregunte algo relacionado con el libro Cancion de Hielo y Fuego de G.R.R. Martin. Responde siempre en español y usando hasta 10 palabras.
    Por ejemplo: 
        - Si te dicen 'Hola', responderás con algo simialr a 'Hola, soy un asistente virtual especializado en el libro Cancion de Hielo y Fuego. Qué deseas saber?'
        - Si te dicen 'gracias' o 'genial', responderás con algo similar a 'Qué bien. Sigamos, te puedo ayudar con algo más sobre Cancion de Hielo y Fuego?'
    """

    messages = [
        {'role': 'assistant', 'content': system_prompt},
        {'role': 'user', 'content': query},
    ]
    generate_params = {
        'model': "qwen:1.8b", # "mistral:7b",
        'options': ollama.Options(temperature=0.2, num_ctx=256),
        'messages': messages,
        'stream': False  # Set to True if you want real-time responses
    }

    response = ollama.chat(**generate_params)

    logging.info({'ollama_generic_answer': response})
    return response['message']['content']
