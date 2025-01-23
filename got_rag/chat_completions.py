import logging
import openai
import ollama
from dotenv import load_dotenv
import os

load_dotenv()

def preprocess_context(context_documents):
    logging.info(context_documents)
    context_divider = "\n---\n"
    context = context_divider.join(context_documents)
    return context

def get_answer_from_openai(query, context):
    client = openai.Client(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-16k",
        temperature=0.1,
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
    system_prompt ="""Sos Asistente experto en Juego de Tronos en español. Recibirás una pregunta y un potencial contexto proveniente de Juego de Tronos. Vas a analizar la pregunta, y luego vas a elaborar una respuesta basandote en el contexto. Sé puntual y conciso con la respuesta. """
    context = preprocess_context([i['preprocess_content'] for i in context])
    prompt = f"""Contexto: \n```{context}```\nMensaje: {query}"""
    messages = [
        {'role': 'assistant', 'content': system_prompt},
        {'role': 'user', 'content': prompt},
    ]
    
    response = ollama.chat(
        'mistral:7b',
        messages=messages

    )
    #logging.info(f"Ollama response: {response['choices'][0]# ['message']['content']}")
    logging.info(response)
    return response['message']['content']