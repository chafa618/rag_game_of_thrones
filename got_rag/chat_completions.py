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


class OpenAiCompletionWrapper:
    def __init__(self, system_prompt=None):
        self.client = openai.Client(api_key=os.environ["OPENAI_API_KEY"])
        self.history = []

    def get_answer(self, query, context):
        messages = [
                {
                    "role": "developer", "content": "Sos un sistema de RAG especializado en el libro de G.R.R. Martin Cancion de hielo y fuego. Voy a proveerte de una pregunta y una serie de párrafos del libro como contexto. Usando esa información vas a redactar una respuesta acorde a la pregunta. Omitirás toda mención a la palabra contexto y a la saga de television de Juego de tronos."
                },
                {
                    "role": "user",
                    "content": f"Pregunta: ```{query}```\nContexto:\n```{context}```"
                },
            ]
        context = preprocess_context([i['preprocess_content'] for i in context])
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=messages
        )
        
        llm_answer = response.choices[0].message.content
        
        messages.append(llm_answer)
        self.history.append(messages)
        return llm_answer
        


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
        {'role': 'system', 'content': system_prompt},
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


class CommonsOllamaCompletionWrapper:
    """
    Clase que encapsula la lógica para obtener respuestas breves
    e insta a continuar la conversación sobre 'Canción de Hielo y Fuego'.
    """

    def __init__(self,
                 model_name="qwen:1.8b",
                 temperature=0.2,
                 num_ctx=256):
        """
        Inicializa la clase con un modelo y parámetros de generación.
        
        Args:
            model_name (str): Nombre del modelo a usar
            temperature (float): Control de aleatoriedad del texto (0.0 determinista - 1.0 creativo).
            num_ctx (int): Tamaño del contexto (tokens).
        """
        self.model_name = model_name
        self.temperature = temperature
        self.num_ctx = num_ctx

        self.system_prompt = """Sos un asistente virtual que puede responder consultas sobre Canción de Hielo y Fuego. Tu función es continuar la charla y conducir al usuario hacia el libro. Responderás con un mensaje corto e invitarás siempre 
        a que se te pregunte algo relacionado con el libro Cancion de Hielo y Fuego de G.R.R. Martin. 
        Responde siempre en español y usando hasta 10 palabras.
        Por ejemplo:
            - Si te dicen 'Hola', responderás con algo similar a 'Hola, soy un asistente virtual especializado en el libro Canción de Hielo y Fuego. ¿Qué deseas saber?'
            - Si te dicen 'gracias' o 'genial', responderás con algo similar a 'Qué bien. Sigamos, ¿te puedo ayudar con algo más sobre Cancion de Hielo y Fuego?'
            - Si te preguntan algo que no tiene relacion al libro, simplemente dirás que no puedes responder eso, e invitarás al otro a que te haga preguntas sobre Canción de Hielo y Fuego.
            """

    def get_answer(self, query):
        """
        Obtiene la respuesta a partir de la consulta (query), usando la API de Ollama.
        Retorna el contenido del mensaje (string).
        
        Args:
            query (str): Mensaje del usuario.
        
        Returns:
            str: Respuesta generada por el modelo.
        """
        messages = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': query},
        ]

        generate_params = {
            'model': self.model_name,
            'options': ollama.Options(
                temperature=self.temperature,
                num_ctx=self.num_ctx
            ),
            'messages': messages,
            'stream': False
        }

        response = ollama.chat(**generate_params)
        logging.info({'ollama_generic_answer': response})

        return response['message']['content']

        
class OllamaRAGCompletionWrapper:
    """
    Clase que encapsula la lógica para generar respuestas usando un modelo de Ollama,
    recibiendo un query y un conjunto de contextos relacionados.
    """

    def __init__(self,
                 model_name="mistral:7b",
                 temperature=0.1,
                 num_ctx=512):
        """
        Inicializa la clase con un modelo y parámetros de generación.

        Args:
            model_name (str): Nombre del modelo a usar en Ollama.
            temperature (float): Control de aleatoriedad en el texto generado.
            num_ctx (int): Tamaño máximo de contexto (tokens).
        """
        self.model_name = model_name
        self.temperature = temperature
        self.num_ctx = num_ctx

        self.system_prompt = (
            "Sos Asistente experto en Juego de Tronos en español. "
            "Recibirás una pregunta y una serie de potenciales contextos proveniente del libro de Juego de Tronos. "
            "Vas a analizar la pregunta y vas a elaborar una respuesta acorde basándote solamente en la "
            "información proporcionada en los contextos. Omitirás en la respuesta toda mención literal a la palabra contexto y evitarás mencionar a la serie de televisión. Una vez que tengas la respuesta, la envias."
            "Si no sabes qué decir, simplemente responde que no tenés información relevante al respecto."
        )

    def preprocess_context(self, context_list):
        """
        Ejemplo de método para preprocesar los fragmentos del contexto.
        Ajusta la lógica según tus necesidades (el original no se muestra aquí).
        """
        processed = []
        for text in context_list:

            clean_text = " ".join(text.split())
            processed.append(clean_text)

        return " ".join(processed)

    def get_answer(self, query, context):
        """
        Genera la respuesta en base a una pregunta (query) y un contexto proveniente del libro.

        Parámetros:
        ---------
            query (str): Pregunta del usuario.
            context (list[dict]): Lista de diccionarios con la clave "preprocess_content"
                                  que contiene el texto relevante del libro.

        Returns:
            str: Respuesta generada por el modelo de Ollama.
        """
        logging.info(f'Query Ollama: {query}')
        context_texts = [item['preprocess_content'] for item in context]
        processed_context = self.preprocess_context(context_texts)

        prompt_user = (
            f"Contexto: \n```{processed_context}```\n"
            f"Mensaje: {query}"
        )
        messages = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': prompt_user},
        ]
        generate_params = {
            'model': self.model_name,
            'options': ollama.Options(
                temperature=self.temperature,
                num_ctx=self.num_ctx
            ),
            'messages': messages,
            'stream': False
        }

        response = ollama.chat(**generate_params)
        logging.info({'ollama_rag_answer': response})

        return response['message']['content']