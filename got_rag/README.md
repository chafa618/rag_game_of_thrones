# Chatbot de Juego de Tronos

Este proyecto implementa un chatbot que responde preguntas sobre el libro "Juego de Tronos" de George R. R. Martin, utilizando un enfoque de Recuperación de Información Asistida por Generación (RAG) con un modelo de lenguaje.

## Propuesta de Solución

El proyecto consiste en tres partes principales:
1. Un chatbot
2. Un sistema de Recuperación de Información Asistida por Generación (RAG)
3. Procesado de datos y armado de una base de datos vectorial

### 1. Procesado de Datos
- A partir del PDF del libro, se genera un archivo JSON con los datos procesados.
- Se utiliza el archivo JSON preprocesado como input en lugar del PDF original para mayor comodidad.
- La aplicación final mantiene la funcionalidad para agregar archivos adicionales.

### 2. RAG
- El sistema RAG se encarga de recuperar los fragmentos más relevantes del libro y generar respuestas utilizando un modelo de lenguaje.

### 3. Chatbot
- El chatbot responde preguntas sobre el libro utilizando el sistema RAG.
- Si la pregunta no está relacionada con el libro, el chatbot responderá que no tiene información.

## Requisitos Previos

1. Archivo PDF con el texto de "Juego de Tronos".
2. Librerías Python para manejo de PDFs, embeddings, y un modelo LLM.
3. Ollama and OpenAi

## Instalación

1. Clona este repositorio:
    ```sh
    https://github.com/chafa618/rag_game_of_thrones.git
    cd got_rag
    ```

2. Crea y activa un entorno virtual:
    ```sh
    python -m venv venv
    source venv/bin/activate  # En Windows usa `venv\Scripts\activate`
    ```

3. Instala las dependencias:
    ```sh
    pip install -r requirements.txt
    ```

4. Asegurarse de tener Ollama instalado. https://ollama.com/  

4. Configura las variables de entorno:
    - Crea un archivo [.env](http://_vscodecontentref_/0) en la raíz del proyecto y agrega tu clave de API de OpenAI:
        ```
        OPENAI_API_KEY=tu_clave_de_api
        ```

5. Descarga los modelos de Ollama:
    ```sh
    ollama pull qwen2:1.5b-instruct
    ```

## Uso

Para ejecutar la applicacion, correr: 



### 1. Procesar el PDF y generar los chunks

Ejecuta el siguiente comando para procesar el PDF y generar los chunks:
```sh
python -c "
from utils import build_chunks_json
build_chunks_json()
```

2. Entrenar el modelo de clasificación de dominio
Ejecuta el siguiente comando para entrenar el modelo de clasificación de dominio:
```
python got_rag/dc_training.py --train
```

3. Crear el índice de embeddings
Ejecuta el siguiente comando para crear el índice de embeddings usando SentenceTransformer:

```
python got_rag/embeddings_es.py --model sentence-transformer --chunks_file data/jdt_chunks_sentences_512.json --index_name indexes/index_juego_de_tronos_chunk_512.ann
```

- 3.2 Para crear el índice de embeddings usando OpenAI, ejecuta:

```
python got_rag/embeddings_es.py --model openai --chunks_file data/jdt_chunks_sentences_512.json --index_name indexes/index_juego_de_tronos_chunks_512_openai.ann
```


4. Ejecutar el Chatbot
Para ejecutar el chatbot, utiliza Streamlit:

```
streamlit run got_rag/st_experiment.py
```


### Estructura del Proyecto
- got_rag: Contiene los scripts principales del proyecto.
    - chatbot.py: Implementación del chatbot.
    - chat_completions.py: Lógica para obtener respuestas usando diferentes modelos de lenguaje.
    - dc_training.py: Entrenamiento del modelo de clasificación de dominio.
    - embeddings_es.py: Creación y manejo de índices de embeddings.
    - utils.py: Utilidades para procesar el PDF y generar los chunks.
    - st_experiment.py: Script para ejecutar el chatbot usando Streamlit.
- data/: Carpeta para almacenar los archivos de datos procesados.
- indexes/: Carpeta para almacenar los índices de embeddings.
- models/: Carpeta para almacenar los modelos entrenados.
- requirements.txt: Lista de dependencias del proyecto.


### Descripcion del proceso:
- Primero me encontré con los obstaculos propios del proceso de segmentación. Después de probar algunas iteraciones con diferentes tamaños de chunks, el que funcionó mejor fue el de 512. Esta version es la utilizada en la entrega. Chequear notebook con el proceso de splitting en data/splitting.ipynb. Guardo en un archivo json los chunks para no tener que estar cosntantemente levantando el pdf al ejecutar el chatbot y sobre esto construyo los siguientes pasos.
- Me quede sin tiempo de implementar un metodo de evaluación y así tomar una decisión más acertada en terminos de accuracy, no obstante creo que en función de los objetivos de la POC, podría ser suficiente.
- La unica limpieza sobre los datos fue la eliminación de espacios multiples y guiones. Me percaté un poco in media res de las particularidades del español (no elimino las tildes o lowerizo el texto, por ejemplo), pero incluiria una serie de funciones de normalización del texto en todos los stages del proceso. Que a fin de cuentas sólo es más exhaustiva en el módulo del clasificador de dominio. 
- Además, hay un tema con el parseo de los pdfs y la division natural del libro en paginas. Creo que el parser se apega a esta división y corta de algun modo con el desarrollo 'humano' del proceso de lectura, quien al llegar al final de una pagina, es capaz de ligar la informacion con la pagina siguiente. Una punta para la que tirar.
- Una vez dividido el texto, utilicé embeddings_es.create_index() para generar un index (indexes/), el que se utilizará más adelante para el RAG
- Una vez que tuve los embeddings, me puse a trabajar en el chatbot en si y me encontré con la dificultad de la restricción, entonces quise implementar algo solamente usando openai y un modelo local usando ollama. 
- De tal forma que me plantee que se decida por UI el modelo encargado de procesar un mensaje que pertenece al dominio de GoT. Como primer paso se generan embeddings para todos los chunks y dichos vectores se almacenan en un index. Se puede usar un modelo de SentenceTransformer local o se puede usar los embeddings de OpenAi. Ambos ya fueron generados y están en la carpeta homónima. Luego desde la parte de procesamiento, se vectoriza la query del usuario y se obtienen los n vecinos más cercanos, candidatos que se utilizan en la generación de la respuesta final. La generación se ejecutará usando el modelo seleccionado.


### La idea del flujo del bot es la siguiente: 
    - Dado un mensaje de usuario, el bot clasifica y decide si pertenece o no al dominio GOT. 
        - En caso negativo, un LLM local se encarga de responder. 
        - En caso afirmativo, se instancia el sistema de RAG (El usuario decide si el motor LLM a utilizar es ollama u openai), se procesa la query y se genera la respuesta. 

