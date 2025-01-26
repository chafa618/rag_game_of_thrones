#!/bin/bash

set -e


# Función para verificar y crear carpetas si no existen
check_and_create_dir() {
    local dir=$1
    if [ ! -d "$dir" ]; then
        echo "La carpeta $dir no existe. Creándola..."
        mkdir -p "$dir"
    else
        echo "La carpeta $dir ya existe. Saltando la creación."
    fi
}

# Verificar y crear las carpetas necesarias
check_and_create_dir "data"
check_and_create_dir "indexes"
check_and_create_dir "models"

# Ruta a los archivos de datos y modelos
PDF_FILE="data/Juego de tronos - Canción de hielo y fuego 1 (1) copy.pdf"
CHUNKS_FILE="data/jdt_chunks_sentences_512.json"
INDEX_NAME_LOCAL="indexes/index_juego_de_tronos_chunk_512.ann"
INDEX_NAME_OPENAI="indexes/index_juego_de_tronos_chunks_512_openai.ann"

# Función para verificar si un archivo existe
check_and_run() {
    local file=$1
    local command=$2
    if [ -f "$file" ]; then
        echo "El archivo $file ya existe. Saltando el comando."
    else
        echo "El archivo $file no existe. Ejecutando el comando."
        eval "$command"
    fi
}

echo "Chequeando Dependencias..."
check_and_run "$CHUNKS_FILE" 
python -c "
from utils import build_chunks_json
build_chunks_json()
"

# Entrenar el modelo de clasificación de dominio
echo "Entrenando el modelo de clasificación de dominio..."
python dc_training.py --train
echo "Modelo de clasificación de dominio entrenado y guardado."

# Crear el índice de embeddings usando SentenceTransformer
echo "Creando el índice de embeddings usando SentenceTransformer..."
check_and_run "$INDEX_NAME_LOCAL" "python embeddings_es.py --model sentence-transformer --chunks_file $CHUNKS_FILE --index_name $INDEX_NAME_LOCAL"
echo "Índice de embeddings creado usando SentenceTransformer."


if [ -z "$OPENAI_API_KEY" ]; then
  echo "Error: La variable de entorno OPENAI_API_KEY no está configurada."
  echo "Por favor, configura la variable de entorno OPENAI_API_KEY antes de ejecutar el script."
  exit 1
fi

# Crear el índice de embeddings usando OpenAI
echo "Creando el índice de embeddings usando OpenAI..."
check_and_run "$INDEX_NAME_OPENAI" "python embeddings_es.py --model openai --chunks_file $CHUNKS_FILE --index_name $INDEX_NAME_OPENAI"
echo "Índice de embeddings creado usando OpenAI."

echo "Entrenamiento y creación de índices completados."