import streamlit as st
import requests


FASTAPI_URL = "http://127.0.0.1:8000"
# ============================
# 1. Simulación de clasificadores
# ============================
def classify_message(message: str) -> str:
    """
    Función de ejemplo para clasificar el mensaje del usuario.
    Dependiendo del texto, retornará 'LLM' o 'RAG'.
    
    En un caso real, podrías usar un modelo de clasificación 
    entrenado, o alguna heurística más avanzada.
    """
    response = requests.post(f"{FASTAPI_URL}/classify_domain", json={"query": message, "context": ""})
    if response.status_code == 200:
        return response.json().get("answer", "No response")
    else:
        return "Error calling DC"

# ============================
# 2. Simulación de llamada a un LLM
# ============================
def call_commons_llm(user_input: str) -> str:
    """
    Función que llama a un LLM (por ej. GPT) para generar una respuesta.
    En un escenario real, aquí harías la llamada a la API de tu modelo (OpenAI, HuggingFace, etc.).
    """
    response = requests.post(f"{FASTAPI_URL}/local_common", json={"query": user_input, "context": ""})
    if response.status_code == 200:
        return response.json().get("answer", "No response")
    else:
        return "Error calling LLM API"

# ============================
# 3. Simulación de llamada a RAG
# ============================
def call_rag(user_input: str, model: str = "local") -> str:
    """
    Función que llama a un flujo RAG (por ej. búsqueda en vector store + LLM).
    En un escenario real, harías:
      1) Búsqueda de documentos relevantes.
      2) Generación de respuesta basada en esos documentos.
    """
    if model == 'local':
        response = requests.post(f"{FASTAPI_URL}/ollama", json={"query": user_input, "context": ""})
    else: 
        response = requests.post(f"{FASTAPI_URL}//openai", json={"query": user_input, "context": ""})
    if response.status_code == 200:
        return response.json().get("answer", "No response")
    else:
        return "Error calling RAG API"

# ============================
# 4. Aplicación de Streamlit
# ============================
def main():
    st.title("Chatbot Sobre Canción de hielo y Fuego")
    llm_engine = st.selectbox('LLM', ['openai', 'local'])
    
    # Inicializa la sesión para almacenar el historial de la conversación.
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Muestra el historial de conversación.
    for role, text in st.session_state["messages"]:
        if role == "User":
            st.markdown(f"**Tú:** {text}")
        else:
            st.markdown(f"**Bot:** {text}")

    # Campo de entrada del usuario.
    user_input = st.text_input("Escribe tu mensaje:", "")

    # Botón para enviar el mensaje.
    if st.button("Enviar"):
        if user_input.strip():
            # 1. Agregamos el mensaje del usuario al historial.
            st.session_state["messages"].append(("User", user_input))

            # 2. Clasificamos el mensaje.
            classification = classify_message(str(user_input))

            # 3. Llamamos al sistema adecuado (LLM o RAG).
            if classification == "commons":
                response = call_commons_llm(user_input)
            else:
                response = call_rag(user_input, llm_engine)

            # 4. Agregamos la respuesta del bot al historial.
            st.session_state["messages"].append(("Bot", response))

            # Refrescamos la página para mostrar la conversación actualizada.
            st.rerun()

if __name__ == "__main__":
    main()
