import streamlit as st
import asyncio
from chatbot import ChatBot

INDEX_LOCAL_PATH = 'index_juego_de_tronos_chunk_512.ann'
MAPPING_PATH = '../data/jdt_chunks_sentences_512.json'
INDEX_OPENAI_PATH = 'index_juego_de_tronos_chunks_512_openai.ann'


def init_chatbot(llm_engine: str):
    """
    Inicializa el chatbot con el motor deseado (e.g., 'local', 'openai').
    """
    try:
        index = INDEX_LOCAL_PATH if llm_engine == 'local' else INDEX_OPENAI_PATH
        return ChatBot(llm_engine, index, MAPPING_PATH)
    except Exception as e:
        st.error(f"Error al inicializar el ChatBot: {e}")
        return None


def display_messages():
    """Muestra el historial de mensajes almacenados en la sesión."""
    with st.container(border=True):
        if len(st.session_state['messages']) > 0:
            for role, text in st.session_state["messages"]:
                if role == "User":
                    st.markdown(f"**:speaking_head_in_silhouette: Tú:** {text}")
                else:
                    st.markdown(f"**:robot_face: Bot:** {text}")
        else:
            text = ("Soy un asistente especializado en la novela Canción de Hielo y Fuego "
                    "de G.G.R. Martin. Preguntame algo relacionado con el libro.\nPor ejemplo: "
                    "\n- ¿Quién es Eddard Stark?\n- Qué animal decora el estandarte de la Casa Baratheon?")
            st.markdown(f"**:robot_face: Bot:** {text}")


def main():
    st.set_page_config(page_title="Juego de Tronos Chatbot", layout="wide")

    if 'llm_choice' not in st.session_state:
        st.session_state.llm_choice = None
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Menú de selección de motor LLM
    if st.session_state.llm_choice is None:
        st.title("Selecciona el motor de LLM")
        llm_choice = st.selectbox(
            "Selecciona el motor de LLM:", ["local", "openai"])
        if st.button("Confirmar selección"):
            st.session_state.llm_choice = llm_choice
            st.rerun()

    # Mostrar configuración y permitir cambio
    if st.session_state.llm_choice is not None:
        st.sidebar.title("Configuración")
        st.sidebar.write(
            f"Motor LLM seleccionado: {st.session_state.llm_choice}")
        if st.sidebar.button("Cambiar selección de motor LLM"):
            st.session_state.llm_choice = None
            st.session_state['messages'] = []
            st.rerun()

        # UI LAYOUT
        st.title("Juego de Tronos Chatbot")

        # Contenedor para entrada de texto y botón juntos
        col1, col2 = st.columns([4, 1])
        with col1:
            user_input = st.text_input("Escribe tu mensaje aquí:", key="user_input")
        with col2:
            send_button = st.button("Enviar", key="send_button", )

        # Mostrar historial
        display_messages()

        if send_button:
            if user_input.strip() == "":
                st.warning("Por favor, ingresa un texto.")
            else:
                st.session_state["messages"].append(("User", user_input))
                chatbot = init_chatbot(st.session_state["llm_choice"])

                if chatbot:
                    try:
                        with st.spinner("Generando respuesta..."):
                            response = asyncio.run(chatbot.get_response(user_input))
                        st.session_state["messages"].append(("Bot", response))
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error al obtener respuesta: {e}")


if __name__ == "__main__":
    main()