import gradio as gr
from engine import run, load_data, load_index

_, chunk_id_mapping = load_data('../data/juego_de_tronos_chunks_300.json')
index = load_index('../index_juego_de_tronos_chunk_300.ann', 768)



# Función para manejar las interacciones
def chatbot_response(user_message):
    # Lógica básica del chatbot
    if "hola" in user_message.lower():
        return "¡Hola! ¿En qué puedo ayudarte?"
    elif "adiós" in user_message.lower():
        return "Adiós, ¡que tengas un gran día!"
    else:
        llm_response, _ = run(user_message, index, chunk_id_mapping, model_type=model_type)
        return llm_response

# Interfaz de Gradio
with gr.Blocks() as demo:
    model_type = gr.Radio(["local", "openai"], label="Selecciona el modelo:")
    #model_type = model_type.value.lower()
    gr.Markdown("## Chatbot con Gradio")
    chat_box = gr.Chatbot()
    user_input = gr.Textbox(label="Escribe tu mensaje:")
    send_button = gr.Button("Enviar")

    def chat_flow(user_input, chat_history):
        response = chatbot_response(user_input)
        chat_history.append(("user", user_input))
        chat_history.append(("chatbot", response))
        return chat_history, ""

    send_button.click(chat_flow, inputs=[user_input, chat_box], outputs=[chat_box, user_input],)

# Ejecuta la aplicación
demo.launch()
