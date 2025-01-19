import gradio as gr

# Función para manejar las interacciones
def chatbot_response(user_message):
    # Lógica básica del chatbot
    if "hola" in user_message.lower():
        return "¡Hola! ¿En qué puedo ayudarte?"
    elif "adiós" in user_message.lower():
        return "Adiós, ¡que tengas un gran día!"
    else:
        return "Lo siento, no entiendo tu mensaje."

# Interfaz de Gradio
with gr.Blocks() as demo:
    gr.Markdown("## Chatbot con Gradio")
    chat_box = gr.Chatbot()
    user_input = gr.Textbox(label="Escribe tu mensaje:")
    send_button = gr.Button("Enviar")

    def chat_flow(user_input, chat_history):
        response = chatbot_response(user_input)
        chat_history.append(("Tú", user_input))
        chat_history.append(("Chatbot", response))
        return chat_history, ""

    send_button.click(chat_flow, inputs=[user_input, chat_box], outputs=[chat_box, user_input],)

# Ejecuta la aplicación
demo.launch()
