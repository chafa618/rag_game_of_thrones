import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
import os
from dotenv import load_dotenv
from utils import chunk_document
import requests
import faiss
import spacy

load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]

nlp = spacy.load("es_core_news_sm")

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path, )
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def index_text_v1(text):
    sentences = chunk_document(text, max_size=50, overlap=5) # text.split('. ')
    print(sentences)
    vectorizer = TfidfVectorizer()
    vectorizer_matrix = vectorizer.fit_transform(sentences)
    return sentences, vectorizer, vectorizer_matrix


def sentence_tokenizer_spacy(text):
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

def index_text(text):

    sentences = chunk_document(text, max_size=300, overlap=50)
    vectorizer = TfidfVectorizer()
    vectorized_sentences = vectorizer.fit_transform(sentences).toarray()
    
    # Create FAISS index
    dimension = vectorized_sentences.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectorized_sentences)
    
    return sentences, vectorizer, index

def save_faiss_index(index, file_path):
    faiss.write_index(index, file_path)

def load_faiss_index(file_path):
    return faiss.read_index(file_path)

def get_relevant_text(query, sentences, vectorizer, index):
    query_vec = vectorizer.transform([query]).toarray()
    _, relevant_idx = index.search(query_vec, 1)
    return sentences[relevant_idx[0][0]]


def get_relevant_text_v1(query, sentences, vectorizer, vectorizer_matrix):
    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, vectorizer_matrix).flatten()
    relevant_idx = similarity.argmax()
    return sentences[relevant_idx]

def get_relevant_text(query, sentences, vectorizer, index):
    query_vec = vectorizer.transform([query]).toarray()
    _, relevant_idx = index.search(query_vec, 1)
    return sentences[relevant_idx[0][0]]

def get_answer_from_openai(query, context):
    client = openai.Client(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-16k",
        temperature=0.1,
        messages=[
            {"role": "developer", "content": "Sos un sistema de RAG. Voy a proveerte de una pregunta o mensaje y su contexto. Necesito que respondas a la pregunta usando esa información. Si el contexto y la pregunta no estuvieran relacionados, solo responde que no hay relacion entre ellos, sin más detalles."},
            {
                "role": "user",
                "content": f"Contexto: {context}\nPregunta: {query}"
            },
        ]
    )
    return response.choices[0].message.content

def chatbot(pdf_path, query):
    text = extract_text_from_pdf(pdf_path)
    # sentences = text.split('. ')
    # vectorizer = TfidfVectorizer().fit_transform(sentences)
    sentences, vectorizer, vectorizer_matrix = index_text(text)
    context = get_relevant_text(query, sentences, vectorizer, 
                                vectorizer_matrix)
    print(context)
    answer = get_answer_from_openai(query, context)
    return answer

def get_answer_from_local_model(query, context):
    import json
    url = 'http://localhost:11434/api/generate'  
    payload = {
        'model': "mistral",
        'prompt': f"""Sos Asistente especializado en Juego de Tronos. Voy a proveerte de una pregunta o mensaje y su contexto. Necesito que elabores una respuesta basandote en el contexto. Si el contexto y la pregunta no estuvieran relacionados, solo responde que no hay relacion entre ellos, sin más detalles.
        
        Contexto: ```{context}```
         
        Pregunta: {query}
        
        """,
        'temperature': 0.1
    }
    
    response = requests.post(url, json=payload)
    print(type(response))
    try:
        # Split the response text into individual JSON objects
        response_text = response.text.strip().split('\n')
        responses = [json.loads(line) for line in response_text]
        
        # Combine the 'response' fields from each JSON object
        combined_response = ''.join([resp['response'] for resp in responses])
    except (requests.exceptions.JSONDecodeError, KeyError) as e:
        print("Error processing response:", e)
        print("Response text:", response.text)
        return "Error: Unable to process response from local model."

    return combined_response


def chatbot_local(pdf_path, query):
    text = extract_text_from_pdf(pdf_path)
    preprocessed_text = text # preprocess_text(text)
    sentences, vectorizer, vectorized_sentences = index_text(preprocessed_text)
    preprocessed_query = query # preprocess_text(query)
    context = get_relevant_text(preprocessed_query, sentences, vectorizer, vectorized_sentences)
    print(context)
    answer = get_answer_from_local_model(preprocessed_query, context)
    return answer



def chatbot_faiss(pdf_path, query, index_file_path=None):
    if index_file_path and os.path.exists(index_file_path):
        index = load_faiss_index(index_file_path)
        sentences = chunk_document(extract_text_from_pdf(pdf_path), max_size=300, overlap=50)
        vectorizer = TfidfVectorizer()
        vectorizer.fit(sentences)
    else:
        text = extract_text_from_pdf(pdf_path)
        preprocessed_text = text
        sentences, vectorizer, index = index_text(preprocessed_text)
        if index_file_path:
            save_faiss_index(index, index_file_path)
    
    preprocessed_query = query
    context = get_relevant_text(preprocessed_query, sentences, vectorizer, index)
    print(context)
    answer = get_answer_from_local_model(preprocessed_query, context)
    return answer

