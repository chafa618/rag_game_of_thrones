import fitz  # PyMuPDF
import json
import pdf4llm

from llama_index.core.node_parser import SentenceSplitter


def parse_pdf_to_md(path=None):
    md_read = pdf4llm.LlamaMarkdownReader()
    path = path if path else "Juego de tronos - Canción de hielo y fuego 1 (1) copy.pdf"
    data = md_read.load_data()
    return data


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def chunk_document(document, max_size=300, overlap=10):
    tokens = document.split()
    chunks = []
    tokens_count = len(tokens)
    i = 0
    while i < tokens_count:
        chunk = tokens[i:i + max_size]
        chunks.append(' '.join(chunk))
        i += max_size - overlap
    return chunks


def preprocess_text(text):
    text = text.strip()
    text = text.replace("_", "")
    text = text.replace("\n", " ")
    text = text.replace("  ", " ")
    text = text.replace("-----", "")
    return text


def save_json(data, file):
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def chunk_text(data, size, overlap):
    splitter = SentenceSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
    )
    nodes = splitter.get_nodes_from_documents(data)
    return nodes


def process_nodes(nodes):
    chunks_sentences = []
    for id, node in enumerate(nodes):
        page = node.metadata['page']
        text = preprocess_text(node.text)
        chunk = {'chunk_id': id, 'page': page, 'preprocess_content': text, 'content': node.text}
        chunks_sentences.append(chunk)
    return chunks_sentences
