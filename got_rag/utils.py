import json
import pdf4llm
import logging
from llama_index.core.node_parser import SentenceSplitter

#logging config



def parse_pdf_to_md(path="data/Juego de tronos - Canción de hielo y fuego 1 (1) copy.pdf"):
    #path = path if path else "data/Juego de tronos - Canción de hielo y fuego 1 (1) copy.pdf"
    logging.info(f'Parseando {path}...')
    md_read = pdf4llm.LlamaMarkdownReader()
    data = md_read.load_data(path)
    return data


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


def chunk_text(data, size=512, overlap=50):
    logging.info('Generando Nodos desde PDF ...')
    splitter = SentenceSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
    )
    nodes = splitter.get_nodes_from_documents(data)
    return nodes


def process_nodes(nodes):
    logging.info('Generando chunks desde nodos ...')
    chunks_sentences = []
    for id, node in enumerate(nodes):
        page = node.metadata['page']
        text = preprocess_text(node.text)
        chunk = {'chunk_id': id, 'page': page, 'preprocess_content': text, 'content': node.text}
        chunks_sentences.append(chunk)
    return chunks_sentences


def build_chunks_json():
    pdf_md = parse_pdf_to_md()
    nodes = chunk_text(pdf_md)
    chunks_sentences = process_nodes(nodes)
    save_json(chunks_sentences, "data/jdt_chunks_sentences_512.json")
