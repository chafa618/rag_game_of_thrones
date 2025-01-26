import logging
from typing import Optional
import openai
import ollama
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from chatbot_ import ChatBot

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()

chatbot = ChatBot()