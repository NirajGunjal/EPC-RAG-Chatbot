import os
import tempfile
import hashlib
import shutil
import base64
import re
import uuid
import requests

import fitz
import pandas as pd

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ================= Voice Libraries =================
import speech_recognition as sr
from pydub import AudioSegment
from gtts import gTTS
from langdetect import detect
# ===================================================


# =====================================
# OLLAMA CONFIG
# =====================================
OLLAMA_HOST = "110.227.253.23"
OLLAMA_PORT = 11434
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
QWEN_MODEL_NAME = "mistral-small3.2:latest"


# =====================================
# FastAPI App
# =====================================
app = FastAPI(
    title="EPC RAG API",
    description="RAG Backend for EPC Contract Analysis with Multi-Language Support",
    version="2.1"
)


# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================
# Session Management
# =====================================
SESSION_STORE = {}
os.makedirs("vector_db", exist_ok=True)


# =====================================
# Supported Languages Mapping
# =====================================
LANGUAGE_MAP = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
    "ar": "Arabic",
    "gu": "Gujarati",

    "english": "English",
    "hindi": "Hindi",
    "marathi": "Marathi",
    "arabic": "Arabic",
    "gujarati": "Gujarati",

    "हिंदी": "Hindi",
    "मराठी": "Marathi",
    "ગુજરાતી": "Gujarati",
    "العربية": "Arabic"
}

SUPPORTED_LANG_CODES = ["en", "hi", "mr", "ar", "gu"]


# =====================================
# Normalize Language
# =====================================
def normalize_language(lang_input: str) -> str:

    if not lang_input:
        return None
    
    key = lang_input.strip().lower()

    for code, name in LANGUAGE_MAP.items():
        if key == code.lower() or key == name.lower():
            return [k for k, v in LANGUAGE_MAP.items() if v == name and len(k) == 2][0]

    return None


# =====================================
# Embeddings
# =====================================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# =====================================
# File Hash
# =====================================
def get_file_hash(file_path):

    hasher = hashlib.md5()

    with open(file_path, "rb") as f:
        hasher.update(f.read())

    return hasher.hexdigest()


# =====================================
# Read PDF
# =====================================
def read_pdf(file_path):

    doc = fitz.open(file_path)

    text = ""

    for i, page in enumerate(doc, start=1):

        page_text = page.get_text()

        if page_text.strip():
            text += f"\n[Page {i}]\n{page_text}"

    return text


# =====================================
# Read Excel
# =====================================
def read_excel(file_path):

    df = pd.read_excel(file_path)

    return df.to_string(index=False)


# =====================================
# Chunk Text
# =====================================
def chunk_text(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", ";"]
    )

    return splitter.split_text(text)


# =====================================
# Create / Load DB
# =====================================
def create_or_load_db(chunks, file_hash):

    db_path = f"vector_db/{file_hash}"

    if os.path.exists(db_path):
        return db_path

    db = FAISS.from_texts(chunks, embeddings)
    db.save_local(db_path)

    return db_path


# =====================================
# Detect Language
# =====================================
def detect_language(text):

    try:
        lang = detect(text)

        if lang in SUPPORTED_LANG_CODES:
            return lang

        return "en"

    except:
        return "en"


# =====================================
# Clean Text
# =====================================
def clean_text_for_tts(text):

    text = re.sub(r"[•●▪■◆►▶➤*]", "", text)
    text = re.sub(r"[|=_#`~<>]", " ", text)
    text = re.sub(r"\s+", " ", text)

    text = text.replace(":", " ")
    text = text.replace(";", " ")
    text = text.replace("/", " ")

    return text.strip()


# =====================================
# Text To Speech
# =====================================
def text_to_speech(text, lang):

    clean_text = clean_text_for_tts(text)

    temp_dir = tempfile.mkdtemp()

    audio_path = os.path.join(temp_dir, "answer.mp3")

    tts = gTTS(text=clean_text, lang=lang)
    tts.save(audio_path)

    with open(audio_path, "rb") as f:
        audio_base64 = base64.b64encode(f.read()).decode()

    shutil.rmtree(temp_dir)

    return audio_base64


# =====================================
# Ask LLM (OLLAMA)
# =====================================
def ask_llm(context, question, language):

    lang_map = {
        "en": "English",
        "hi": "Hindi",
        "mr": "Marathi",
        "ar": "Arabic",
        "gu": "Gujarati"
    }

    lang_name = lang_map.get(language, "English")

    prompt = f"""
You are a professional legal compliance assistant.

RULES:
- Use ONLY the given context
- Do NOT guess
- Do NOT add external knowledge
- If information is missing, say: "Not mentioned in the document"
- Always give page/clause reference
- NEVER use asterisk (*)

LANGUAGE:
Answer ONLY in {lang_name}

Context:
{context}

Question:
{question}

Give a clear and professional answer.
"""

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": QWEN_MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 700
                }
            },
            timeout=300
        )

        response.raise_for_status()

        result = response.json()

        return result.get("response", "").strip()

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Local LLM Error: {str(e)}"
        )


# =====================================
# Reformat Answer (OLLAMA)
# =====================================
def reformat_answer(previous_answer, user_request, language):

    lang_map = {
        "en": "English",
        "hi": "Hindi",
        "mr": "Marathi",
        "ar": "Arabic",
        "gu": "Gujarati"
    }

    lang_name = lang_map.get(language, "English")

    prompt = f"""
Reformat the answer based on user request.

Previous Answer:
{previous_answer}

User Request:
{user_request}

Rules:
- Do NOT add new information
- Only reformat or translate to {lang_name}
- Keep professional format
- NEVER use asterisk (*)
"""

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": QWEN_MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 500
                }
            },
            timeout=300
        )

        response.raise_for_status()

        result = response.json()

        return result.get("response", "").strip()

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Local LLM Error: {str(e)}"
        )


# =====================================
# Health Check
# =====================================
@app.get("/")
def home():

    return {"status": "EPC RAG API Running"}


# =====================================
# Upload
# =====================================
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):

    session_id = str(uuid.uuid4())
    
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)

    try:

        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        file_hash = get_file_hash(temp_path)

        if file.filename.endswith(".pdf"):
            text = read_pdf(temp_path)

        elif file.filename.endswith((".xlsx", ".xls")):
            text = read_excel(temp_path)

        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        chunks = chunk_text(text)

        db_path = create_or_load_db(chunks, file_hash)

        SESSION_STORE[session_id] = {
            "db_path": db_path,
            "file_hash": file_hash,
            "last_answer": None
        }

        return {
            "message": "Document processed successfully",
            "session_id": session_id,
            "file_hash": file_hash
        }

    finally:
        shutil.rmtree(temp_dir)


# =====================================
# Text Model
# =====================================
class TextQuestion(BaseModel):

    session_id: str
    question: str
    language: str = Field(default=None)


# =====================================
# Ask Text
# =====================================
@app.post("/ask")
async def ask_question(data: TextQuestion):

    if data.session_id not in SESSION_STORE:

        raise HTTPException(
            status_code=400,
            detail="Invalid session. Upload again."
        )

    session_data = SESSION_STORE[data.session_id]

    db_path = session_data["db_path"]

    vector_db = FAISS.load_local(
        db_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    question = data.question

    output_language = None

    if data.language:

        output_language = normalize_language(data.language)

        if not output_language:

            raise HTTPException(status_code=400, detail="Unsupported language")

    if not output_language:

        output_language = detect_language(question)

    keywords = [
        "convert", "simple", "simplify", "summarize",
        "format", "reformat", "explain again", "translate"
    ]

    if any(word in question.lower() for word in keywords) and session_data["last_answer"]:

        answer = reformat_answer(
            session_data["last_answer"],
            question,
            output_language
        )

    else:

        docs = vector_db.similarity_search(question, k=8)

        context = "\n\n".join([d.page_content for d in docs])

        answer = ask_llm(context, question, output_language)

    session_data["last_answer"] = answer

    voice = text_to_speech(answer, output_language)

    return {
        "language": output_language,
        "question": question,
        "answer": answer,
        "audio_base64": voice,
        "show_as_table": False,
        "session_id": data.session_id
    }


# =====================================
# Voice Model
# =====================================
class VoiceBase64(BaseModel):

    session_id: str
    audio_base64: str
    language: str = Field(default=None)


# =====================================
# Ask Voice
# =====================================
@app.post("/ask-voice-base64")
async def ask_voice_base64(data: VoiceBase64):

    if data.session_id not in SESSION_STORE:

        raise HTTPException(
            status_code=400,
            detail="Invalid session. Upload again."
        )

    session_data = SESSION_STORE[data.session_id]

    db_path = session_data["db_path"]

    vector_db = FAISS.load_local(
        db_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    temp_dir = tempfile.mkdtemp()

    try:

        audio_bytes = base64.b64decode(data.audio_base64)

        audio_path = os.path.join(temp_dir, "input_audio")

        with open(audio_path, "wb") as f:
            f.write(audio_bytes)

        sound = AudioSegment.from_file(audio_path)

        wav_path = os.path.join(temp_dir, "audio.wav")

        sound.export(wav_path, format="wav")

        recognizer = sr.Recognizer()

        with sr.AudioFile(wav_path) as source:

            audio_data = recognizer.record(source)

            question = recognizer.recognize_google(audio_data)

        output_language = None

        if data.language:

            output_language = normalize_language(data.language)

        if not output_language:

            output_language = detect_language(question)

        docs = vector_db.similarity_search(question, k=8)

        context = "\n\n".join([d.page_content for d in docs])

        answer = ask_llm(context, question, output_language)

        session_data["last_answer"] = answer

        voice = text_to_speech(answer, output_language)

        return {
            "language": output_language,
            "recognized_text": question,
            "answer": answer,
            "audio_base64": voice,
            "show_as_table": False,
            "session_id": data.session_id
        }

    finally:

        shutil.rmtree(temp_dir)
