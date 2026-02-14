import os
import tempfile
import hashlib
import shutil
import base64
import re
import uuid
from typing import List
import pickle
import numpy as np
from rank_bm25 import BM25Okapi

import fitz
import pandas as pd

from dotenv import load_dotenv
from groq import Groq

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
# Load API Key
# =====================================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)


# =====================================
# FastAPI App
# =====================================
app = FastAPI(
    title="EPC RAG API",
    description="RAG Backend with Hybrid Search (BM25 + Vector) & Specialized Agents for Dates/Clauses",
    version="2.5"  # UPDATED VERSION
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
# Normalize Language Input
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
# Date Extraction with Regex (OPTIMIZED FOR DATE AGENT)
# =====================================
def extract_dates_from_chunks(chunks):
    """
    Extract dates from chunks using comprehensive regex patterns.
    Returns list of (date_string, source_reference, page_number) tuples.
    """
    # Comprehensive date patterns covering common legal document formats
    date_patterns = [
        # DD Month YYYY (e.g., 15 June 2023, 1st January 2024)
        r'\b(\d{1,2})(?:st|nd|rd|th)?\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b',
        # Month DD, YYYY (e.g., June 15, 2023, March 1st, 2024)
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b',
        # DD/MM/YYYY or DD-MM-YYYY (e.g., 15/06/2023, 12-05-2022)
        r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b',
        # YYYY-MM-DD (ISO format)
        r'\b(\d{4})-(\d{1,2})-(\d{1,2})\b',
        # Month YYYY (e.g., March 2023, December 2024)
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b',
        # Month-DD-YYYY (e.g., June-15-2023)
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December)-(\d{1,2})-(\d{4})\b',
        # Dated [Month DD, YYYY] (common in legal docs)
        r'\bdated\s+(?:the\s+)?(\d{1,2})(?:st|nd|rd|th)?\s+(January|February|March|April|May|June|July|August|September|October|November|December),?\s+(\d{4})\b',
        # Effective from [Date]
        r'\beffective\s+(?:from\s+)?(?:the\s+)?(\d{1,2})(?:st|nd|rd|th)?\s+(January|February|March|April|May|June|July|August|September|October|November|December),?\s+(\d{4})\b',
    ]
    
    extracted_dates = []
    month_map = {
        'january': 'January', 'february': 'February', 'march': 'March', 'april': 'April',
        'may': 'May', 'june': 'June', 'july': 'July', 'august': 'August',
        'september': 'September', 'october': 'October', 'november': 'November', 'december': 'December'
    }
    
    for chunk in chunks:
        # Extract source and page from chunk metadata
        source_match = re.search(r'\[Source:\s*([^|]+)\s*\|\s*Page:\s*(\d+)\]', chunk)
        source = source_match.group(1).strip() if source_match else "Unknown"
        page = int(source_match.group(2)) if source_match else 999
        
        for pattern in date_patterns:
            matches = re.finditer(pattern, chunk, re.IGNORECASE)
            for match in matches:
                # Normalize date string to standard format
                date_str = match.group(0)
                
                # Skip likely false positives (years before 1900 or section numbers)
                if len(date_str) == 4:
                    year = int(date_str)
                    if year < 1900 or year > 2100:
                        continue
                
                # Skip dates that look like version numbers or amounts
                if re.search(r'\b(v|version|ver)\s*\d', chunk, re.IGNORECASE):
                    continue
                if re.search(r'\$\s*\d', chunk) or re.search(r'\b\d+\s*(dollars|USD|INR)', chunk, re.IGNORECASE):
                    continue
                
                extracted_dates.append((date_str, source, page))
    
    return extracted_dates


# =====================================
# Find Best Date Match (RANKING LOGIC)
# =====================================
def find_best_date_match(question, extracted_dates):
    """
    Rank dates by:
    1. Page priority (page 1 = highest, early pages favored)
    2. Frequency in document
    3. Semantic alignment with question keywords
    Returns best matching date with source reference or None
    """
    if not extracted_dates:
        return None
    
    # Count frequency of each unique date
    date_freq = {}
    for date_str, source, page in extracted_dates:
        date_freq[date_str] = date_freq.get(date_str, 0) + 1
    
    # Score each date occurrence
    scored_dates = []
    question_lower = question.lower()
    
    # Question intent keywords for semantic alignment
    start_keywords = ["start", "commencement", "beginning", "effective", "initiation", "kick-off", "inception"]
    end_keywords = ["completion", "final", "end", "finish", "termination", "expiry", "conclusion", "deadline", "due"]
    milestone_keywords = ["milestone", "payment", "deliverable", "review", "inspection", "acceptance"]
    
    question_intent = "neutral"
    if any(kw in question_lower for kw in start_keywords):
        question_intent = "start"
    elif any(kw in question_lower for kw in end_keywords):
        question_intent = "end"
    elif any(kw in question_lower for kw in milestone_keywords):
        question_intent = "milestone"
    
    for date_str, source, page in extracted_dates:
        score = 0
        
        # Page priority scoring (earlier pages = higher score)
        if page == 1:
            score += 100
        elif page <= 3:
            score += 80
        elif page <= 5:
            score += 60
        elif page <= 10:
            score += 40
        elif page <= 20:
            score += 20
        else:
            score += max(5, 100 - page)  # Diminishing returns for later pages
        
        # Frequency bonus (repeated dates are more likely correct)
        score += date_freq[date_str] * 15
        
        # Semantic alignment bonus
        if question_intent == "start" and page <= 5:
            score += 30  # Start dates typically in early pages
        elif question_intent == "end" and page > 20:
            score += 30  # End dates typically in later pages
        elif question_intent == "milestone":
            # Milestones can be anywhere, but mid-document is common
            if 5 <= page <= 50:
                score += 20
        
        # Question proximity bonus (if date appears near question-relevant terms)
        if any(kw in question_lower for kw in ["payment", "invoice", "fee"]):
            if any(term in date_str.lower() for term in ["payment", "invoice"]):
                score += 25
        
        scored_dates.append((date_str, source, page, score))
    
    # Sort by score descending
    scored_dates.sort(key=lambda x: x[3], reverse=True)
    
    # Return top match if confidence is sufficient
    if scored_dates and scored_dates[0][3] > 30:  # Minimum confidence threshold
        best_date, best_source, best_page, best_score = scored_dates[0]
        return f"{best_date} [Source: {best_source} | Page: {best_page}]"
    
    return None


# =====================================
# Question Classifier for Agent Routing
# =====================================
def classify_question_type(question: str) -> str:
    """
    Lightweight classifier to route questions to specialized agents.
    Returns: 'date', 'clause', or 'general'
    """
    question_lower = question.lower().strip()
    
    # Date-focused keywords (prioritize date agent)
    date_keywords = [
        "date", "deadline", "timeline", "when", "by when", "due date", "completion date",
        "effective date", "expiry", "valid until", "commencement", "termination date",
        "milestone", "schedule", "timeframe", "period", "duration", "month", "year",
        "day", "week", "quarter", "anniversary", "renewal date","dated"
    ]
    
    # Clause-focused keywords (prioritize clause agent)
    clause_keywords = [
        "clause", "section", "article", "term", "condition", "obligation", "liability",
        "indemnity", "penalty", "breach", "termination", "force majeure", "warranty",
        "representation", "covenant", "provision", "sub-clause", "paragraph", "stipulation",
        "jurisdiction", "governing law", "arbitration", "dispute", "remedy", "entitlement"
    ]
    
    # Count keyword matches (simple but effective)
    date_score = sum(1 for kw in date_keywords if kw in question_lower)
    clause_score = sum(1 for kw in clause_keywords if kw in question_lower)
    
    # Route based on highest score with tie-breaking
    if date_score > clause_score:
        return "date"
    elif clause_score > date_score:
        return "clause"
    else:
        # Default to clause agent for legal documents (more common use case)
        return "clause" if any(kw in question_lower for kw in clause_keywords) else "general"


# =====================================
# Embedding Model
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
# Read PDF WITH FILENAME METADATA
# =====================================
def read_pdf(file_path, filename: str):
    doc = fitz.open(file_path)
    text = ""
    for i, page in enumerate(doc, start=1):
        page_text = page.get_text()
        if page_text.strip():
            text += f"\n[Source: {filename} | Page: {i}]\n{page_text}"
    return text


# =====================================
# Read Excel WITH FILENAME METADATA
# =====================================
def read_excel(file_path, filename: str):
    df = pd.read_excel(file_path)
    return f"[Source: {filename}]\n{df.to_string(index=False)}"


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
# Create Session DB + BM25 Index
# =====================================
def create_session_db(chunks, session_id):
    db_path = f"vector_db/session_{session_id}"
    db = FAISS.from_texts(chunks, embeddings)
    db.save_local(db_path)
    
    chunks_path = f"vector_db/session_{session_id}_chunks.pkl"
    with open(chunks_path, 'wb') as f:
        pickle.dump(chunks, f)
    
    tokenized_chunks = [chunk.split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    bm25_path = f"vector_db/session_{session_id}_bm25.pkl"
    with open(bm25_path, 'wb') as f:
        pickle.dump(bm25, f)
    
    return db_path


# =====================================
# Hybrid Retrieval (BM25 + Vector) with RRF
# =====================================
# =====================================
# Hybrid Retrieval (BM25 + Vector) with RRF + Score Filtering
# =====================================
def hybrid_retrieval(vector_db, session_id, question, k=12, score_threshold=0.45):

    try:
        chunks_path = f"vector_db/session_{session_id}_chunks.pkl"
        bm25_path = f"vector_db/session_{session_id}_bm25.pkl"

        # -------------------------------
        # FAISS with Similarity Scores
        # -------------------------------
        faiss_docs_with_scores = vector_db.similarity_search_with_score(
            question,
            k=k * 2  # get extra and filter later
        )

        # Filter weak vector matches
        filtered_faiss = [
            (doc.page_content, rank + 1)
            for rank, (doc, score) in enumerate(faiss_docs_with_scores)
            if score >= score_threshold
        ]

        # Keep only top-k after filtering
        filtered_faiss = filtered_faiss[:k]

        # Fallback if nothing passes filter
        if not filtered_faiss:
            filtered_faiss = [
                (doc.page_content, rank + 1)
                for rank, (doc, _) in enumerate(faiss_docs_with_scores[:k])
            ]

        # -------------------------------
        # If BM25 not available → FAISS only
        # -------------------------------
        if not (os.path.exists(chunks_path) and os.path.exists(bm25_path)):
            return [text for text, _ in filtered_faiss]

        # -------------------------------
        # Load BM25
        # -------------------------------
        with open(chunks_path, "rb") as f:
            all_chunks = pickle.load(f)

        with open(bm25_path, "rb") as f:
            bm25 = pickle.load(f)

        tokenized_query = question.split()

        bm25_scores = bm25.get_scores(tokenized_query)

        top_bm25_indices = np.argsort(bm25_scores)[::-1][:k]

        bm25_results = [
            (all_chunks[i], rank + 1)
            for rank, i in enumerate(top_bm25_indices)
        ]

        # -------------------------------
        # RRF Fusion
        # -------------------------------
        rrf_scores = {}

        for text, rank in filtered_faiss:
            rrf_scores[text] = rrf_scores.get(text, 0) + 1 / (60 + rank)

        for text, rank in bm25_results:
            rrf_scores[text] = rrf_scores.get(text, 0) + 1 / (60 + rank)

        sorted_results = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [text for text, _ in sorted_results[:k]]

    except Exception:
        # Safe fallback
        docs = vector_db.similarity_search(question, k=k)
        return [doc.page_content for doc in docs]



# =====================================
# Language Detection
# =====================================
def detect_language(text):
    try:
        lang = detect(text)
        return lang if lang in SUPPORTED_LANG_CODES else "en"
    except:
        return "en"


# =====================================
# Clean Text for TTS
# =====================================
def clean_text_for_tts(text):
    text = re.sub(r"[•●▪■◆►▶➤*]", "", text)
    text = re.sub(r"[|=_#`~<>]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.replace(":", " ").replace(";", " ").replace("/", " ")
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
# Specialized Agent Prompts
# =====================================
def ask_llm(context, question, language, agent_type="general"):
    lang_map = {"en": "English", "hi": "Hindi", "mr": "Marathi", "ar": "Arabic", "gu": "Gujarati"}
    lang_name = lang_map.get(language, "English")
    
    if agent_type == "date":
        system_prompt = f"""You are a DATE EXTRACTION SPECIALIST agent for EPC contracts.
        
CRITICAL RULES:
- Extract ONLY explicit dates, deadlines, timelines, and time periods mentioned in context
- NEVER infer or calculate dates not explicitly stated
- Format all dates as: "DD Month YYYY" (e.g., "15 June 2025")
- For durations: "X days/weeks/months/years from [trigger event]"
- ALWAYS include source reference: [Source: filename | Page: X]
- If no date found: "No specific date mentioned in the document"
- NEVER use asterisk (*)
- Answer ONLY in {lang_name}

Context:
{context}

Question:
{question}

Provide ONLY the precise date/timeline information with source references."""
    
    elif agent_type == "clause":
        system_prompt = f"""You are a LEGAL CLAUSE SPECIALIST agent for EPC contracts.
        
CRITICAL RULES:
- Extract ONLY explicit clauses, terms, conditions, obligations, and liabilities
- NEVER interpret or summarize legal meaning - quote exact wording where possible
- ALWAYS include clause numbers/references (e.g., "Clause 8.2", "Section 4.1(a)")
- ALWAYS include source reference: [Source: filename | Page: X]
- For complex clauses: break into bullet points WITHOUT asterisks (use dashes or numbers)
- If clause not found: "No relevant clause found in the document"
- NEVER use asterisk (*)
- Answer ONLY in {lang_name}

Context:
{context}

Question:
{question}

Provide precise clause text with exact references and source locations."""
    
    else:  # general agent (original behavior)
        system_prompt = f"""You are a professional legal compliance assistant.

RULES:
- Use ONLY the given context
- Do NOT guess or add external knowledge
- If information is missing: "Not mentioned in the document"
- ALWAYS reference source document and page from context markers (e.g., [Source: contract.pdf | Page: 5])
- NEVER use asterisk (*)

LANGUAGE:
Answer ONLY in {lang_name}

Context:
{context}

Question:
{question}

Provide a precise, professional answer with explicit source references."""
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": system_prompt}],
        temperature=0.1,
        max_tokens=700
    )
    return response.choices[0].message.content


# =====================================
# Reformat Previous Answer
# =====================================
def reformat_answer(previous_answer, user_request, language):
    lang_map = {"en": "English", "hi": "Hindi", "mr": "Marathi", "ar": "Arabic", "gu": "Gujarati"}
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
- Preserve all source references
- Maintain professional format
- NEVER use asterisk (*)
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=500
    )
    return response.choices[0].message.content


# =====================================
# Health Check
# =====================================
@app.get("/")
def home():
    return {"status": "EPC RAG API Running (Hybrid Search + Date/Clause Agents + Date Extraction)"}


# =====================================
# Upload MULTIPLE Files (MAX 5)
# =====================================
@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    if len(files) > 5:
        raise HTTPException(
            status_code=400, 
            detail="Maximum 5 files allowed per upload. Please upload fewer files."
        )
    
    session_id = str(uuid.uuid4())
    temp_dir = tempfile.mkdtemp()
    combined_text = ""
    uploaded_filenames = []
    
    try:
        for file in files:
            if not (file.filename.endswith(".pdf") or file.filename.endswith((".xlsx", ".xls"))):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Unsupported file format: {file.filename}. Only PDF and Excel files allowed."
                )
            
            temp_path = os.path.join(temp_dir, file.filename)
            with open(temp_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            
            if file.filename.endswith(".pdf"):
                text = read_pdf(temp_path, file.filename)
            else:
                text = read_excel(temp_path, file.filename)
            
            combined_text += text + "\n\n"
            uploaded_filenames.append(file.filename)
        
        chunks = chunk_text(combined_text)
        db_path = create_session_db(chunks, session_id)
        
        SESSION_STORE[session_id] = {
            "db_path": db_path,
            "files": uploaded_filenames,
            "last_answer": None
        }
        
        return {
            "message": f"Successfully processed {len(uploaded_filenames)} document(s) with hybrid search and agent routing",
            "session_id": session_id,
            "uploaded_files": uploaded_filenames,
            "total_chunks": len(chunks)
        }
    
    finally:
        shutil.rmtree(temp_dir)


# =====================================
# Text Question Model
# =====================================
class TextQuestion(BaseModel):
    session_id: str = Field(..., example="a1b2c3d4-5678-90ef-ghij-klmnopqrstuv")
    question: str = Field(..., example="What is the project completion date?")
    language: str = Field(
        default=None,
        description="Output language (codes: en/hi/mr/ar/gu or names)",
        example="marathi"
    )


# =====================================
# Ask Question (TEXT) - WITH DATE EXTRACTION OPTIMIZATION
# =====================================
@app.post("/ask")
async def ask_question(data: TextQuestion):
    if data.session_id not in SESSION_STORE:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid session '{data.session_id}'. Please upload documents again."
        )
    
    session_data = SESSION_STORE[data.session_id]
    db_path = session_data["db_path"]
    
    try:
        vector_db = FAISS.load_local(
            db_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load document database: {str(e)}")

    question = data.question
    
    output_language = None
    if data.language:
        output_language = normalize_language(data.language)
        if not output_language:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language '{data.language}'. Supported: english, hindi, marathi, arabic, gujarati (or codes: en, hi, mr, ar, gu)"
            )
    
    if not output_language:
        output_language = detect_language(question)

    table_keywords = ["table", "tabular", "rows and columns", "table format"]
    show_as_table = any(word in question.lower() for word in table_keywords)

    keywords = ["convert", "simple", "simplify", "summarize", "format", "reformat", "explain again", "translate"]

    # AGENT ROUTING
    agent_type = "general"
    if not (any(word in question.lower() for word in keywords) and session_data["last_answer"]):
        agent_type = classify_question_type(question)
    
    # =====================================
    # DATE AGENT OPTIMIZATION: Regex extraction BEFORE hybrid retrieval
    # =====================================
    if agent_type == "date":
        try:
            chunks_path = f"vector_db/session_{data.session_id}_chunks.pkl"
            if os.path.exists(chunks_path):
                with open(chunks_path, 'rb') as f:
                    all_chunks = pickle.load(f)
                
                # Extract dates using regex patterns
                extracted_dates = extract_dates_from_chunks(all_chunks)
                
                # Find best matching date based on question context
                best_date_match = find_best_date_match(question, extracted_dates)
                
                if best_date_match:
                    # Return date directly WITHOUT LLM call (performance optimization)
                    answer = f"The relevant date is: {best_date_match}"
                    voice = text_to_speech(answer, output_language)
                    
                    return {
                        "language": output_language,
                        "question": question,
                        "answer": answer,
                        "audio_base64": voice,
                        "show_as_table": show_as_table,
                        "session_id": data.session_id,
                        "referenced_files": session_data["files"],
                        "agent_used": agent_type,
                        "retrieval_method": "regex_extraction"  # NEW FIELD: Shows optimization used
                    }
        except Exception as e:
            # Silent fallback to hybrid retrieval on ANY error (production-safe)
            pass
    
    # =====================================
    # EXISTING LOGIC FOR ALL OTHER CASES (clause, general, or date fallback)
    # =====================================
    if any(word in question.lower() for word in keywords) and session_data["last_answer"]:
        answer = reformat_answer(session_data["last_answer"], question, output_language)
    else:
        top_chunks = hybrid_retrieval(vector_db, data.session_id, question, k=8)
        context = "\n\n".join(top_chunks)
        answer = ask_llm(context, question, output_language, agent_type=agent_type)

    session_data["last_answer"] = answer
    voice = text_to_speech(answer, output_language)

    return {
        "language": output_language,
        "question": question,
        "answer": answer,
        "audio_base64": voice,
        "show_as_table": show_as_table,
        "session_id": data.session_id,
        "referenced_files": session_data["files"],
        "agent_used": agent_type,
        "retrieval_method": "hybrid_rag"  # Default method
    }


# =====================================
# Voice Base64 Model
# =====================================
class VoiceBase64(BaseModel):
    session_id: str = Field(..., example="a1b2c3d4-5678-90ef-ghij-klmnopqrstuv")
    audio_base64: str = Field(..., example="UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=")
    language: str = Field(
        default=None,
        description="Output language (codes: en/hi/mr/ar/gu or names)",
        example="marathi"
    )


# =====================================
# Ask Question (VOICE) - WITH DATE EXTRACTION OPTIMIZATION
# =====================================
@app.post("/ask-voice-base64")
async def ask_voice_base64(data: VoiceBase64):
    if data.session_id not in SESSION_STORE:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid session '{data.session_id}'. Please upload documents again."
        )
    
    session_data = SESSION_STORE[data.session_id]
    db_path = session_data["db_path"]
    
    try:
        vector_db = FAISS.load_local(
            db_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load document database: {str(e)}")

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
            try:
                question = recognizer.recognize_google(audio_data)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Speech recognition failed: {str(e)}")

        output_language = None
        if data.language:
            output_language = normalize_language(data.language)
            if not output_language:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported language '{data.language}'. Supported: english, hindi, marathi, arabic, gujarati (or codes: en, hi, mr, ar, gu)"
                )
        
        if not output_language:
            output_language = detect_language(question)

        table_keywords = ["table", "tabular", "rows and columns", "table format"]
        show_as_table = any(word in question.lower() for word in table_keywords)

        # AGENT ROUTING
        agent_type = classify_question_type(question)
        
        # =====================================
        # DATE AGENT OPTIMIZATION: Regex extraction BEFORE hybrid retrieval
        # =====================================
        if agent_type == "date":
            try:
                chunks_path = f"vector_db/session_{data.session_id}_chunks.pkl"
                if os.path.exists(chunks_path):
                    with open(chunks_path, 'rb') as f:
                        all_chunks = pickle.load(f)
                    
                    # Extract dates using regex patterns
                    extracted_dates = extract_dates_from_chunks(all_chunks)
                    
                    # Find best matching date based on question context
                    best_date_match = find_best_date_match(question, extracted_dates)
                    
                    if best_date_match:
                        # Return date directly WITHOUT LLM call (performance optimization)
                        answer = f"The relevant date is: {best_date_match}"
                        voice = text_to_speech(answer, output_language)
                        
                        return {
                            "language": output_language,
                            "recognized_text": question,
                            "answer": answer,
                            "audio_base64": voice,
                            "show_as_table": show_as_table,
                            "session_id": data.session_id,
                            "referenced_files": session_data["files"],
                            "agent_used": agent_type,
                            "retrieval_method": "regex_extraction"  # NEW FIELD: Shows optimization used
                        }
            except Exception as e:
                # Silent fallback to hybrid retrieval on ANY error (production-safe)
                pass
        
        # =====================================
        # EXISTING LOGIC FOR ALL OTHER CASES (clause, general, or date fallback)
        # =====================================
        top_chunks = hybrid_retrieval(vector_db, data.session_id, question, k=12)
        context = "\n\n".join(top_chunks)
        answer = ask_llm(context, question, output_language, agent_type=agent_type)

        session_data["last_answer"] = answer
        voice = text_to_speech(answer, output_language)

        return {
            "language": output_language,
            "recognized_text": question,
            "answer": answer,
            "audio_base64": voice,
            "show_as_table": show_as_table,
            "session_id": data.session_id,
            "referenced_files": session_data["files"],
            "agent_used": agent_type,
            "retrieval_method": "hybrid_rag"  # Default method
        }

    finally:
        shutil.rmtree(temp_dir)