import os
import uuid
import fitz  # PyMuPDF
import re
import json
from tqdm import tqdm
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import unicodedata
import nltk
from collections.abc import MutableMapping
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import tldextract
from urllib.robotparser import RobotFileParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from playwright.sync_api import sync_playwright
import hashlib
import logging
from urls_config import urls_to_scrape
from datetime import datetime

# ==== Initial Setup ====
nltk.download("punkt")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ==== Configuration ====
CHROMA_PERSIST_DIR = "./chroma_db"
MAIN_DIR = "POSH"
PRODUCT_SUBDIR = "Products"
DIVISIONS_SUBDIR = "Divisions"
INFO_COLLECTION_NAME = "information_embeddings"
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en"

# ==== Load Embedding Model ====
model = SentenceTransformer(EMBEDDING_MODEL_NAME)
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# ==== Initialize Chroma ====
info_vectordb = Chroma(
    persist_directory=CHROMA_PERSIST_DIR,
    embedding_function=embedding_model,
    collection_name=INFO_COLLECTION_NAME
)
info_collection = info_vectordb._collection

CHUNK_CONFIG = {
    "min_len": 800,
    "max_len": 1200,
    "overlap": 150
}

# ==== Utility Functions ====

def clean_text_pdf(text: str, keep_unicode=True) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r"[ \t]+", " ", text)
    if not keep_unicode:
        text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text.strip()

def clean_chunk_content(text: str) -> str:
    text = re.sub(r"([.\-_]){3,}", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def merge_small_chunks(chunks: List[Dict], min_length=200) -> List[Dict]:
    if not chunks:
        return chunks
    merged_chunks = []
    buffer_chunk = None
    for chunk in chunks:
        content = chunk['content']
        if len(content) < min_length:
            if buffer_chunk is None and merged_chunks:
                merged_chunks[-1]['content'] += " " + content
            elif buffer_chunk is None:
                buffer_chunk = chunk
            else:
                buffer_chunk['content'] += " " + content
        else:
            if buffer_chunk:
                small_chunk = buffer_chunk
                small_chunk['content'] += " " + content
                merged_chunks.append(small_chunk)
                buffer_chunk = None
            else:
                merged_chunks.append(chunk)
    if buffer_chunk and len(buffer_chunk['content']) >= min_length:
        merged_chunks.append(buffer_chunk)
    return merged_chunks

def remove_redundant_chunks(chunks: List[str]) -> List[str]:
    seen = set()
    unique_chunks = []
    for chunk in chunks:
        chunk_clean = chunk.strip()
        if chunk_clean and chunk_clean not in seen:
            seen.add(chunk_clean)
            unique_chunks.append(chunk_clean)
    return unique_chunks

def deduplicate_chunks(chunks: List[Dict]) -> List[Dict]:
    seen = set()
    deduped = []
    for chunk in chunks:
        content_hash = hashlib.md5(chunk['content'].encode('utf-8')).hexdigest()
        if content_hash not in seen:
            seen.add(content_hash)
            deduped.append(chunk)
    return deduped

def fallback_recursive_chunks(text: str, min_len=800, max_len=1200, overlap=150) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_len,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = [c.strip() for c in splitter.split_text(text) if len(c.strip()) >= min_len]
    return chunks

def is_heading(line: str) -> bool:
    line = line.strip()
    if len(line) > 100:
        return False
    return (
        bool(re.match(r"^(CHAPTER|SECTION|Annex(ure)?)\s+\w+", line, re.I)) or
        bool(re.match(r"^\d+(\.\d+)+\s+.+", line)) or
        bool(re.match(r"^[IVXLCDM]+\.\s+.+", line)) or
        (line.isupper() and len(line.split()) <= 10)
    )

def split_by_headings(text: str, min_len=800, max_len=1200) -> List[str]:
    lines = text.splitlines()
    chunks, buffer, current_chunk = [], [], []
    def flush_chunk(chunk_lines: List[str]):
        joined = "\n".join(chunk_lines).strip()
        if not joined:
            return
        if min_len <= len(joined) <= max_len:
            chunks.append(joined)
        elif len(joined) > max_len:
            chunks.extend(fallback_recursive_chunks(joined, min_len, max_len))
        else:
            buffer.extend(chunk_lines)
    for line in lines:
        line = clean_text_pdf(line)
        if not line:
            continue
        if is_heading(line) and current_chunk:
            flush_chunk(current_chunk)
            current_chunk = []
        current_chunk.append(line)
    flush_chunk(current_chunk)
    if buffer:
        buffer_text = "\n".join(buffer).strip()
        if len(buffer_text) >= min_len:
            chunks.append(buffer_text)
        elif chunks:
            chunks[-1] += "\n" + buffer_text
        else:
            chunks.append(buffer_text)
    return [c.strip() for c in chunks if len(c.strip()) >= min_len]

# ==== PDF Extraction and Chunking ====

def extract_tables_from_pdf(pdf_path: str) -> List[Dict[str, str]]:
    tables = []
    with fitz.open(pdf_path) as doc:
        doc_id = os.path.splitext(os.path.basename(pdf_path))[0]
        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict")["blocks"]
            table_blocks = []
            current_table = []
            last_y = None
            for block in sorted(blocks, key=lambda b: b["bbox"][1]):
                if block.get("type") != 0 or "lines" not in block:
                    continue
                y_start = block["bbox"][1]
                if last_y is not None and y_start - last_y > 50 and current_table:
                    table_blocks.append(current_table)
                    current_table = []
                lines = block["lines"]
                multiline_spans = any(len(line["spans"]) > 1 for line in lines if "spans" in line)
                if len(lines) > 1 or multiline_spans:
                    current_table.append(block)
                last_y = block["bbox"][3]
            if current_table:
                table_blocks.append(current_table)
            for table_idx, blocks in enumerate(table_blocks):
                table_text = []
                for block in blocks:
                    for line in block["lines"]:
                        row = []
                        for span in line["spans"]:
                            text = clean_text_pdf(span["text"])
                            if text:
                                row.append(text.replace("|", "/"))
                        if row:
                            table_text.append(" | ".join(row))
                if not table_text:
                    continue
                table_content = f"Table {table_idx + 1} (Page {page_num + 1}):\n" + "\n".join(table_text)
                tables.append({
                    "content": table_content,
                    "metadata": {
                        "doc_id": doc_id,
                        "chunk_id": f"table-{page_num}-{table_idx}",
                        "doc_name": os.path.basename(pdf_path),
                        "chunk_type": "table",
                        "category": "information",
                        "page_number": page_num + 1
                    }
                })
    return tables

def extract_text_from_pdf_layout_aware(pdf_path: str) -> List[Dict]:
    texts = []
    with fitz.open(pdf_path) as doc:
        doc_id = os.path.splitext(os.path.basename(pdf_path))[0]
        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict")["blocks"]
            blocks = sorted(blocks, key=lambda b: b["bbox"][1])
            page_text_lines = []
            for block in blocks:
                if block.get("type") != 0 or "lines" not in block:
                    continue
                for line in block["lines"]:
                    line_text_parts = []
                    for span in line["spans"]:
                        span_text = clean_text_pdf(span["text"])
                        if span_text:
                            line_text_parts.append(span_text)
                    if line_text_parts:
                        page_text_lines.append(" ".join(line_text_parts))
            page_text = "\n".join(page_text_lines).strip()
            if page_text:
                texts.append({
                    "content": page_text,
                    "metadata": {
                        "doc_id": doc_id,
                        "doc_name": os.path.basename(pdf_path),
                        "page_number": page_num + 1,
                        "chunk_type": "text",
                        "category": "information"
                    }
                })
    return texts

def split_text_chunks(text_chunks: List[Dict], min_len=800, max_len=1200) -> List[Dict]:
    result_chunks = []
    for chunk in text_chunks:
        content = chunk["content"]
        metadata = chunk["metadata"]
        pieces = split_by_headings(content, min_len=min_len, max_len=max_len)
        if not pieces or sum(len(p) for p in pieces) < 0.5 * len(content):
            pieces = fallback_recursive_chunks(content, min_len, max_len)
        pieces = remove_redundant_chunks(pieces)
        for i, piece in enumerate(pieces):
            new_metadata = metadata.copy()
            new_metadata["chunk_id"] = f"text-{metadata['page_number']}-{i}"
            result_chunks.append({
                "content": piece,
                "metadata": new_metadata
            })
    return result_chunks

def extract_and_chunk_pdf(pdf_path: str, min_len=800, max_len=1200, overlap=150) -> List[Dict]:
    logging.info(f"Processing PDF: {pdf_path}")
    tables = extract_tables_from_pdf(pdf_path)
    table_pages = set(t["metadata"]["page_number"] for t in tables)
    text_pages = extract_text_from_pdf_layout_aware(pdf_path)
    filtered_text_pages = text_pages
    text_chunks = split_text_chunks(filtered_text_pages, min_len=min_len, max_len=max_len)
    all_chunks = tables + text_chunks
    logging.info(f"Extracted {len(all_chunks)} chunks from {pdf_path}")
    return all_chunks

# Products Section
def extract_text_from_pdf(pdf_path: str) -> str:
    with fitz.open(pdf_path) as doc:
        return "\n".join(page.get_text() for page in doc)

def chunk_product_pdf_by_sections(pdf_path: str, file_name: str) -> List[Dict]:
    product_name = os.path.splitext(file_name)[0]
    full_text = extract_text_from_pdf(pdf_path)
    full_text = unicodedata.normalize("NFKC", full_text).replace('\u00a0', ' ')
    lines = [line.strip() for line in full_text.splitlines() if line.strip()]
    section_titles = [
        "Contact",
        "Statistical Presentation and Description",
        "Institutional Mandate",
        "Quality Management",
        "Accuracy and Reliability",
        "Timeliness",
        "Comparability",
        "Statistical Processing",
        "Metadata Update"
    ]
    section_indices = []
    for idx, line in enumerate(lines):
        for title in section_titles:
            if line.lower() == title.lower():
                section_indices.append((title, idx))
                break
    if len(section_indices) < 9:
        logging.warning(f"Skipping {file_name} — expected 9 clean headings, found {len(section_indices)}")
        return []
    sections = []
    for i in range(len(section_indices)):
        start_idx = section_indices[i][1]
        end_idx = section_indices[i + 1][1] if i + 1 < len(section_indices) else len(lines)
        section_lines = lines[start_idx:end_idx]
        sections.append("\n".join(section_lines).strip())
    grouped_chunks = []
    for i in range(0, 9, 3):
        chunk_sections = sections[i:i + 3]
        chunk_text = f"{product_name}\n\n" + "\n\n".join(chunk_sections)
        grouped_chunks.append(chunk_text)
    chunk_dicts = []
    for idx, chunk in enumerate(grouped_chunks):
        chunk_dicts.append({
            "content": chunk,
            "metadata": {
                "doc_id": product_name,
                "chunk_id": idx,
                "doc_name": "https://www.mospi.gov.in",
                "category": "information"
            }
        })
    return chunk_dicts

# For divisions
def extract_division_name(text):
    match = re.search(r"1\.\s*Division Name\s*:?\s*(.*?)\n", text, re.DOTALL)
    return match.group(1).strip() if match else "Unknown"

def extract_sections(text):
    pattern = re.compile(r"(?:^|\n)(\s*\d+\.\s+.*?)(?=\n\s*\d+\.\s+|\Z)", re.DOTALL)
    sections = pattern.findall(text)
    return [sec.strip() for sec in sections if sec.strip()]

def process_structured_division_pdf(file_path: str, file_name: str) -> List[Dict]:
    doc_id = os.path.splitext(file_name)[0]
    raw_text = extract_text_from_pdf(file_path)
    if not raw_text.strip():
        logging.warning(f"No extractable text in {file_name}. Skipping.")
        return []
    division_name = extract_division_name(raw_text)
    sections = extract_sections(raw_text)
    if not sections:
        logging.warning(f"No valid sections in {file_name}. Skipping.")
        return []
    chunks = []
    for chunk_id, section in enumerate(sections):
        content = section.strip()
        if not content:
            continue
        enriched_chunk = f"[Division: {division_name}]\n{content}"
        chunks.append({
            "content": clean_chunk_content(enriched_chunk),
            "metadata": {
                "doc_id": division_name,
                "chunk_id": chunk_id,
                "doc_name": "MoSPI Divisions",
                "chunk_type": "text",
                "category": "information"
            }
        })
    return chunks

# OCR Part

import cv2
from PIL import Image
from skimage.io import imread
from paddleocr import PaddleOCR
import pytesseract

logging.basicConfig(level=logging.INFO)

# ---------- PDF to Images ----------
def convert_pdf_to_images(pdf_path: str, output_dir: str = "/tmp/pdf_pages", dpi: int = 350):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    os.makedirs(output_dir, exist_ok=True)
    zoom = dpi / 72
    magnify = fitz.Matrix(zoom, zoom)
    images = []
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=magnify)
        img_path = os.path.join(
            output_dir, 
            f"{os.path.splitext(os.path.basename(pdf_path))[0]}_page_{page_num+1}.png"
        )
        pix.save(img_path)
        logging.info(f"Saved page {page_num+1}/{len(doc)} as image: {img_path}")
        images.append(img_path)
    doc.close()
    logging.info(f"Total {len(images)} pages converted to images")
    return images

# ---------- Preprocessing ----------
def preprocess_image(image_path):
    logging.info(f"Preprocessing image: {image_path}")
    try:
        osd = pytesseract.image_to_osd(Image.open(image_path), output_type=pytesseract.Output.DICT)
        rotate_angle = osd.get('rotate', 0)
        if rotate_angle != 0:
            img = Image.open(image_path)
            rotated = img.rotate(-rotate_angle, expand=True)
            rotated.save(image_path)
            logging.info(f"Image rotated by {rotate_angle} degrees using OSD")
    except Exception as e:
        logging.warning(f"Rotation detection failed: {e}")
    return cv2.cvtColor(imread(image_path), cv2.COLOR_RGB2BGR)

# ---------- OCR with Paddle + Tesseract fallback ----------
def ocr_page_with_paddle_and_tesseract(img_cv, ocr_model):
    page_texts = []
    try:
        results = ocr_model.ocr(img_cv, cls=True)
        page_result = results[0] if isinstance(results[0], list) else results
        for line in page_result:
            try:
                text_info = line[1] if isinstance(line, (list, tuple)) and len(line) >= 2 else line
                text = text_info[0] if isinstance(text_info, (list, tuple)) else text_info
                text = str(text).strip()
                if text:
                    page_texts.append(text)
            except Exception:
                continue
    except Exception as e:
        logging.warning(f"PaddleOCR failed: {e}")
        try:
            tesseract_text = pytesseract.image_to_string(img_cv, lang="eng", config="--psm 6")
            if tesseract_text.strip():
                page_texts.append(tesseract_text.strip())
        except Exception as tess_error:
            logging.error(f"Tesseract fallback failed: {tess_error}")
    return "\n".join(page_texts).strip()

# ---------- OCR PDF to Text ----------
def ocr_pdf_to_text(pdf_path: str, dpi: int = 350):
    logging.info(f"Starting OCR for PDF: {pdf_path}")
    ocr_model = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)
    image_paths = convert_pdf_to_images(pdf_path, dpi=dpi)
    all_pages_text = []
    for page_index, img_path in enumerate(image_paths):
        logging.info(f"Processing page {page_index+1}/{len(image_paths)}")
        img_cv = preprocess_image(img_path)
        page_text = ocr_page_with_paddle_and_tesseract(img_cv, ocr_model)
        all_pages_text.append({
            "page_index": page_index + 1,
            "image_path": img_path,
            "ocrdata": page_text,
            "text_length": len(page_text)
        })
    return all_pages_text

# ---------- Page-level chunking ----------
def split_page_to_chunks(page_text, doc_id, doc_name, page_number, max_len=1200):
    chunks = []
    start = 0
    text_len = len(page_text)
    chunk_counter = 1
    while start < text_len:
        end = min(start + max_len, text_len)
        chunk_text = page_text[start:end].strip()
        if chunk_text:
            chunks.append({
                "content": chunk_text,
                "metadata": {
                    "doc_id": doc_id,
                    "chunk_id": f"ocr-{page_number}-{chunk_counter}",
                    "doc_name": doc_name,
                    "page_number": page_number,
                    "chunk_type": "text",
                    "category": "information"
                }
            })
            chunk_counter += 1
        start = end
    return chunks

# ---------- Main Integration Function ----------
def extract_and_chunk_pdf_with_ocr(pdf_path: str, max_len=1200):
    """Complete pipeline: OCR PDF -> Extract text -> Create page-level chunks (max 1200 chars)"""
    try:
        doc_name = os.path.basename(pdf_path)
        doc_id = os.path.splitext(doc_name)[0]
        pages_text = ocr_pdf_to_text(pdf_path)
        if not pages_text:
            logging.warning(f"No OCR data extracted from {pdf_path}")
            return []
        chunks = []
        for page_data in pages_text:
            page_text = page_data["ocrdata"].strip()
            if not page_text:
                continue
            page_chunks = split_page_to_chunks(page_text, doc_id, doc_name, page_data["page_index"], max_len=max_len)
            chunks.extend(page_chunks)
        logging.info("\n{0}".format("="*60))
        logging.info("Pipeline completed successfully!")
        logging.info(f"Document: {doc_name}")
        logging.info(f"Total pages processed: {len(pages_text)}")
        logging.info(f"Total chunks created: {len(chunks)}")
        logging.info("{0}".format("="*60))
        return chunks
    except Exception as e:
        logging.error(f"Failed to process {pdf_path}: {e}", exc_info=True)
        return []

# ==== Main Ingestion Loop ====

def get_existing_doc_names():
    try:
        existing = info_collection.get(include=["metadatas"], limit=999999)
        return set(m["doc_name"] for m in existing["metadatas"] if "doc_name" in m)
    except Exception as e:
        logging.warning(f"Could not fetch existing doc_names: {e}")
        return set()

def add_all_pdfs_with_tables_and_text_chunks(base_dir: str):
    existing_doc_names = get_existing_doc_names()
    for root, _, files in os.walk(base_dir):
        for file in files:
            if not file.lower().endswith(".pdf"):
                continue
            file_path = os.path.join(root, file)
            doc_name = file
            if doc_name in existing_doc_names:
                logging.info(f"⏭️ Skipping {doc_name} — already in database.")
                continue
            try:
                if PRODUCT_SUBDIR in root:
                    chunks = chunk_product_pdf_by_sections(file_path, file)
                elif DIVISIONS_SUBDIR in root:
                    chunks = process_structured_division_pdf(file_path, file)
                else:
                    chunks = extract_and_chunk_pdf(
                        file_path,
                        min_len=CHUNK_CONFIG["min_len"],
                        max_len=CHUNK_CONFIG["max_len"],
                        overlap=CHUNK_CONFIG["overlap"]
                    )
                # 🔹 Fallback to OCR if no chunks found
                if not chunks:
                    logging.warning(f"No chunks found in {file}, trying OCR...")
                    chunks = extract_and_chunk_pdf_with_ocr(
                        file_path,
                        max_len=CHUNK_CONFIG["max_len"],
                    )
                # ---- continue with cleaning + uploading ----
                for chunk in chunks:
                    chunk['content'] = clean_chunk_content(chunk['content'])
                chunks = [c for c in chunks if c['content']]
                chunks = merge_small_chunks(chunks, min_length=200)
                chunks = deduplicate_chunks(chunks)
                uploaded_at = datetime.now().isoformat()
                for chunk in chunks:
                    chunk['metadata']["uploaded_at"] = uploaded_at
                for chunk in chunks:
                    embedding = embedding_model.embed_documents([f"passage: {chunk['content']}"])[0]
                    info_collection.add(
                        ids=[f"{chunk['metadata']['doc_id']}-{chunk['metadata']['chunk_id']}"],
                        embeddings=[embedding],
                        documents=[chunk["content"]],
                        metadatas=[chunk["metadata"]],
                    )
                logging.info(f"✅ Added {len(chunks)} chunks from {doc_name}")
            except Exception as e:
                logging.error(f"Failed to process {file_path}: {e}")


#For Json files
# ==== Utility: Flatten JSON ====
def flatten_json(y: dict, parent_key='', sep='.'):
    items = []
    for k, v in y.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_json(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

# ==== Process JSON Files ====
def process_json_files(folder_path):
    for filename in tqdm(os.listdir(folder_path), desc="Processing JSONs"):
        if not filename.lower().endswith(".json"):
            continue

        file_path = os.path.join(folder_path, filename)
        doc_id = str(uuid.uuid4())

        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f"❌ Failed to parse {filename}: {e}")
                continue

        if not isinstance(data, list):
            print(f"⚠️ Skipping {filename} — not a list of objects.")
            continue

        ids, docs, metas = [], [], []

        # === Detect Q&A Format ===
        is_qa_format = all(
            isinstance(d, dict) and "question" in d and "answer" in d for d in data
        )

        if is_qa_format and filename == "EnviStats_FAQ_2025.json":
            for i, qa in enumerate(data):
                q = str(qa.get("question", "")).strip()
                a = str(qa.get("answer", "")).strip()
                if not q or not a:
                    continue

                text = f"Q: {q} \t A: {a}"
                uid = f"{doc_id}-{i}"
                ids.append(uid)
                docs.append(text)
                metas.append({
                    "doc_id": doc_id,
                    "chunk_id": i,
                    "doc_name": filename,
                    "chunk_type": "json",
                    "category": "information"
                })

        elif is_qa_format:
            # Default: chunk in groups of 3 for all other QA-type files
            for i in range(0, len(data), 3):
                chunk = data[i:i+3]
                text = ""
                for qa in chunk:
                    q = str(qa.get("question", "")).strip()
                    a = str(qa.get("answer", "")).strip()
                    if q and a:
                        text += f"Q: {q} \t A: {a}\n\n"
                text = text.strip()
                if not text:
                    continue

                uid = f"{doc_id}-{i//3}"
                ids.append(uid)
                docs.append(text)
                metas.append({
                    "doc_id": doc_id,
                    "chunk_id": i//3,
                    "doc_name": "MOSPI FAQs",
                    "chunk_type": "json",
                    "category": "information"
                })

        else:
            # Original flattening logic
            for i, obj in enumerate(data):
                flat = flatten_json(obj)
                text = ", ".join([f"{k.replace('_', ' ')}: {str(v).strip()}" for k, v in flat.items() if v is not None]) + "."
                if not text.strip():
                    continue

                uid = f"{doc_id}-{i}"
                ids.append(uid)
                docs.append(text)
                metas.append({
                    "doc_id": doc_id,
                    "chunk_id": i,
                    "doc_name": filename,
                    "chunk_type": "json",
                    "category": "information"
                })

        if not docs:
            continue

        try:
            embeddings = model.encode(
                [f"passage: {d}" for d in docs],
                show_progress_bar=False,
                convert_to_numpy=True
            ).tolist()
        except Exception as e:
            print(f"❌ Embedding error in {filename}: {e}")
            continue

        info_collection.add(
            ids=ids,
            documents=docs,
            metadatas=metas,
            embeddings=embeddings
        )

        print(f"✅ Embedded {len(docs)} {'Q&A chunks' if is_qa_format else 'flattened JSON objects'} from: {filename}")


# Utility: Validate URL
def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False

# Utility: Respect robots.txt
def is_allowed_by_robots(url, user_agent="*"):
    try:
        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        robots_url = urljoin(base_url, "/robots.txt")

        rp = RobotFileParser()
        rp.set_url(robots_url)
        rp.read()

        return rp.can_fetch(user_agent, url)
    except Exception as e:
        print(f"⚠️ robots.txt not found or unreadable at {url}: {e}")
        return True  # Default allow

def scrape_dynamic_page_text(url: str, wait_time: int = 5) -> str:
    """Use Playwright for dynamic JS-rendered pages."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            print(f"🔄 Loading dynamic page: {url}")
            page.goto(url, timeout=30000)
            page.wait_for_timeout(wait_time * 1000)
            content = page.content()
            soup = BeautifulSoup(content, "html.parser")
            text_blocks = []
            for tag in soup.find_all(["p", "h1", "h2", "h3", "li", "td", "div"]):
                text = tag.get_text(separator=" ", strip=True)
                if text and len(text) > 30:
                    text_blocks.append(text)
            return "\n".join(text_blocks)
        except Exception as e:
            print(f"❌ Failed to scrape dynamic page: {url} — {e}")
            return ""
        finally:
            browser.close()

def scrape_static_page_text(url: str) -> str:
    """Scrape basic HTML with BeautifulSoup."""
    try:
        print(f"🌐 Scraping static page: {url}")
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        text_blocks = []
        for tag in soup.find_all(["p", "h1", "h2", "h3", "li", "td", "div"]):
            text = tag.get_text(separator=" ", strip=True)
            if text and len(text) > 30:
                text_blocks.append(text)
        return "\n".join(text_blocks)
    except Exception as e:
        print(f"❌ Error scraping static page {url}: {e}")
        return ""

def scrape_text_smart(url: str) -> str:
    if "sdgindia2030" in url or "esankhyiki.mospi.gov.in" in url or "nsc.mospi.gov.in" in url:
        return scrape_dynamic_page_text(url)
    else:
        return scrape_static_page_text(url)


# Utility: Hash for deduplication
def get_chunk_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

# Optional: Normalize text to avoid whitespace-based duplication
def clean_text(text: str) -> str:
    return " ".join(text.split())

# Main Function
def embed_and_store_url_data(urls: Dict[str, str]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    # Get already existing chunks to avoid duplication
    try:
        existing_chunks = info_collection.get(include=["documents", "metadatas"])
        existing_hashes = set(get_chunk_hash(clean_text(doc)) for doc in existing_chunks["documents"])
    except Exception as e:
        print(f"⚠️ Could not fetch existing chunks: {e}")
        existing_hashes = set()

    for title, url in urls.items():
        print(f"\n🔍 Processing: {title} - {url}")

        if not is_valid_url(url):
            print(f"❌ Invalid URL: {url}")
            continue

        if not is_allowed_by_robots(url):
            print(f"🚫 Disallowed by robots.txt: {url}")
            continue

        try:
            raw_text = scrape_text_smart(url)  # your smart scraper using Selenium/requests
        except Exception as e:
            print(f"❌ Failed to scrape {url}: {e}")
            continue

        raw_text = unicodedata.normalize("NFKC", raw_text).strip()
        if not raw_text:
            print(f"⚠️ Empty content from: {url}")
            continue

        try:
            chunks = text_splitter.split_text(raw_text)
        except Exception as e:
            print(f"❌ Error splitting text from {url}: {e}")
            continue

        print(f"📄 {len(chunks)} chunks created from {url}")

        for i, chunk in enumerate(chunks):
            full_chunk = f"{title}\n\n{chunk}"
            cleaned_chunk = clean_text(full_chunk)
            chunk_hash = get_chunk_hash(cleaned_chunk)

            if chunk_hash in existing_hashes:
                print(f"⏭️ Skipping duplicate chunk [Hash]: {chunk_hash}")
                continue
            existing_hashes.add(chunk_hash)

            uid = str(uuid.uuid4())
            metadata = {
                "source": "web",
                "url": url,
                "chunk_id": i,
                "page_title": title,
                "chunk_type": "url",
                "category": "url"
            }

            try:
                embedding = embedding_model.embed_documents([f"passage: {full_chunk}"])[0]
                info_collection.add(
                    ids=[uid],
                    documents=[full_chunk],
                    embeddings=[embedding],
                    metadatas=[metadata]
                )
                print(f"✅ Chunk added: {url} [Chunk ID: {i}]")
            except Exception as e:
                print(f"❌ Embedding error for {url} - Chunk ID {i}: {e}")




def process_visualization_json(file_path: str, delete_existing: bool = True):
    """
    Processes a visualization JSON file where each entry contains:
    - header (chart title)
    - url (public Flourish studio URL)
    - embed_code (iframe HTML)

    Embeds the `header + url`, stores `embed_code` in metadata,
    and includes `doc_name` for better traceability.
    """

    # Removed: clear_visualization_collection()

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load {file_path}: {e}")
        return

    if not isinstance(data, list):
        print(f"⚠️ Expected a list of visualizations in: {file_path}")
        return

    doc_name = os.path.basename(file_path)
    doc_id = str(uuid.uuid4())
    ids, docs, metas = [], [], []

    for i, vis in enumerate(data):
        header = vis.get("header", "").strip()
        url = vis.get("url", "").strip()
        embed_code = vis.get("embed_code", "").strip()

        if not header:
            print(f"⚠️ Skipping entry {i} due to missing header.")
            continue

        content = f"Chart Title: {header}\nChart URL: {url}"
        # Optional: append embed code to make it searchable
        if embed_code:
            content += f"\nEmbed Code:\n{embed_code}"

        uid = f"{doc_id}-{i}"
        ids.append(uid)
        docs.append(content)

        metas.append({
            "doc_id": doc_id,
            "chunk_id": i,
            "doc_name": doc_name,
            "chart_title": header,
            "chart_url": url,
            "embed_code": embed_code,
            "source": "visualization",
            "chunk_type": "visualization",
            "category": "visualization"
        })

    if not docs:
        print("⚠️ No valid visualization entries found.")
        return

    try:
        embeddings = model.encode(
            [f"passage: {doc}" for doc in docs],
            show_progress_bar=False,
            convert_to_numpy=True
        ).tolist()
    except Exception as e:
        print(f"❌ Embedding error for visualizations: {e}")
        return

    info_collection.add(
        ids=ids,
        documents=docs,
        metadatas=metas,
        embeddings=embeddings
    )

    print(f"✅ Added {len(docs)} visualization chunks from {file_path}")




if __name__ == "__main__":
    add_all_pdfs_with_tables_and_text_chunks(MAIN_DIR)
    #process_json_files(MAIN_DIR)
    #embed_and_store_url_data(urls_to_scrape)
    #process_visualization_json("MoSPI_Data/Chart_Data.json")

    print("✅ All URL data scraped and stored in Chroma.")
    print("✅ All PDFs and JSON files processed and stored in Chroma.")


