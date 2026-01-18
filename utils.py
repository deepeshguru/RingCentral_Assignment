"""Utility functions for the car manual Q&A application."""

import re
import requests
import pdfplumber
from pathlib import Path
from typing import List, Dict, Optional
from config import CAR_MANUALS
import logging

logger = logging.getLogger(__name__)


def download_manual(car_model: str, save_dir: str = "manuals") -> str:
    """Download a car manual PDF if not already present."""
    Path(save_dir).mkdir(exist_ok=True)

    manual_info = CAR_MANUALS[car_model]
    filepath = Path(save_dir) / manual_info["filename"]

    if filepath.exists():
        logger.info(f"✓ Manual for {car_model} already exists at {filepath}")
        return str(filepath)

    logger.info(f"📥 Downloading manual for {car_model}...")
    logger.info(f"   URL: {manual_info['url']}")
    response = requests.get(manual_info["url"], stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    logger.info(f"   File size: {total_size / (1024*1024):.2f} MB")

    with open(filepath, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    logger.info(f"✓ Downloaded: {filepath}")
    return str(filepath)


def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, any]]:
    """Extract text from PDF with page numbers."""
    logger.info(f"📄 Extracting text from PDF: {pdf_path}")
    pages_data = []

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        logger.info(f"   Total pages: {total_pages}")
        
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                pages_data.append({"page": page_num, "text": text.strip()})
            
            if page_num % 10 == 0:
                logger.info(f"   Processed {page_num}/{total_pages} pages...")

    logger.info(f"✓ Extracted text from {len(pages_data)} pages")
    return pages_data


def chunk_text(
    pages_data: List[Dict], chunk_size: int = 800, overlap: int = 100
) -> List[Dict]:
    """Split text into overlapping chunks with metadata."""
    logger.info(f"✂️  Chunking text (chunk_size={chunk_size}, overlap={overlap})...")
    chunks = []

    for page_data in pages_data:
        text = page_data["text"]
        page_num = page_data["page"]

        # Split into chunks
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            chunks.append({"text": chunk_text, "page": page_num, "start_char": start})

            start += chunk_size - overlap

    logger.info(f"✓ Created {len(chunks)} chunks from {len(pages_data)} pages")
    return chunks


def detect_car_model(query: str) -> Optional[str]:
    """Detect car model from user query."""
    logger.info(f"Detecting car model from query: '{query[:100]}...'")
    query_lower = query.lower()

    for car_model, info in CAR_MANUALS.items():
        for keyword in info["keywords"]:
            if keyword in query_lower:
                logger.info(f"✓ Detected car model: {car_model} (matched keyword: '{keyword}')")
                return car_model

    logger.warning("Could not detect car model from query")
    return None


def format_citation(page_num: int) -> str:
    """Format citation for display."""
    return f"[Page {page_num}]"
