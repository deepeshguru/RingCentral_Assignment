# Scaling Guide: Adding More Data and Handling Complex Content

This guide explains how to extend the application to handle more car manuals, tabular data, and images.

---

## 1. Adding New Car Manuals

### Step 1: Update Configuration

Edit `config.py` and add the new car model:

```python
CAR_MANUALS = {
    "MG Astor": {
        "url": "https://example.com/mg_astor.pdf",
        "filename": "mg_astor_manual.pdf",
        "keywords": ["mg astor", "astor", "mg"]
    },
    "Tata Tiago": {
        "url": "https://example.com/tata_tiago.pdf",
        "filename": "tata_tiago_manual.pdf",
        "keywords": ["tata tiago", "tiago", "tata"]
    },
    # Add new car here
    "Honda City": {
        "url": "https://example.com/honda_city.pdf",
        "filename": "honda_city_manual.pdf",
        "keywords": ["honda city", "city", "honda"]
    }
}
```

### Step 2: Run Setup Script

```bash
# Delete existing database to rebuild with new data
rm -rf chroma_db/

# Run setup to download and process all manuals
python setup_data.py
```

### Step 3: Test

The new car manual is now available! Test with:
- "What is the fuel capacity of Honda City?"

---

## 2. Handling Tabular Data

Tables in PDFs require special processing. Here's how to enhance the system:

### Option A: Simple Table Extraction (Recommended for Start)

**Update `utils.py`** to extract tables:

```python
import pdfplumber
import pandas as pd

def extract_tables_from_pdf(pdf_path: str) -> List[Dict]:
    """Extract tables from PDF."""
    tables_data = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            
            for table_idx, table in enumerate(tables):
                if table:
                    # Convert to DataFrame for better formatting
                    df = pd.DataFrame(table[1:], columns=table[0])
                    
                    # Convert to text representation
                    table_text = f"Table {table_idx + 1} on Page {page_num}:\n"
                    table_text += df.to_string(index=False)
                    
                    tables_data.append({
                        "page": page_num,
                        "table_index": table_idx,
                        "text": table_text,
                        "type": "table"
                    })
    
    return tables_data
```

**Update `setup_data.py`** to include tables:

```python
def setup_database():
    vector_store = VectorStore()
    
    for car_model in CAR_MANUALS.keys():
        pdf_path = download_manual(car_model)
        
        # Extract text
        pages_data = extract_text_from_pdf(pdf_path)
        text_chunks = chunk_text(pages_data, CHUNK_SIZE, CHUNK_OVERLAP)
        
        # Extract tables
        tables_data = extract_tables_from_pdf(pdf_path)
        table_chunks = chunk_text(tables_data, CHUNK_SIZE, CHUNK_OVERLAP)
        
        # Combine and add to vector store
        all_chunks = text_chunks + table_chunks
        vector_store.add_documents(all_chunks, car_model)
```

### Option B: Advanced Table Understanding

For better table understanding, use specialized models:

```python
# Add to requirements.txt
table-transformer>=1.0.0
pytesseract>=0.3.10

# Use table-transformer for complex tables
from transformers import TableTransformerForObjectDetection
```

---

## 3. Handling Images and Diagrams

Images require multimodal processing. Here are approaches:

### Option A: OCR for Text in Images

**Add to `requirements.txt`:**
```
pytesseract>=0.3.10
Pillow>=10.0.0
```

**Update `utils.py`:**

```python
from PIL import Image
import pytesseract
import io

def extract_images_from_pdf(pdf_path: str) -> List[Dict]:
    """Extract text from images in PDF using OCR."""
    images_data = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            # Extract images
            for img_idx, img in enumerate(page.images):
                try:
                    # Get image
                    image = page.within_bbox(
                        (img["x0"], img["top"], img["x1"], img["bottom"])
                    ).to_image()
                    
                    # Convert to PIL Image
                    pil_image = image.original
                    
                    # OCR
                    text = pytesseract.image_to_string(pil_image)
                    
                    if text.strip():
                        images_data.append({
                            "page": page_num,
                            "image_index": img_idx,
                            "text": f"Image {img_idx + 1} on Page {page_num}:\n{text}",
                            "type": "image"
                        })
                except Exception as e:
                    print(f"Error processing image on page {page_num}: {e}")
    
    return images_data
```

### Option B: Multimodal LLM (Advanced)

For understanding diagrams and images semantically:

**Use a multimodal model like:**
- IBM Granite Vision (when available)
- LLaVA (open source)
- BLIP-2 (open source)

```python
# Add to requirements.txt
transformers>=4.37.2
Pillow>=10.0.0

# Example with BLIP-2
from transformers import Blip2Processor, Blip2ForConditionalGeneration

def describe_image(image_path: str) -> str:
    """Generate description of image using BLIP-2."""
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
    
    image = Image.open(image_path)
    inputs = processor(image, return_tensors="pt")
    
    generated_ids = model.generate(**inputs, max_length=100)
    description = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return description
```

---

## 4. Complete Enhanced Pipeline

Here's how to integrate everything:

### Enhanced `setup_data.py`:

```python
def setup_database():
    """Enhanced setup with tables and images."""
    vector_store = VectorStore()
    
    for car_model in CAR_MANUALS.keys():
        print(f"\nProcessing {car_model}...")
        
        pdf_path = download_manual(car_model)
        
        # 1. Extract text
        print("Extracting text...")
        pages_data = extract_text_from_pdf(pdf_path)
        text_chunks = chunk_text(pages_data, CHUNK_SIZE, CHUNK_OVERLAP)
        
        # 2. Extract tables
        print("Extracting tables...")
        tables_data = extract_tables_from_pdf(pdf_path)
        table_chunks = chunk_text(tables_data, CHUNK_SIZE, CHUNK_OVERLAP)
        
        # 3. Extract images (OCR)
        print("Extracting images...")
        images_data = extract_images_from_pdf(pdf_path)
        image_chunks = chunk_text(images_data, CHUNK_SIZE, CHUNK_OVERLAP)
        
        # 4. Combine all chunks
        all_chunks = text_chunks + table_chunks + image_chunks
        
        # 5. Add to vector store
        print(f"Adding {len(all_chunks)} chunks to database...")
        vector_store.add_documents(all_chunks, car_model)
    
    print("\n✓ Enhanced database setup complete!")
```

---

## 5. Storage Optimization for Scale

### Option A: Separate Collections per Car

```python
# In vector_store.py
class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.collections = {}
    
    def get_collection(self, car_model: str):
        """Get or create collection for specific car."""
        if car_model not in self.collections:
            collection_name = car_model.lower().replace(" ", "_")
            self.collections[car_model] = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        return self.collections[car_model]
```

### Option B: Database Sharding

For 100+ car models:

```python
# Shard by first letter of car name
def get_shard_name(car_model: str) -> str:
    first_letter = car_model[0].upper()
    return f"shard_{first_letter}"

# Create separate ChromaDB instances per shard
```

---

## 6. Incremental Updates

To add data without rebuilding everything:

```python
def add_single_manual(car_model: str, pdf_path: str):
    """Add a single manual without rebuilding entire database."""
    vector_store = VectorStore()
    
    # Check if already exists
    existing_docs = vector_store.collection.get(
        where={"car_model": car_model}
    )
    
    if existing_docs["ids"]:
        print(f"Deleting existing data for {car_model}...")
        vector_store.collection.delete(
            where={"car_model": car_model}
        )
    
    # Process and add new data
    pages_data = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(pages_data, CHUNK_SIZE, CHUNK_OVERLAP)
    vector_store.add_documents(chunks, car_model)
    
    print(f"✓ Added {car_model} to database")
```

---

## 7. Performance Optimization

### Batch Processing

```python
# Process multiple PDFs in parallel
from concurrent.futures import ThreadPoolExecutor

def setup_database_parallel():
    """Process manuals in parallel."""
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for car_model in CAR_MANUALS.keys():
            future = executor.submit(process_single_manual, car_model)
            futures.append(future)
        
        for future in futures:
            future.result()
```

### Caching

```python
# Cache embeddings to disk
import pickle

def cache_embeddings(chunks, car_model):
    """Cache embeddings to avoid recomputation."""
    cache_file = f"cache/{car_model}_embeddings.pkl"
    
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    
    embeddings = embedding_model.encode(chunks)
    
    os.makedirs("cache", exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump(embeddings, f)
    
    return embeddings
```

---

## 8. Summary

| Feature | Complexity | Implementation Time | Benefits |
|---------|-----------|-------------------|----------|
| Add new car manuals | Low | 5 minutes | More coverage |
| Table extraction | Medium | 2-3 hours | Better structured data |
| Image OCR | Medium | 2-3 hours | Extract text from diagrams |
| Multimodal understanding | High | 1-2 days | Understand visual content |
| Parallel processing | Medium | 1-2 hours | Faster setup |
| Database sharding | High | 1 day | Handle 1000+ manuals |

**Recommended Priority:**
1. Add more car manuals (easy wins)
2. Table extraction (high value)
3. Image OCR (medium value)
4. Parallel processing (if you have 10+ manuals)
5. Multimodal models (advanced use cases)

The current architecture is designed to be extensible - you can add these features incrementally without major refactoring!