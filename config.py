"""Configuration file for the car manual Q&A application."""

# Supported car models and their manual URLs
CAR_MANUALS = {
    "MG Astor": {
        "url": "https://www.team-bhp.com/forum/attachments/official-new-car-reviews/2238569d1638110150-mg-astor-review-astor-manual.pdf",
        "filename": "mg_astor_manual.pdf",
        "keywords": ["mg astor", "astor", "mg"],
    },
    "Tata Tiago": {
        "url": "https://tmlcars.tatamotors.com/images/service/owners/owners-manual/pdf/tiago/APP-TIAGO-FINAL-OMSB.pdf",
        "filename": "tata_tiago_manual.pdf",
        "keywords": ["tata tiago", "tiago", "tata"],
    },
}

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# IBM Granite model
GRANITE_MODEL = "ibm-granite/granite-3b-code-instruct"

# ChromaDB settings
CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "car_manuals"

# PDF processing settings
CHUNK_SIZE = 800  # characters
CHUNK_OVERLAP = 100  # characters

# Retrieval settings
TOP_K_RESULTS = 3
