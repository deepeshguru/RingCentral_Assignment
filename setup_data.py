"""Script to download manuals and setup vector database."""

from utils import download_manual, extract_text_from_pdf, chunk_text
from vector_store import VectorStore
from config import CAR_MANUALS, CHUNK_SIZE, CHUNK_OVERLAP
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_database():
    """Download manuals and populate vector database."""
    logger.info("="*80)
    logger.info("🚀 Starting Car Manual Database Setup")
    logger.info("="*80)

    # Initialize vector store
    logger.info("\n📦 Initializing vector store...")
    vector_store = VectorStore()

    # Check if already populated
    if not vector_store.is_empty():
        logger.warning("⚠️  Database already populated. Skipping setup.")
        logger.info("   To rebuild, delete the 'chroma_db' directory and run again.")
        return

    # Process each car manual
    total_cars = len(CAR_MANUALS)
    for idx, car_model in enumerate(CAR_MANUALS.keys(), 1):
        logger.info("\n" + "="*80)
        logger.info(f"🚗 Processing {car_model} ({idx}/{total_cars})")
        logger.info("="*80)

        try:
            # Download manual
            logger.info(f"\n📥 Step 1: Downloading manual...")
            pdf_path = download_manual(car_model)

            # Extract text
            logger.info(f"\n📄 Step 2: Extracting text...")
            pages_data = extract_text_from_pdf(pdf_path)

            # Chunk text
            logger.info(f"\n✂️  Step 3: Chunking text...")
            chunks = chunk_text(pages_data, CHUNK_SIZE, CHUNK_OVERLAP)

            # Add to vector store
            logger.info(f"\n💾 Step 4: Adding to vector database...")
            vector_store.add_documents(chunks, car_model)
            
            logger.info(f"\n✓ Successfully processed {car_model}")
            
        except Exception as e:
            logger.error(f"\n❌ Error processing {car_model}: {str(e)}")
            raise

    logger.info("\n" + "="*80)
    logger.info("✓ Database setup complete!")
    logger.info(f"   Total cars processed: {total_cars}")
    logger.info(f"   Total documents in database: {vector_store.collection.count()}")
    logger.info("="*80)


if __name__ == "__main__":
    setup_database()
