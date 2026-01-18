"""Streamlit application for car manual Q&A."""

import streamlit as st
import logging
from datetime import datetime
from utils import detect_car_model
from vector_store import VectorStore
from rag_pipeline import RAGPipeline
from config import CAR_MANUALS, TOP_K_RESULTS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Page configuration
st.set_page_config(page_title="Car Manual Q&A", page_icon="🚗", layout="centered")

# Title and description
st.title("🚗 Car Manual Q&A Assistant")
st.markdown("Ask questions about your car manual (MG Astor or Tata Tiago)")


# Initialize components (with caching)
@st.cache_resource
def load_vector_store():
    """Load vector store (cached)."""
    logger.info("Loading vector store...")
    vector_store = VectorStore()
    logger.info("✓ Vector store loaded successfully")
    return vector_store


@st.cache_resource
def load_rag_pipeline():
    """Load RAG pipeline (cached)."""
    logger.info("Loading RAG pipeline with IBM Granite model...")
    rag_pipeline = RAGPipeline()
    logger.info("✓ RAG pipeline loaded successfully")
    return rag_pipeline


# Main application
def main():
    # Load components
    try:
        logger.info("Initializing application components...")
        vector_store = load_vector_store()
        rag_pipeline = load_rag_pipeline()
        logger.info("✓ All components initialized successfully")
    except Exception as e:
        logger.error(f"Error loading components: {str(e)}")
        st.error(f"Error loading components: {str(e)}")
        st.info(
            "Please run `python setup_data.py` first to download manuals and setup the database."
        )
        return

    # Check if database is empty
    if vector_store.is_empty():
        logger.warning("Database is empty")
        st.warning("⚠️ Database is empty. Please run `python setup_data.py` first.")
        return

    # User input
    question = st.text_input(
        "Enter your question:",
        placeholder="e.g., How to turn on indicator in MG Astor?",
        key="question_input",
    )

    # Submit button
    if st.button("Get Answer", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
            return

        # Log question received
        logger.info("="*80)
        logger.info(f"📝 Question received: {question}")
        logger.info(f"⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        with st.spinner("Processing your question..."):
            # Detect car model
            logger.info("🔍 Step 1: Detecting car model...")
            car_model = detect_car_model(question)
            
            if not car_model:
                logger.warning(f"❌ Could not detect car model from question: {question}")
                st.error("❌ Could not detect car model from your question.")
                st.info(f"Supported models: {', '.join(CAR_MANUALS.keys())}")
                st.info(
                    "Please mention the car model in your question (e.g., 'MG Astor' or 'Tata Tiago')"
                )
                return
            
            logger.info(f"✓ Detected car model: {car_model}")

            # Search vector database
            logger.info(f"🔍 Step 2: Searching vector database for '{car_model}'...")
            logger.info(f"   Retrieving top {TOP_K_RESULTS} relevant chunks...")
            retrieved_docs = vector_store.search(question, car_model, TOP_K_RESULTS)

            if not retrieved_docs:
                logger.error(f"❌ No documents found for {car_model}")
                st.error(f"❌ Manual is not available for {car_model}.")
                return
            
            logger.info(f"✓ Retrieved {len(retrieved_docs)} relevant chunks")
            for i, doc in enumerate(retrieved_docs, 1):
                logger.info(f"   Chunk {i}: Page {doc['page']}, Distance: {doc.get('distance', 'N/A'):.4f}")

            # Generate answer
            logger.info("🤖 Step 3: Generating answer using IBM Granite LLM...")
            result = rag_pipeline.generate_answer(question, retrieved_docs)
            logger.info("✓ Answer generated successfully")
            logger.info(f"   Answer length: {len(result['answer'])} characters")
            logger.info(f"   Citations: Pages {result['citations']}")

            # Display results
            st.success(f"✓ Found information in {car_model} manual")
            logger.info("✓ Response sent to user")
            logger.info("="*80)

            # Answer
            st.markdown("### Answer")
            st.markdown(result["answer"])

            # Citations
            if result["citations"]:
                st.markdown("### Citations")
                citation_text = ", ".join(
                    [f"Page {page}" for page in result["citations"]]
                )
                st.info(f"📖 {citation_text}")

            # Show retrieved context (expandable)
            with st.expander("View Retrieved Context"):
                for i, doc in enumerate(result["retrieved_chunks"], 1):
                    st.markdown(f"**Chunk {i} (Page {doc['page']})**")
                    st.text(
                        doc["text"][:300] + "..."
                        if len(doc["text"]) > 300
                        else doc["text"]
                    )
                    st.markdown("---")


# Sidebar with information
with st.sidebar:
    st.markdown("### About")
    st.markdown("This application uses:")
    st.markdown("- 🤖 IBM Granite LLM")
    st.markdown("- 🔍 Semantic search with ChromaDB")
    st.markdown("- 📚 RAG (Retrieval Augmented Generation)")

    st.markdown("### Supported Cars")
    for car_model in CAR_MANUALS.keys():
        st.markdown(f"- {car_model}")

    st.markdown("### Example Questions")
    st.markdown("- How to turn on indicator in MG Astor?")
    st.markdown("- Which engine oil to use in Tiago?")
    st.markdown("- What is the fuel tank capacity of MG Astor?")


if __name__ == "__main__":
    main()
