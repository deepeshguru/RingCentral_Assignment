"""RAG pipeline using LangChain and IBM Granite model."""

from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import List, Dict
from config import GRANITE_MODEL
import torch
import logging

logger = logging.getLogger(__name__)


class RAGPipeline:
    """RAG pipeline for question answering."""

    def __init__(self):
        """Initialize the IBM Granite model via LangChain."""
        logger.info("🤖 Initializing IBM Granite model...")
        logger.info(f"   Model: {GRANITE_MODEL}")
        logger.info(f"   Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

        # Load tokenizer and model
        logger.info("   Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(GRANITE_MODEL)
        logger.info("   ✓ Tokenizer loaded")
        
        logger.info("   Loading model (this may take a few minutes)...")
        model = AutoModelForCausalLM.from_pretrained(
            GRANITE_MODEL,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
        )
        logger.info("   ✓ Model loaded")

        # Create pipeline
        logger.info("   Creating text generation pipeline...")
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.1,
        )
        logger.info("   ✓ Pipeline created")

        # Wrap in LangChain
        logger.info("   Wrapping in LangChain...")
        self.llm = HuggingFacePipeline(pipeline=pipe)

        # Define prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""Based on the following context from a car manual, answer the question.
If the answer is not in the context, say "I cannot find this information in the manual."

Context:
{context}

Question: {question}

Answer (be concise and include page references):""",
        )

        logger.info("✓ IBM Granite RAG pipeline initialized successfully!")

    def generate_answer(self, question: str, retrieved_docs: List[Dict]) -> Dict:
        """Generate answer using retrieved documents."""
        logger.info(f"Generating answer for question: {question[:100]}...")
        
        if not retrieved_docs:
            logger.warning("No retrieved documents provided")
            return {
                "answer": "No relevant information found in the manual.",
                "citations": [],
            }

        # Prepare context
        logger.info(f"Preparing context from {len(retrieved_docs)} documents...")
        context_parts = []
        citations = []

        for doc in retrieved_docs:
            context_parts.append(f"[Page {doc['page']}]: {doc['text']}")
            if doc["page"] not in citations:
                citations.append(doc["page"])

        context = "\n\n".join(context_parts)
        logger.info(f"   Context length: {len(context)} characters")
        logger.info(f"   Pages referenced: {citations}")

        # Generate answer
        logger.info("Invoking LLM to generate answer...")
        prompt = self.prompt_template.format(context=context, question=question)
        logger.info(f"   Prompt length: {len(prompt)} characters")
        
        raw_answer = self.llm.invoke(prompt)
        logger.info(f"   Raw answer length: {len(raw_answer)} characters")
        
        # Extract only the generated answer (remove the prompt echo)
        # The LLM sometimes returns the full prompt + answer, so we need to extract just the answer
        answer = raw_answer.strip()
        
        # Try to find where the actual answer starts (after "Answer:")
        if "Answer (be concise and include page references):" in answer:
            answer = answer.split("Answer (be concise and include page references):")[-1].strip()
        elif "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()
        
        logger.info(f"✓ Answer generated ({len(answer)} characters)")

        return {
            "answer": answer,
            "citations": sorted(citations),
            "retrieved_chunks": retrieved_docs,
        }
