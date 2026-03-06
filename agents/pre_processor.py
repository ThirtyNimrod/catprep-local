import os
import glob
import json
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.logger import get_logger

logger = get_logger("pre_processor")

def parse_document(file_path):
    """Parses a document (PDF, MD, TXT) and returns chunked text."""
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return []

    logger.info(f"Parsing document: {file_path}")
    ext = os.path.splitext(file_path)[1].lower()
    
    documents = []
    if ext == '.pdf':
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            return []
    elif ext in ['.md', '.txt']:
        try:
            loader = TextLoader(file_path, encoding='utf-8')
            documents = loader.load()
        except Exception as e:
            logger.error(f"Error loading Text {file_path}: {e}")
            return []
    else:
        logger.warning(f"Unsupported file type: {ext}. Only PDF, MD, and TXT are currently supported.")
        return []

    # Split into manageable chunks for extraction
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100
    )
    
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Successfully extracted {len(chunks)} chunks from {file_path}.")
    
    # Return string content of chunks
    return [chunk.page_content for chunk in chunks]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Parse documents for Knowledge Graph extraction.")
    parser.add_argument("--file", type=str, required=True, help="Path to the file to parse")
    parser.add_argument("--output", type=str, default="parsed_chunks.json", help="Path to save extracted chunks")
    
    args = parser.parse_args()
    
    chunks = parse_document(args.file)
    if chunks:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2)
        logger.info(f"Saved {len(chunks)} chunks to {args.output}")
        logger.debug("--- SAMPLE OF CHUNK 0 ---")
        logger.debug(chunks[0][:500])
