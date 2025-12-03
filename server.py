from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForCausalLM, pipeline as hf_pipeline
from setfit import SetFitModel
from peft import PeftModel
import torch
from contextlib import asynccontextmanager
from huggingface_hub import login
from dotenv import load_dotenv
import os
from utils import logger

# Load environment variables and login to HuggingFace
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)
    logger.info("Logged in to HuggingFace!")

# Import RAG search functionality
from search_books import BookSearcher

# Genre names for classification model
GENRE_NAMES = ['Classics', 'Contemporary', 'Fantasy', 'Historical Fiction', 'Mystery', 'Nonfiction', 'Romance', 'Young Adult']

# Model IDs
SETFIT_MODEL = "aicoral048/setfit-fined-tuned-books"
NER_MODEL = "aicoral048/ner-books-model-final"
QWEN_MODEL = "aicoral048/qwen-finetuned-final"
QWEN_BASE_MODEL = "Qwen/Qwen3-0.6B"#"google/gemma-3-270m-it"#Qwen/Qwen2.5-0.5B-Instruct"


# Global model variables
classification_model = None
ner_pipeline = None
qwen_tokenizer = None
qwen_model = None
book_searcher = None


def load_models():
    """Load all models into memory"""
    global classification_model, ner_pipeline, qwen_tokenizer, qwen_model, book_searcher
    
    try:
        logger.info("Loading SetFit classification model...")
        classification_model = SetFitModel.from_pretrained(SETFIT_MODEL)
        
        logger.info("Loading NER pipeline...")
        ner_tokenizer = AutoTokenizer.from_pretrained(NER_MODEL)
        ner_model = AutoModelForTokenClassification.from_pretrained(NER_MODEL)
        
        # Create NER pipeline
        device = 0 if torch.cuda.is_available() else -1
        ner_pipeline = hf_pipeline(
            "token-classification",
            model=ner_model,
            tokenizer=ner_tokenizer,
            aggregation_strategy="simple",
            device=device
        )
        
        logger.info("Loading Qwen3 model...")
        qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_BASE_MODEL)
        
        # Load Qwen3 with auto device mapping
        logger.info("Loading base model...")
        qwen_model = AutoModelForCausalLM.from_pretrained(
            QWEN_BASE_MODEL,
            torch_dtype="auto",
            device_map="auto"
        )
        
        logger.info("Loading RAG book searcher...")
        book_searcher = BookSearcher()
        
        logger.info("All models loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise e


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup"""
    logger.info("Loading models...")
    load_models()
    logger.info("Models loaded, server ready!")
    yield
    # Cleanup on shutdown (if needed)
    logger.info("Shutting down...")


app = FastAPI(
    title="Book Recommendation API",
    description="API for book recommendations using SetFit, NER, and Qwen models",
    lifespan=lifespan
)

# Add CORS middleware to allow requests from Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your Vercel domain: ["https://your-app.vercel.app"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []
    system_message: str = "You are a helpful assistant specialized in books. Provide concise, focused book recommendations with brief explanations."
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.6


def classify_genres(text: str) -> list[str]:
    """Step 1: Classify genres using SetFit model"""
    try:
        if classification_model is None:
            return []
        
        # SetFit predict returns a list of binary predictions
        predicted_genres_binary = classification_model.predict([text])[0]
        
        # Find indices where binary is 1
        predicted_genres_indices = [i for i, val in enumerate(predicted_genres_binary) if val == 1]
        
        # Map indices to genre names
        predicted_genre_names = [GENRE_NAMES[i] for i in predicted_genres_indices if i < len(GENRE_NAMES)]
        
        return predicted_genre_names
    except Exception as e:
        logger.error(f"Classification error: {e}")
        return []


def extract_entities(text: str) -> dict:
    """Step 2: Extract authors and titles using NER model"""
    try:
        if ner_pipeline is None:
            return {'authors': [], 'titles': []}
        
        entities = ner_pipeline(text)
        authors = [ent['word'] for ent in entities if ent['entity_group'] == 'AUTHOR']
        titles = [ent['word'] for ent in entities if ent['entity_group'] == 'TITLE']

        # Filter out invalid titles (e.g., too long or containing query-like words)
        filtered_titles = []
        for title in titles:
            title_words = title.split()
            # Discard if longer than 5 words or contains words like "like", "similar", "recommend"
            if len(title_words) <= 5 and not any(word.lower() in ['like', 'similar', 'recommend', 'book', 'novel'] for word in title_words):
                filtered_titles.append(title)
        
        return {'authors': authors, 'titles': filtered_titles}
    except Exception as e:
        logger.error(f"NER error: {e}")
        return {'authors': [], 'titles': []}


def generate_with_qwen(prompt: str, max_tokens: int, temperature: float, top_p: float):
    """Generate response using Qwen3 model with thinking mode and streaming"""
    try:
        if qwen_tokenizer is None or qwen_model is None:
            yield "Error: Qwen model not loaded"
            return
        
        # Prepare chat template with thinking enabled
        messages = [{"role": "user", "content": prompt}]
        text = qwen_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # Enable thinking mode for better reasoning
        )
        model_inputs = qwen_tokenizer([text], return_tensors="pt").to(qwen_model.device)
        
        # Use TextIteratorStreamer for streaming generation
        from transformers import TextIteratorStreamer
        from threading import Thread
        
        streamer = TextIteratorStreamer(qwen_tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        generation_kwargs = dict(
            **model_inputs,
            max_new_tokens=2048,  # Increased for longer responses
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            streamer=streamer
        )
        
        # Start generation in a separate thread
        thread = Thread(target=qwen_model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Stream output directly (thinking is disabled)
        for new_text in streamer:
            # Clean up special tokens
            clean_text = new_text.replace("<think>", "").replace("</think>", "")
            clean_text = clean_text.replace("<|im_end|>", "").replace("<|im_start|>", "")
            clean_text = clean_text.replace("<|endoftext|>", "")
            if clean_text:
                yield clean_text
        
        thread.join()
        
    except Exception as e:
        logger.error(f"Qwen error: {e}")
        yield f"Error generating response: {e}"


def stream_response(request: ChatRequest):
    """
    Pipeline: User Query -> SetFit (genre classification) -> NER (authors/titles extraction) -> RAG Search -> Qwen (generation)
    """
    message = request.message
    history = request.history
    max_tokens = request.max_tokens
    temperature = request.temperature
    top_p = request.top_p
    
    # Step 1: Classify genres with SetFit
    predicted_genres = classify_genres(message)
    
    # Step 2: Extract authors and titles with NER
    entities = extract_entities(message)
    authors = entities['authors']
    titles = entities['titles']
    
    logger.info(f"[Pipeline] Query: {message}")
    logger.info(f"[Pipeline] Genres: {predicted_genres}")
    logger.info(f"[Pipeline] Authors: {authors}")
    logger.info(f"[Pipeline] Titles: {titles}")
    
    # Step 3: RAG Search - Hybrid search with genre prioritization
    retrieved_books = []
    books_context = "No books found in database."
    if book_searcher is not None:
        try:
            retrieved_books = book_searcher.hybrid_search(
                query=message,
                genres=predicted_genres if predicted_genres else None,
                authors=authors if authors else None,
                titles=titles if titles else None,
                n_results=2  # 2 recommendations
            )
            books_context = book_searcher.format_results_for_prompt(retrieved_books)
            logger.debug(books_context)
        except Exception as e:
            logger.error(f"RAG search error: {e}")
            books_context = "Error searching book database."
    
    # Step 4: Build enriched context for Qwen
    genres_info = f"Predicted Genres: {', '.join(predicted_genres)}" if predicted_genres else "Predicted Genres: None detected"
    authors_info = f"Mentioned Authors: {', '.join(authors)}" if authors else ""
    titles_info = f"Mentioned Titles: {', '.join(titles)}" if titles else ""
    
    # Count how many books we found
    num_books = len(retrieved_books)
    
    # Build the full prompt for Qwen - simple and direct
    prompt = f"""Here are {num_books} book recommendations:

{books_context}

Write a brief, friendly response presenting these {num_books} books. For each book use:
📚 **Title** by Author - Why it's great (1 sentence)

RULES:
- Only mention the {num_books} books above
- Do not add any other books
- Keep explanations short"""

    # Step 5: Generate response with Qwen (streaming)
    for chunk in generate_with_qwen(prompt, max_tokens, temperature, top_p):
        yield chunk


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Stream chat endpoint for book recommendations.
    
    Pipeline: User Query -> SetFit (classification) -> NER (entity extraction) -> Qwen (generation)
    """
    return StreamingResponse(
        stream_response(request),
        media_type="text/plain"
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": {
            "classification": classification_model is not None,
            "ner": ner_pipeline is not None,
            "qwen": qwen_model is not None,
            "rag_searcher": book_searcher is not None
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
