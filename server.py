from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForCausalLM, pipeline as hf_pipeline
from setfit import SetFitModel
from peft import PeftModel
import torch
from contextlib import asynccontextmanager

# Genre names for classification model
GENRE_NAMES = ['Classics', 'Contemporary', 'Fantasy', 'Historical Fiction', 'Mystery', 'Nonfiction', 'Romance', 'Young Adult']

# Model IDs
SETFIT_MODEL = "aicoral048/setfit-fined-tuned-books"
NER_MODEL = "aicoral048/ner-books-model-final"
QWEN_MODEL = "aicoral048/qwen-finetuned-final"
QWEN_BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# Global model variables
classification_model = None
ner_pipeline = None
qwen_tokenizer = None
qwen_model = None


def load_models():
    """Load all models into memory"""
    global classification_model, ner_pipeline, qwen_tokenizer, qwen_model
    
    try:
        print("Loading SetFit classification model...")
        classification_model = SetFitModel.from_pretrained(SETFIT_MODEL)
        
        print("Loading NER pipeline...")
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
        
        print("Loading Qwen model...")
        qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL)
        
        # Load base model
        print("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            QWEN_BASE_MODEL,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        )
        
        # Load PEFT adapter
        qwen_model = PeftModel.from_pretrained(base_model, QWEN_MODEL)
        
        if torch.cuda.is_available():
            qwen_model.to('cuda')
        
        print("All models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")
        raise e


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup"""
    print("Loading models...")
    load_models()
    print("Models loaded, server ready!")
    yield
    # Cleanup on shutdown (if needed)
    print("Shutting down...")


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
    max_tokens: int = 150
    temperature: float = 0.4
    top_p: float = 0.8


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
        print(f"Classification error: {e}")
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
        print(f"NER error: {e}")
        return {'authors': [], 'titles': []}


def generate_with_qwen(prompt: str, max_tokens: int, temperature: float, top_p: float):
    """Step 3: Generate response using Qwen model with streaming"""
    try:
        if qwen_tokenizer is None or qwen_model is None:
            yield "Error: Qwen model not loaded"
            return
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        inputs = qwen_tokenizer(prompt, return_tensors="pt").to(device)
        
        # Use TextIteratorStreamer for streaming generation
        from transformers import TextIteratorStreamer
        from threading import Thread
        
        streamer = TextIteratorStreamer(qwen_tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            num_beams=1,
            early_stopping=True,
            repetition_penalty=1.5,
            length_penalty=1.0,
            pad_token_id=qwen_tokenizer.eos_token_id,
            streamer=streamer
        )
        
        # Start generation in a separate thread
        thread = Thread(target=qwen_model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Stream the output
        for new_text in streamer:
            yield new_text
        
        thread.join()
        
    except Exception as e:
        print(f"Qwen error: {e}")
        yield f"Error generating response: {e}"


def stream_response(request: ChatRequest):
    """
    Pipeline: User Query -> SetFit (genre classification) -> NER (authors/titles extraction) -> Qwen (generation)
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
    
    # Step 3: Build enriched context for Qwen
    genres_info = f"Predicted Genres: {', '.join(predicted_genres)}" if predicted_genres else "Predicted Genres: None detected"
    authors_info = f"Extracted Authors: {', '.join(authors)}" if authors else "Extracted Authors: None detected"
    titles_info = f"Extracted Titles: {', '.join(titles)}" if titles else "Extracted Titles: None detected"
    
    # Build conversation history as text
    history_text = ""
    for msg in history:
        role = msg.role
        content = msg.content
        history_text += f"{role.capitalize()}: {content}\n"
    
    # Build the full prompt for Qwen (optimized for small model)
    context = f"""Analysis of user query:
- {genres_info}
- {authors_info}
- {titles_info}"""
    
    prompt = f"""You are a book recommendation expert. Provide ONE book recommendation in 2-3 sentences max.

Query: {message}
Context: {context}

Book recommendation (title and author only):"""

    # Step 4: Generate response with Qwen (streaming)
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
            "qwen": qwen_model is not None
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
