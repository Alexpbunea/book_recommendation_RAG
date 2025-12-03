# 📚 Book Recommendation RAG System

A full-stack book recommendation system using Retrieval-Augmented Generation (RAG) with fine-tuned ML models. Features a FastAPI backend with multiple specialized models and a modern Next.js frontend.

## 🏗️ Architecture

```
User Query
    ↓
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                          │
├─────────────────────────────────────────────────────────────┤
│  1. SetFit Model (Genre Classification)                     │
│     └── Classifies query into 8 genres                      │
│  2. NER Model (Entity Extraction)                           │
│     └── Extracts author names and book titles               │
│  3. ChromaDB + Hybrid Search                                │
│     ├── Keyword matching for authors/titles                 │
│     └── Semantic search with reranking for genres           │
│  4. Qwen3-0.6B (Response Generation)                        │
│     └── Generates friendly recommendations                  │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│              Next.js + assistant-ui Frontend                │
│     └── Chat interface with streaming responses             │
└─────────────────────────────────────────────────────────────┘
```

## ✨ Features

- **Multi-Model Pipeline**: Combines genre classification, NER, and LLM generation
- **Hybrid Search**: Keyword matching for exact author/title queries + semantic search for descriptive queries
- **Reranking**: Uses fine-tuned SetFit embeddings for result reranking
- **Streaming Responses**: Real-time token streaming for better UX
- **Modern Frontend**: Next.js 15 with assistant-ui components
- **Deployable**: Backend runs on any server, frontend deploys to Vercel

## 🤖 Models Used

| Model | Purpose | Source |
|-------|---------|--------|
| `aicoral048/setfit-fined-tuned-books` | Genre classification (8 genres) | HuggingFace |
| `aicoral048/ner-books-model-final` | Author/Title extraction | HuggingFace |
| `Qwen/Qwen3-0.6B` | Response generation | HuggingFace |
| SetFit embeddings | ChromaDB indexing & reranking | HuggingFace |

**Supported Genres**: Classics, Contemporary, Fantasy, Historical Fiction, Mystery, Nonfiction, Romance, Young Adult

## 📁 Project Structure

```
book_recommendation_RAG/
├── server.py              # FastAPI backend with full pipeline
├── search_books.py        # Hybrid search with ChromaDB + reranking
├── index_books.py         # One-time script to index books into ChromaDB
├── books.csv              # Book dataset (Best Books Ever)
├── data/
│   ├── chroma.sqlite3     # ChromaDB persistent storage
│   ├── books_1.Best_Books_Ever.csv  # Original dataset
│   ├── generated_dataset.csv        # Synthetic RAG training data
│   ├── text_clasification.csv       # Genre classification data
│   └── ner_*.jsonl        # NER training/eval data
├── frontend/              # Next.js + assistant-ui frontend
│   ├── app/               # Next.js app router
│   ├── components/        # React components
│   └── api/               # API routes for chat
├── jupyter_notebooks/     # Training notebooks
│   ├── First_two_model_pipeline.ipynb  # SetFit & NER training
│   └── NER.ipynb          # NER model development
├── data_augmentation/     # Data augmentation scripts
├── pyproject.toml         # Python dependencies (uv)
└── requirements.txt       # Python dependencies (pip)
```

## 🚀 Installation

### Backend Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Alexpbunea/book_recommendation_RAG.git
   cd book_recommendation_RAG
   ```

2. Install Python dependencies (using uv - recommended):
   ```bash
   pip install uv
   uv sync
   ```

   Or using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Add your HuggingFace token to .env
   HF_TOKEN=your_huggingface_token
   ```

4. Index books into ChromaDB (first time only):
   ```bash
   uv run index_books.py
   ```

5. Start the backend server:
   ```bash
   uv run server.py
   ```
   Server runs at `http://localhost:8000`

### Frontend Setup

1. Navigate to frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   pnpm install
   # or npm install
   ```

3. Configure backend URL in `api/route.ts`:
   ```typescript
   const BACKEND_URL = "http://localhost:8000";  // or your server URL
   ```

4. Start development server:
   ```bash
   pnpm dev
   ```
   Frontend runs at `http://localhost:3000`

## 📡 API Endpoints

### `POST /api/chat/stream`
Stream chat responses for book recommendations.

**Request:**
```json
{
  "message": "I want a fantasy book with magic",
  "history": [],
  "max_tokens": 4096,
  "temperature": 0.7
}
```

**Response:** Streaming text with book recommendations

### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": {
    "classification": true,
    "ner": true,
    "qwen": true,
    "rag_searcher": true
  }
}
```

## 🔍 Search Pipeline

The system uses a smart hybrid search approach:

1. **Author/Title Queries** (e.g., "books by J.K. Rowling"):
   - Direct keyword matching on metadata fields
   - Bypasses semantic search for accuracy

2. **Genre/Description Queries** (e.g., "fantasy with magic"):
   - Semantic search with genre-enriched queries (85% weight)
   - Entity-enriched queries for context (15% weight)
   - Reranking with SetFit embeddings
   - Returns top 2 results

## 🛠️ Data Generation

Generate synthetic training data using OpenAI:

### RAG Dataset
```bash
python main.py --mode rag --num_examples 300
```
Output: `data/generated_dataset.csv`

### Text Classification Dataset
```bash
python main.py --mode classification --num_examples 100
```
Output: `data/text_clasification.csv`

## 🌐 Deployment

### Backend (VM/Server)
```bash
# Run with uvicorn
uv run server.py

# Expose via Cloudflare Tunnel (recommended)
cloudflared tunnel --url http://localhost:8000
```

### Frontend (Vercel)
```bash
cd frontend
vercel deploy
```

Update `BACKEND_URL` in the frontend to point to your tunnel URL.

## 📊 Example Queries

| Query | Search Type | Result |
|-------|-------------|--------|
| "books by J.K. Rowling" | Keyword (author) | Harry Potter series |
| "look for Harry Potter" | Keyword (title) | Harry Potter books |
| "fantasy with magic" | Semantic + rerank | Fantasy genre books |
| "mystery thriller" | Semantic + rerank | Mystery genre books |

## 🧪 Testing

Test the search functionality:
```bash
uv run search_books.py
```

Test the full pipeline:
```bash
curl -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "recommend a fantasy book"}'
```

## 📝 License

MIT License

## 🙏 Acknowledgments

- Book dataset from [Best Books Ever](https://www.kaggle.com/datasets/thedevastator/best-books-ever-dataset)
- Frontend built with [assistant-ui](https://github.com/assistant-ui/assistant-ui)
- Models hosted on [HuggingFace](https://huggingface.co/aicoral048)