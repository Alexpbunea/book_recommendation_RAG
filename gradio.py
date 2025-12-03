import gradio as gr
import os
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForCausalLM, pipeline as hf_pipeline
from setfit import SetFitModel
from peft import PeftModel
import torch
import spaces

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
        classification_model = SetFitModel.from_pretrained("aicoral048/setfit-fined-tuned-books")
        
        print("Loading NER pipeline...")
        ner_tokenizer = AutoTokenizer.from_pretrained("aicoral048/ner-books-model-final")
        ner_model = AutoModelForTokenClassification.from_pretrained("aicoral048/ner-books-model-final")
        
        # Create NER pipeline (device will be handled by ZeroGPU)
        ner_pipeline = hf_pipeline(
            "token-classification",
            model=ner_model,
            tokenizer=ner_tokenizer,
            aggregation_strategy="simple",
            device=-1  # CPU by default, moved to GPU by decorator
        )
        
        print("Loading Qwen model...")
        qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL)
        
        # Load base model (will be moved to GPU by ZeroGPU decorator when needed)
        print("Loading base model (CPU)...")
        base_model = AutoModelForCausalLM.from_pretrained(
            QWEN_BASE_MODEL,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        
        # Load PEFT adapter
        qwen_model = PeftModel.from_pretrained(base_model, QWEN_MODEL)
        
        print("All models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")
        raise e


@spaces.GPU(duration=3)
def classify_genres(text):
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


@spaces.GPU(duration=3)
def extract_entities(text):
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


@spaces.GPU(duration=10)
def generate_with_qwen(prompt, max_tokens, temperature, top_p):
    """Step 3: Generate response using Qwen model with streaming"""
    try:
        if qwen_tokenizer is None or qwen_model is None:
            yield "Error: Qwen model not loaded"
            return
        
        # Move model to GPU (handled automatically by ZeroGPU)
        inputs = qwen_tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to('cuda')
            qwen_model.to('cuda')
        
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
        generated_text = ""
        for new_text in streamer:
            generated_text += new_text
            yield generated_text.strip()
        
        thread.join()
        
    except Exception as e:
        print(f"Qwen error: {e}")
        yield f"Error generating response: {e}"


def respond(
    message,
    history: list[dict[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    """
    Pipeline: User Query -> SetFit (genre classification) -> NER (authors/titles extraction) -> Qwen (generation)
    """
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
        role = msg.get('role', 'user')
        content = msg.get('content', '')
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
    for response in generate_with_qwen(prompt, max_tokens, temperature, top_p):
        yield response


"""
Pipeline: User Query -> SetFit (classification) -> NER (entity extraction) -> Qwen (generation)
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""
chatbot = gr.ChatInterface(
    respond,
    type="messages",
    additional_inputs=[
        gr.Textbox(value="You are a helpful assistant specialized in books. Provide concise, focused book recommendations with brief explanations.", label="System message"),
        gr.Slider(minimum=1, maximum=300, value=150, step=10, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.4, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.8,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
    ],
)

with gr.Blocks() as demo:
    chatbot.render()


if __name__ == "__main__":
    # Load models before starting the app
    print("Loading models...")
    load_models()
    print("Models loaded, starting Gradio app...")
    
    demo.launch(
        share=False,  
        debug=False,
        show_error=True,
        auth=None  
    )
