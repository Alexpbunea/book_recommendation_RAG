import asyncio
import os
import pandas as pd
import argparse
from huggingface_hub import login
from openai_client.setup import Provider
from openai_client.generate import Generation

async def main():
    print("Generating synthetic datasets for book recommendations...")
    parser = argparse.ArgumentParser(description="Generate synthetic datasets for book recommendations.")
    parser.add_argument("--mode", choices=["rag", "classification", "ner"], default="rag", help="Generation mode: 'rag' (default), 'classification', or 'ner'")
    parser.add_argument("--num_examples", type=int, default=300, help="Number of examples to generate")
    args = parser.parse_args()

    try:
        # Load books dataset (used for all modes)
        data_path = "./data/books_1.Best_Books_Ever.csv"
        dataset = pd.read_csv(data_path)
        df = dataset.dropna(subset=['title', 'author', 'description', 'genres', 'pages'])
        
        provider = Provider()
        provider.setup()
        gen = Generation(provider)
        
        num_examples = args.num_examples  # Adjust as needed
        
        # Semaphore to limit concurrent requests to 10
        semaphore = asyncio.Semaphore(5)
        
        async def generate_with_semaphore(sample, mode):
            async with semaphore:
                if mode == "rag":
                    return await gen.generate_synthetic_example(sample)
                elif mode == "classification":
                    return await gen.generate_classification_example(sample)
                elif mode == "ner":
                    return await gen.generate_ner_example(sample)
        
        # Target genres for balanced classification
        TARGET_GENRES = [
            "Romance", "Fantasy", "Young Adult", "Contemporary", 
            "Nonfiction", "Mystery", "Historical Fiction", "Classics"
        ]
        
        # Create tasks for parallel execution
        tasks = []
        for i in range(num_examples):
            if args.mode == "classification":
                # Round-robin selection of target genre
                target_genre = TARGET_GENRES[i % len(TARGET_GENRES)]
                # Filter books that contain the target genre
                # Using string contains for simplicity as these genre names are distinct enough
                genre_df = df[df['genres'].astype(str).str.contains(target_genre, case=False, regex=False)]
                
                if len(genre_df) >= 10:
                    books_sample = genre_df.sample(10, random_state=i)
                else:
                    # Fallback if not enough books for this genre (unlikely for top genres)
                    books_sample = df.sample(10, random_state=i)
                sample = books_sample
            elif args.mode == "ner":
                # Sample 3 books to get real titles and authors for NER generation
                books_sample = df.sample(3, random_state=i)
                sample = books_sample
            else:
                # Default random sampling for RAG mode
                sample = df.sample(10, random_state=i)
                
            task = generate_with_semaphore(sample, args.mode)
            tasks.append(task)
        
        # Execute all tasks in parallel with concurrency limit
        print(f"Generating {num_examples} examples in '{args.mode}' mode (max 20 concurrent)...")
        results_raw = await asyncio.gather(*tasks)
        
        # Convert to dictionaries
        if args.mode == "ner":
            # Flatten NERList items into individual rows
            results = []
            seen = set()  # For deduplication
            for ner_list in results_raw:
                for item in ner_list.items:
                    # Create a hashable key from tokenized_text for deduplication
                    key = tuple(item.tokenized_text)
                    if key not in seen:
                        seen.add(key)
                        results.append(item.model_dump())
            print(f"Deduplicated: {len(results)} unique examples from {sum(len(nl.items) for nl in results_raw)} total")
        else:
            results = [example.model_dump() for example in results_raw]
        
        # Create DataFrame from results
        df_results = pd.DataFrame(results)
        
        if args.mode == "rag":
            # Convert retrieved_indices list to comma-separated string for CSV
            df_results['retrieved_indices'] = df_results['retrieved_indices'].apply(lambda x: ','.join(map(str, x)))
            output_path = "./data/generated_dataset.csv"
        elif args.mode == "classification":
            # Convert genres list to comma-separated string for CSV
            df_results['genres'] = df_results['genres'].apply(lambda x: ','.join(x))
            output_path = "./data/text_clasification.csv"
        elif args.mode == "ner":
            # Convert lists to JSON strings for CSV (matching original dataset format)
            import json
            df_results['tokenized_text'] = df_results['tokenized_text'].apply(lambda x: json.dumps(x))
            # Convert NER spans to original format: [[start, end, label], ...]
            df_results['ner'] = df_results['ner'].apply(
                lambda spans: json.dumps([[s['start_idx'], s['end_idx'], s['label']] for s in spans])
            )
            output_path = "./data/ner_data_augmentation.csv"
        
        # Save to CSV
        df_results.to_csv(output_path, index=False)
        print(f"Dataset saved to {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await provider.close()

if __name__ == "__main__":
    asyncio.run(main())

