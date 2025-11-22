import asyncio
import pandas as pd
import argparse
from openai_client.setup import Provider
from openai_client.generate import Generation

async def main():
    print("Generating synthetic datasets for book recommendations...")
    parser = argparse.ArgumentParser(description="Generate synthetic datasets for book recommendations.")
    parser.add_argument("--mode", choices=["rag", "classification"], default="rag", help="Generation mode: 'rag' (default) or 'classification'")
    parser.add_argument("--num_examples", type=int, default=300, help="Number of examples to generate")
    args = parser.parse_args()

    try:
        data_path = "./data/books_1.Best_Books_Ever.csv"
        dataset = pd.read_csv(data_path)
        df = dataset.dropna(subset=['title', 'author', 'description', 'genres', 'pages'])
        
        provider = Provider()
        provider.setup()
        gen = Generation(provider)
        
        num_examples = args.num_examples  # Adjust as needed
        
        # Semaphore to limit concurrent requests to 10
        semaphore = asyncio.Semaphore(10)
        
        async def generate_with_semaphore(books_sample, mode):
            async with semaphore:
                if mode == "rag":
                    return await gen.generate_synthetic_example(books_sample)
                elif mode == "classification":
                    return await gen.generate_classification_example(books_sample)
        
        # Create tasks for parallel execution
        tasks = []
        for i in range(num_examples):
            books_sample = df.sample(10, random_state=i)  # Sample 10 books for more realistic RAG scenario
            task = generate_with_semaphore(books_sample, args.mode)
            tasks.append(task)
        
        # Execute all tasks in parallel with concurrency limit
        print(f"Generating {num_examples} examples in '{args.mode}' mode (max 20 concurrent)...")
        results_raw = await asyncio.gather(*tasks)
        
        # Convert to dictionaries
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
        
        # Save to CSV
        df_results.to_csv(output_path, index=False)
        print(f"Dataset saved to {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await provider.close()

if __name__ == "__main__":
    asyncio.run(main())

