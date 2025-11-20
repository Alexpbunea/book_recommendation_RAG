import asyncio
import pandas as pd
from openai_client.setup import Provider
from openai_client.generate import Generation

async def main():
    data_path = "./data/books_1.Best_Books_Ever.csv"
    dataset = pd.read_csv(data_path)
    df = dataset.dropna(subset=['title', 'author', 'description', 'genres', 'pages'])
    
    provider = Provider()
    provider.setup()
    gen = Generation(provider)
    
    num_examples = 300  # Adjust as needed
    
    # Semaphore to limit concurrent requests to 20
    semaphore = asyncio.Semaphore(20)
    
    async def generate_with_semaphore(books_sample):
        async with semaphore:
            return await gen.generate_synthetic_example(books_sample)
    
    # Create tasks for parallel execution
    tasks = []
    for i in range(num_examples):
        books_sample = df.sample(10, random_state=i)  # Sample 10 books for more realistic RAG scenario
        task = generate_with_semaphore(books_sample)
        tasks.append(task)
    
    # Execute all tasks in parallel with concurrency limit
    print(f"Generating {num_examples} examples (max 20 concurrent)...")
    results_raw = await asyncio.gather(*tasks)
    
    # Convert to dictionaries
    results = [example.model_dump() for example in results_raw]
    
    # Create DataFrame from results
    df_results = pd.DataFrame(results)
    # Convert retrieved_indices list to comma-separated string for CSV
    df_results['retrieved_indices'] = df_results['retrieved_indices'].apply(lambda x: ','.join(map(str, x)))
    
    # Save to CSV
    output_path = "./data/generated_dataset.csv"
    df_results.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")

    await provider.close()

if __name__ == "__main__":
    asyncio.run(main())

