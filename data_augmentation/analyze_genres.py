import csv
import ast
from collections import Counter
import sys

# Increase CSV field size limit to a safe 32-bit integer
csv.field_size_limit(2147483647)

try:
    with open('data/text_clasification.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        all_genres = []
        for row in reader:
            genres_str = row.get('genres')
            if not genres_str:
                continue
                
            try:
                # Try to evaluate as a list if it looks like one
                if genres_str.strip().startswith('[') and genres_str.strip().endswith(']'):
                    genres = ast.literal_eval(genres_str)
                else:
                    # Otherwise assume comma separated
                    genres = [g.strip() for g in genres_str.split(',')]
                
                all_genres.extend(genres)
            except Exception as e:
                # print(f"Error parsing genres: {genres_str} - {e}")
                pass

    genre_counts = Counter(all_genres)
    print("\nTop 20 Genres:")
    for genre, count in genre_counts.most_common(20):
        print(f"{genre}: {count}")

except Exception as e:
    print(f"Error: {e}")
