import random
import json

json_file_path = '/home/d.maximov/data/T5_prediction_20-06-2023-22-15/Examples.json'
prefix = '<LM> '
with open(json_file_path, 'r') as f:
    data = json.load(f)

# Russian alphabet symbols
russian_alphabet = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя '

# Create TSV file and write query and generated text
with open('/home/d.maximov/data/random/Examples.tsv', 'w', encoding='utf-8') as file:
    for item in data:
        query = item['query']
        if query.startswith(prefix):
            query = query[len(prefix):]
        else:
            raise RuntimeError
        random_symbols = ''.join(random.choice(russian_alphabet) for _ in range(750))
        file.write(f"{query}\t{random_symbols}\n")

print("Output saved to output.tsv file.")
