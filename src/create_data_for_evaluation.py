import json

def create_tsv_from_json(json_file, prefix, n):
    tsv_file = f'/home/d.maximov/data/T5_prediction_19-06-2023-18-54/Examples_{n}.tsv'

    with open(json_file, 'r') as f:
        data = json.load(f)

    with open(tsv_file, 'w') as f:
        for item in data:
            query = item['query']
            passage = item['generated_text']
            if query.startswith(prefix):
                query = query[len(prefix):]
            else:
                raise RuntimeError
            new_prefix = ' '.join([query] * n)
            f.write(f"{query}\t{new_prefix + ' ' + passage}\n")

    print("TSV file created successfully.")


if __name__ == "__main__":
    # json_file_path = '/home/d.maximov/data/made_valid_data/val.json'
    # tsv_file_path = '/home/d.maximov/data/made_valid_data/val.tsv'
    json_file_path = '/home/d.maximov/data/T5_prediction_19-06-2023-18-54/Examples.json'
    prefix = '<LM> '
    for n in range(1, 4):
        create_tsv_from_json(json_file_path, prefix, n)
