import json

def create_tsv_from_json(json_file, tsv_file, prefix):

    with open(json_file, 'r') as f:
        data = json.load(f)

    with open(tsv_file, 'w') as f:
        for item in data:
            query = item['query']
            if query.startswith(prefix):
                query = query[len(prefix):]
            else:
                raise RuntimeError
            if len(passage) == 0:
                passage = '.'
            f.write(f"{query}\t{passage}\n")

    print("TSV file created successfully.")


if __name__ == "__main__":
    json_file_path = '/home/d.maximov/data/made_valid_data/val.json'
    tsv_file_path = '/home/d.maximov/data/made_valid_data/val.tsv'
    # json_file_path = '/home/d.maximov/data/T5_prediction_22-06-2023-20-48/Examples.json'
    # tsv_file_path = '/home/d.maximov/data/T5_prediction_22-06-2023-20-48/Examples.tsv'
    prefix = '<LM> '
    create_tsv_from_json(json_file_path, tsv_file_path, prefix)
