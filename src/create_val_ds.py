import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import json
from pathlib import Path


def create_val_ds():
    CHUNKSIZE = 100000
    FILE = '/mnt/DATA/n.ermolaev/assessors_test_l_q_u_t_m_b_ql.tsv.gz'
    COLUMNS = ['label', 'query', 'url', 'title', 'meta', 'body', 'qlinks']
    df = pd.DataFrame(columns=COLUMNS)
    with pd.read_csv(FILE, chunksize=CHUNKSIZE, sep='\t', names=COLUMNS, compression='gzip') as reader:
        for chunk in tqdm(reader):
            good_query = chunk.loc[chunk.url.str.startswith('kakprosto.ru'), 'query'].unique()
            df = pd.concat([df, chunk.loc[chunk['query'].isin(good_query)]])

    val = defaultdict(lambda: defaultdict(list))
    for _, row in df.iterrows():
        val[row.query]['is_selected'].append(row.label)
        text = str(row.body).split(' ')
        text = ' '.join(text[:1000])
        val[row.query]['passage_text'].append(text)

    data = []
    for key in val:
        row = {
            'query': key,
            'passages': dict(val[key])
        }
        data.append(row)

    path = '/home/d.maximov/data/datasets/val.json'
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as final:
        json.dump(data, final, ensure_ascii=False)

if __name__ == "__main__":
    create_val_ds()
