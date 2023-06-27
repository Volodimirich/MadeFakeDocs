import re
separator = ' 🚩'
def check_rules(row):
    try:
        query_words = row.query.split(" ")
        body_words = row.body.split(" ")
    except:
        return False
    if row.label in (3, 2) and row.url.startswith('kakprosto.ru'):
        match = re.search(r'» - \d+ ответ(а|ов)? ', row.body)
        if match is None:
            title = re.search(separator, row.title)
            if title is not None:
                match = re.search(row.title[:title.span()[0]], row.body)
        if match is not None:
            end = re.search(r' Совет полезен\? Да Нет', row.body)
            if end is not None:
                new_str = row.body[match.span()[1]:end.span()[0]]
                filer = re.search(' Источники: ', new_str)
                if filer is not None:
                    new_str = new_str[:filer.span()[0]]
                filer = re.search(' Связанная статья ', new_str)
                if filer is not None:
                    new_str = new_str[:filer.span()[0]]
                return new_str 
    return False

CHUNKSIZE = 100000
FILE = '/mnt/DATA/n.ermolaev/assessors_train_l_q_u_t_m_b_ql.tsv.gz'
COLUMNS = ['label', 'query', 'url', 'title', 'meta', 'body', 'qlinks']
df = pd.DataFrame(columns=COLUMNS)
with pd.read_csv(FILE, chunksize=CHUNKSIZE, sep='\t', names=COLUMNS, compression='gzip') as reader:
    for chunk in tqdm_notebook(reader):
        chunk['body'] = chunk.apply(check_rules, axis=1)
        df = pd.concat([df, chunk.loc[chunk['body'] != False]])