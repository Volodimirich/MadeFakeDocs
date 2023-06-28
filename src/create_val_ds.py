import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import json
from pathlib import Path

import pandas as pd

import json
import tqdm
import gzip
import re
import os
import gc
import ftfy
import sys
# import unicodedata
# import numpy as np
# from collections import defaultdict
import functools
from functools import partial
from tqdm import tqdm

import nltk

nltk.download('punkt')
from nltk.tokenize import word_tokenize

sys.path.insert(1, './cc_net')  # путь к репе

import kenlm  # поставить библиотеку
from pathlib import Path
from cc_net.perplexity import *
from cc_net import text_normalizer

models = './/cc_net\support_files//ru.arpa.bin'  # путь к файлу
tokenizer_path = './/cc_net\support_files//ru.sp.model'  # путь к файлу
lm = DocLM({'ru': Path(models)}, "tokenized", normalize=False, output_field="perplexity")
lm._prepare()

tokenizer = SentencePiece(Path(tokenizer_path), field='text', normalize=True)
tokenizer = tokenizer._prepare()
cutoff_path = './/cc_net\support_files//cutoff.csv'  # путь к файлу
bucketer = PerplexityBucket(Path(cutoff_path), 40, 60)
bucketer._prepare()
bucketer.cutoffs['ru']


def make_bucket(item):
    item['language'] = 'ru'
    item = lm.do(tokenizer.do(item))
    item = bucketer.do(item)
    return item


def get_russian_alphabet():
    alphabet = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
    alphabet += alphabet.upper()
    return alphabet


def index_end_body(body, end_body_markers):
    ind = len(body)
    for marker in end_body_markers:
        new_ind = re.search(marker, body)
        if new_ind is None:
            continue
        ind = min(ind, new_ind.span()[0])
    return ind


def check_presence_russian_words(list_words, percentage=0.7):
    # Процентное соотношение русских слов к общему количеству слов
    alphabet = get_russian_alphabet()
    cnt_russian_words = 0

    def check(word):
        for ch in cur_word:
            if not ch in alphabet:
                return False
        return True

    for cur_word in list_words:
        if check(cur_word):
            cnt_russian_words += 1

    return True if cnt_russian_words / len(list_words) >= percentage else False


def check_rules(row,
                good_labels,
                good_tags,
                end_body_markers,
                rules_list_words):
    """
    end_body_markers предполагаемые маркеры окончания тела текста
    """
    try:
        query_words = row.query.split(" ")
        body_words = row.body.split(" ")
    except:
        return False

    if row.label in good_labels:
        ind = index_end_body(row.body, end_body_markers)
        body = row.body[:ind]
        if len(body) < 200:
            return False

        list_words = word_tokenize(body)
        # Проверка переданных правил
        for cur_rule in rules_list_words:
            if not cur_rule(list_words):
                return False

        input_body_json = {"text": body}
        make_bucket(input_body_json)
        if input_body_json["bucket"] in good_tags:
            return body

    return False


def create_val_ds():
    CHUNKSIZE = 100000
    FILE = 'D:\MADE\DIPLOM\perplexy\\assessors_train_l_q_u_t_m_b_ql.tsv.gz'
    PATH_TO_SAVE = 'val.json'
    COLUMNS = ['label', 'query', 'url', 'title', 'meta', 'body', 'qlinks']

    MAX_NUM = 4000
    print("Start of generation")
    fun_check_rules = functools.partial(check_rules,
                                        good_labels=[3, 2, 1, 0],
                                        good_tags=["head", "middle", "tail"],
                                        end_body_markers=["Мне нравится:", "Комментарии:", "Оценивайте пожалуйста:",
                                                          "Дата обращения:"],
                                        rules_list_words=[check_presence_russian_words])

    df = pd.DataFrame(columns=COLUMNS)
    with pd.read_csv(FILE, chunksize=CHUNKSIZE, sep='\t', names=COLUMNS, compression='gzip') as reader:
        for chunk in tqdm(reader):
            chunk['body'] = chunk.apply(fun_check_rules, axis=1)
            good_query = chunk.loc[chunk['body'] != False, 'query'].unique()
            df = pd.concat([df, chunk.loc[chunk['query'].isin(good_query)]])
            if MAX_NUM < len(good_query):
                break

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

    Path(PATH_TO_SAVE).parent.mkdir(parents=True, exist_ok=True)
    with open(PATH_TO_SAVE, "w", encoding="utf8") as final:
        json.dump(data, final, ensure_ascii=False)


if __name__ == "__main__":
    create_val_ds()
