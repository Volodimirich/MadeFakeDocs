import pandas as pd

import json
import tqdm
import gzip
import re
import os
import gc
import ftfy
import sys
import functools
from functools import partial
from tqdm import tqdm
from transformers import GPT2Tokenizer

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

tokenizer_gpt = GPT2Tokenizer.from_pretrained("ai-forever/rugpt3large_based_on_gpt2")

if tokenizer_gpt.pad_token is None:
    SPECIAL_TOKENS = {'bos_token': '<bos>', 'eos_token': '<s>', 'pad_token': '<pad>', 'sep_token': '<sep>'}
    tokenizer_gpt.add_special_tokens(SPECIAL_TOKENS)

tokenizer_gpt.padding_side = 'left'


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
                rules_list_words,
                tokenizer_gpt):
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
        if len(body) < 200 or 1_000 < len(tokenizer_gpt.encode(body)):
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


if __name__ == "__main__":
    CHUNKSIZE = 100000
    FILE = 'D:\MADE\DIPLOM\perplexy\\assessors_train_l_q_u_t_m_b_ql.tsv.gz'
    PATH_TO_SAVE = "train.csv"
    COLUMNS = ['label', 'query', 'url', 'title', 'meta', 'body', 'qlinks']
    df = pd.DataFrame(columns=COLUMNS)
    cnt = 0
    MAX_NUM = 2000
    print("Start of generation")
    fun_check_rules = functools.partial(check_rules,
                                        good_labels=[3],
                                        good_tags=["head"],
                                        end_body_markers=["Мне нравится:", "Комментарии:", "Оценивайте пожалуйста:",
                                                          "Дата обращения:"],
                                        rules_list_words=[check_presence_russian_words],
                                        tokenizer_gpt=tokenizer_gpt)

    with pd.read_csv(FILE, chunksize=CHUNKSIZE, sep='\t', names=COLUMNS, compression='gzip') as reader:
        for chunk in tqdm(reader):
            chunk['body'] = chunk.apply(fun_check_rules, axis=1)
            df = pd.concat([df, chunk.loc[chunk['body'] != False]])
            cnt += len(chunk.loc[chunk['body'] != False])
            if cnt > MAX_NUM:
                break

    df.sample(MAX_NUM).to_csv(PATH_TO_SAVE)
