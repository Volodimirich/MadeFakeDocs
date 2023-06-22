import pandas as pd
import numpy as np
from tqdm import tqdm


def read_float_file(filename):
    numbers = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            number = float(line.strip())
            numbers.append(number)
    return numbers


if __name__ == "__main__":
    val_texts_file_path = '/home/d.maximov/data/made_valid_data/val.tsv'
    df_val = pd.read_csv(val_texts_file_path, sep='\t', names=['query', 'passage'])
    models = [
        'T5_prediction_19-06-2023-18-54',
        'random',
        'T5_prediction_20-06-2023-17-58',
        'T5_prediction_20-06-2023-18-00',
        'T5_prediction_20-06-2023-18-22',
        'T5_prediction_20-06-2023-18-35',
        'T5_prediction_20-06-2023-22-15'
        ]
    scores_suffixes = [
        ['1.txt', '2.txt', '3.txt', '4.txt', '5.txt', '5_1_repeat.txt', '5_2_repeat.txt', '5_3_repeat.txt'],
        ['5_random.txt'],
        ['5_3416_bs.txt'],
        ['5_1952_sampling.txt'],
        ['5_1952_bs.txt'],
        ['5_4880_sampling.txt'],
        ['5_4880_bs.txt']
    ]
    for scores_suffix, model in zip(scores_suffixes, models):
        print(f'checkpoint: {model}')
        test_texts_file_path = f'/home/d.maximov/data/{model}/Examples.tsv'
        df_test = pd.read_csv(test_texts_file_path, sep='\t', names=['query', 'passage'])
        df_val_query = set(df_val['query'].unique())
        df_test_query = set(df_test['query'].unique())
        diff = df_val_query ^ df_test_query
        for i, score_suffix in enumerate(scores_suffix):
            print(f'model {score_suffix}')
            test_results_file_path = f'/home/d.maximov/scores_val_for_model_{score_suffix}'
            if len(scores_suffix) > 1 and i <= 4:
                val_results_file_path = f'/home/d.maximov/scores_val_ranking_model_{i + 1}.txt'
            else:
                val_results_file_path = f'/home/d.maximov/scores_val_ranking_model_5.txt'

            df_val['scores'] = read_float_file(val_results_file_path)
            df_val['is_model'] = 0
            
            df_test['scores'] = read_float_file(test_results_file_path)
            df_test['is_model'] = 1

            df_val_filt = df_val[~df_val['query'].isin(diff)].copy()
            df_test_filt = df_test[~df_test['query'].isin(diff)].copy()
            df = pd.concat([df_val_filt, df_test_filt])
            positions = []
            docs_count = []
            queries = df['query'].unique()
            for query in tqdm(queries):
                pos = 0
                temp_df = df[df['query'].eq(query)]
                temp_df = temp_df.sort_values(by='scores', ascending=False)
                for row in temp_df.is_model:
                    pos += 1
                    if row == 1:
                        positions.append(pos)
                        break
                docs_count.append(len(temp_df))
            print(f'медианная позиция: {round(np.median(positions), 2)}')
            print(f'средне взвешенная позиция: {round(np.average(positions, weights=1 / np.array(docs_count)), 2)}')

            percent_pos = []
            for pos, len_docks in zip(positions, docs_count):
                percent_pos.append((pos - 1) / len_docks * 100)
            print(f'медианная позиция в процентах: {round(np.median(percent_pos), 2)}%')
            bad = 0
            good = 1
            norm = 0
            if score_suffix == '1.txt':
                bad_query = None
                norm_query = None
                good_query = None
                for j in range(len(positions)):
                    if docs_count[j] < 4:
                        continue
                    if positions[j] / docs_count[j] > bad:
                        bad = positions[j] / docs_count[j]
                        bad_query = queries[j]
                    if positions[j] / docs_count[j] < good:
                        good = positions[j] / docs_count[j]
                        good_query = queries[j]
                    if abs(positions[j] / docs_count[j] - 0.5) < abs(norm - 0.5):
                        norm = positions[j] / docs_count[j]
                        norm_query = queries[j]
                score_df = df[df['query'].isin([bad_query, norm_query, good_query])]
                score_df = score_df.rename(columns={"scores": f"scores_1"})
            elif model == 'T5_prediction_19-06-2023-18-54' and i <= 4:
                score_df = pd.merge(score_df, 
                df[df['query'].isin([bad_query, norm_query, good_query])].rename(columns={"scores": f"scores_{i + 1}"}),
                how='inner', on=['query', 'passage', 'is_model']
                )
        # score_df.to_csv('/home/d.maximov/data/examples/texts_1952_sampling.csv', index=False)
    for query in score_df['query'].unique():
        df = score_df[score_df['query'].eq(query)]
        print('#'*100)
        print(f'запрос: {query}')
        print('текст сгенерированный моделью:')
        print(f"оценка: {df.loc[df.is_model.eq(1), ['scores_1', 'scores_2', 'scores_3', 'scores_4', 'scores_5']]}")
        print(df[df.is_model.eq(1)]['passage'].values[0])
        print('Лучший немодельный текст:')
        tmp = df[df.is_model.eq(0)].sort_values(by='scores_5', ascending=False).iloc[0]
        print(f"оценки: {tmp[['scores_1', 'scores_2', 'scores_3', 'scores_4', 'scores_5']]}")
        print(tmp['passage'])
