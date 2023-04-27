from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
from src.metrics.evaluation_metrics import (
    TopK, AverageLoc, FDARO
)

class Bm25:
    """Класс метрики ранжирования bm25"""

    def __init__(self):
        pass

    def __name__(self):
        return "Bm25"

    def ranking(self, query: str, sentences: list[str], labels: list[int]) -> list:
        """
        Функция ранжирования bm25

        Parameters
        ------------
        query: `str`
            Список токенов запроса
        sentences: `list[str]`
            Список списков токенов текстов
        labels: `list[int]`
            Список меток текстов

        Returns
        ------------
        `list`
            Список оценок релевантности текстов
        """
        tokenized_query = self._encode(query)[0]
        tokenized_sentences = self._encode(sentences)
        bm25 = BM25Okapi(tokenized_sentences)
        scores = bm25.get_scores(tokenized_query)

        scores = self._sorted(scores, labels)
        return scores

    def _sorted(self, scores: list[float], labels: list[int]):
        """
        Функция сортировки оценки и лейблов

        Parameters
        ------------
        scores: `list[float]`
            Массив оценок ранка присвоенных ранкером
        labels: `list[int]`
            Массив меток

        Returns
        ------------
        `list[list]`
            Отсортированный список ранжируемых элементов по релевантности
        """
        return sorted([item for item in zip(scores, labels)], key=lambda x: x[0], reverse=True)

    def _encode(self, sentences: str or list[str]):
        """
        Функция для декодирования предложений в последовательность токенов

        Parameters
        ------------
        sentences: `str, list[str]`

        Returns
        ------------
        `list[str]` or `list[list[str]]`
            Последовательность токенов
        """
        # tokenized_sentences = self.tokenizer.encode(sentences, return_tensors="pt")
        if isinstance(sentences, str):
            sentences = [sentences]

        tokenized_sentences = []
        for cur_sent in sentences:
            tokenized_sentences.append(cur_sent.split(" "))
        return tokenized_sentences


class LaBSE:
    """Класс метрики ранжирования LaBSE"""

    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/LaBSE')

    def __name__(self):
        return "LaBSE"

    def ranking(self, query: str, sentences: list[str], labels: list[int]) -> list:
        """
        Функция ранжирования LaBSE

        Parameters
        ------------
        query: `str`
            Строка запроса
        sentences: `list[str]`
            Список строк текстов
        labels: `list[int]`
            Список меток текстов

        Returns
        ------------
        `list`
            Список оценок релевантности текстов
        """
        query = self.model.encode(query)
        embeddings = self.model.encode(sentences)
        scores = util.pytorch_cos_sim(query, embeddings).numpy()[0]
        scores = self._sorted(scores, labels)
        return scores

    def _sorted(self, scores: list[str], labels: list[str]):
        """
            Функция сортировки оценки и лейблов

            Parameters
            ------------
            scores: `list[float]`
                Массив оценок ранка присвоенных ранкером
            labels: `list[int]`
                Массив меток

            Returns
            ------------
            `list[list]`
                Отсортированный список ранжируемых элементов по релевантности
            """
        return sorted([item for item in zip(scores, labels)], key=lambda x: x[0])


class RankingMetrics:
    """Класс аккумулирующий все метрики"""
    FAKE_DOC_LABEL: int = 2

    def __init__(self, metrics):
        # Среднее место фейковых документов в финальной выдаче
        self.average_place_fake_doc = AverageLoc(metrics)
        # Количество случаев когда фейковый документ выше релевантного
        self.fake_doc_above_relevant_one = FDARO(metrics)
        # Количество случаев когда фейковый документ вошел в топ 1
        self.fake_top_k = TopK(metrics)
        # Классы метрик для подсчета
        self.metrics = metrics

    def update(self, query: str, sentences: list[str], labels: list[int]):
        """
           Функция обновления всех метрик по переданным данным

           Parameters
           ------------
           query: `str`
               Строка запроса
           sentences: `list[str]`
               Список строк текстов
           labels: `list[int]`
               Список меток текстов

           """
        if not isinstance(query, str):
            raise TypeError("The request must be a string!")

        if len(sentences) != len(labels):
            raise "len(labels) must be equal to len(sentences)"

        for item in sentences:
            if not isinstance(item, str):
                raise "The sentences must be of the `list[str]` type!"

        for item in labels:
            if not isinstance(item, int):
                raise "The labels must be of the `list[int]` type!"

        for cur_metric in self.metrics:
            ranking_list = cur_metric.ranking(query, sentences, labels)
            self.fake_top_k.update(cur_metric.__name__(), ranking_list, RankingMetrics.FAKE_DOC_LABEL)
            self.fake_doc_above_relevant_one.update(cur_metric.__name__(), ranking_list, RankingMetrics.FAKE_DOC_LABEL)
            self.average_place_fake_doc.update(cur_metric.__name__(), ranking_list, RankingMetrics.FAKE_DOC_LABEL)

    def get(self):
        """
        Функция для получения значения всех метрик

        Returns
        ----------
        `dict`
            Словарь значений метрик
        """
        result = {}
        for key_, value in self.average_place_fake_doc.get().items():
            result[key_] = value

        for key_, value in self.fake_top_k.get().items():
            result[key_] = value

        for key_, value in self.fake_doc_above_relevant_one.get().items():
            result[key_] = value

        return result

