
class TopK:
    TOP1 = 1
    TOP3 = 3
    TOP5 = 5

    def __init__(self, ranking_metrics):
        self.metrics = {}
        self._calls_cnt = {}
        self._separator = "_"
        self._top_numbers = [TopK.TOP1, TopK.TOP3, TopK.TOP5]
        for cur_metrics in ranking_metrics:
            for cur_top in self._top_numbers:
                self.metrics[cur_metrics.__name__() + self._separator + self.__name__() + str(cur_top)] = 0
                self._calls_cnt[cur_metrics.__name__() + self._separator + self.__name__() + str(cur_top)] = 0
        #
        # for cur_top in self._top_numbers:
        #     self._calls_cnt[self.__name__() + str(cur_top)] = 0

    def update(self, metric_name: str, ranking_list: list, fake_doc_label: int or list = 2):
        """
           Функция для обновления значений метрики

           Pparameters
           -------------
           metric_name:
           ranking_list:
           fake_doc_label:
           real_doc_label:

           Returns
           -------------
           """

        if isinstance(fake_doc_label, int):
            fake_doc_label = [fake_doc_label]

        if not isinstance(fake_doc_label, list):
            raise "The tags of fake documents must be 'int' or 'list!'"

        for top_num in self._top_numbers:
            for item in ranking_list[:min(len(ranking_list), top_num)]:
                if item[1] in fake_doc_label:
                    self.metrics[metric_name + self._separator + self.__name__() + str(top_num)] += 1

            self._calls_cnt[metric_name + self._separator + self.__name__() + str(top_num)] += 1

    def get(self):
        result = {}
        for metric_name, value in self.metrics.items():
            metric_name_ = metric_name.split("_")[0]
            top_num = int(metric_name.split('@')[1])
            result[metric_name_ + self._separator + self.__name__() + str(top_num)] = \
                value / self._calls_cnt[metric_name_ + self._separator + self.__name__() + str(top_num)]

        return result

    def __name__(self):
        return "Top@"


class FDARO:
    """Метрика для оценки как часто фейковый документ выше релевантного"""

    def __init__(self, ranking_metrics):
        self._separator = "_"
        self._metrics, self._calls_cnt = {}, {}
        for cur_metric in ranking_metrics:
            self._metrics[cur_metric.__name__() + self._separator + self.__name__()] = 0
            self._calls_cnt[cur_metric.__name__() + self._separator + self.__name__()] = 0

    def update(self, metric_name: str,
               ranking_list: list,
               fake_doc_label: int or list = 2,
               real_doc_label: int or list = 1):
        """
        Функция для обновления значений метрики

        Pparameters
        -------------
        metric_name:
        ranking_list:
        fake_doc_label:
        real_doc_label:

        Returns
        -------------
        """
        if isinstance(fake_doc_label, int):
            fake_doc_label = [fake_doc_label]

        if not isinstance(fake_doc_label, list):
            raise "The tags of fake documents must be 'int' or 'list!'"

        if isinstance(real_doc_label, int):
            real_doc_label = [real_doc_label]

        if not isinstance(real_doc_label, list):
            raise "The labels of these documents should be 'int' or 'list!'"

        is_first = False
        for item in ranking_list:
            if item[1] in real_doc_label:
                break
            elif item[1] in fake_doc_label:
                is_first = True
                break

        if is_first:
            self._metrics[metric_name + self._separator + self.__name__()] += 1

        self._calls_cnt[metric_name + self._separator + self.__name__()] += 1

    def get(self):
        result = {}
        for metric_name, value in self._metrics.items():
            metric_name = metric_name.split("_")[0]
            result[metric_name + self._separator + self.__name__()] = \
                value / self._calls_cnt[metric_name + self._separator + self.__name__()]

        return result

    def __name__(self):
        return "FDARO"


class AverageLoc:
    """Метрика для оценки среднего места фейкового документа"""

    def __init__(self, ranking_metrics):
        self._separator = "_"
        self._metrics, self._calls_cnt = {}, {}
        for cur_metric in ranking_metrics:
            self._metrics[cur_metric.__name__() + self._separator + self.__name__()] = 0
            self._calls_cnt[cur_metric.__name__() + self._separator + self.__name__()] = 0

    def update(self, metric_name: str,
               ranking_list: list,
               fake_doc_label: int or list = 2):
        """
        Функция для обновления значений метрики

        Pparameters
        -------------
        metric_name:
        ranking_list:
        fake_doc_label:
        real_doc_label:

        Returns
        -------------
        """
        if isinstance(fake_doc_label, int):
            fake_doc_label = [fake_doc_label]

        if not isinstance(fake_doc_label, list):
            raise "The tags of fake documents must be 'int' or 'list!'"

        fake_doc_index = 0
        for ind, item in enumerate(ranking_list):
            if item[1] in fake_doc_label:
                self._metrics[metric_name + self._separator + self.__name__()] += (ind + 1)

        self._calls_cnt[metric_name + self._separator + self.__name__()] += 1

    def get(self):
        result = {}
        for metric_name, value in self._metrics.items():
            metric_name = metric_name.split("_")[0]
            result[metric_name + self._separator + self.__name__()] = \
                value / self._calls_cnt[metric_name + self._separator + self.__name__()]

        return result

    def __name__(self):
        return "AverageLoc"
