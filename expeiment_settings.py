class ExperimentSettingsMeta(type):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._relation_embedding_dimension = 50
        self._word_embedding_dimension = 50
        self._max_T = 10
        self._gamma = 0.8
        self._learning_rate = 1e-3

    @property
    def relation_embedding_dimension(self):
        return self._relation_embedding_dimension

    @property
    def word_embedding_dimension(self):
        return self._word_embedding_dimension

    @property
    def max_T(self):
        return self._max_T

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def gamma(self):
        return self._gamma


class ExperimentSettings(metaclass=ExperimentSettingsMeta):
    pass
