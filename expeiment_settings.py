import argparse

class ExperimentSettingsMeta(type):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._relation_embedding_dimension = 50
        self._word_embedding_dimension = 50
        self._dim = 50
        self._max_T = 10
        self._gamma = 0.8
        self._learning_rate = 1e-3
        self._enable_cuda = False
        self._path_KB = './datasets/2H-kb.txt'
        self._path_QA = './datasets/PQ-2H.txt'
        self._enable_cache = True
        self._use_attention = True
        self._use_perceptron = True
        self._epochs = 200
        self._episodes = 1
        self._seed = 123456

    @property
    def seed(self):
        return self._seed

    @property
    def epochs(self):
        return self._epochs

    @property
    def episodes(self):
        return self._episodes

    @property
    def use_attention(self):
        return self._use_attention

    @property
    def use_perceptron(self):
        return self._use_perceptron

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

    @property
    def dim(self):
        return self._dim

    @property
    def enable_cuda(self):
        return self._enable_cuda

    @property
    def path_KB(self):
        return self._path_KB

    @property
    def path_QA(self):
        return self._path_QA

    @property
    def enable_cache(self):
        return self._enable_cache


class ExperimentSettings(metaclass=ExperimentSettingsMeta):
    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--cuda', dest='cuda', action='count', default=0,
                            help='Enable CUDA.')
        parser.add_argument('--seed', dest='seed', type=int, default=12345, help='Random seed.')
        parser.add_argument('--epochs', dest='epochs', type=int, default=200,
                            help='Number of epochs to train. (default 200)')
        parser.add_argument('--episodes', dest='episodes', type=int, default=1,
                            help='Number of episodes to train. (default 1)')
        parser.add_argument('--lr', dest='lr', type=float, default=0.001,
                            help='Initial learning rate. (default 0.001)')
        parser.add_argument('--using-attention', dest='attention', action='count', default=0,
                            help='Using attention in policy network. (default True)')
        parser.add_argument('--using-perceptron', dest='perceptron', action='count', default=0,
                            help='Using of episodes to train. (default True)')
        parser.add_argument('--dataset-kg', dest='dskg', action='store', type=str, default='./datasets/2H-kb.txt',
                            help='Path to KG file. (default ./datasets/2H-kb.txt)')
        parser.add_argument('--dataset-qa', dest='dsqa', action='store', type=str, default='./datasets/PQ-2H.txt',
                            help='Path to QA file. (default ./datasets/PQ-2H.txt)')
        parser.add_argument('--load-model',  dest='load_model',action='store', type=str, default='',
                            help='Load saved model')
        parser.add_argument('--gamma', dest='gamma', action='store', type=float, default=0.8,
                            help='Gamma (Discount factor of reward on reinforcement learning, default 0.8)')
        parser.add_argument('--max-hop', dest='T', action='store', type=float, default=3,
                            help='Max hop on graph (default 3)')
        parser.add_argument('--disable-cache', dest='disable_cache',  action='count', default=0,
                            help='Disable cache on dataset')
        args = parser.parse_args()
        ExperimentSettings._enable_cuda = args.cuda > 0
        ExperimentSettings._seed = args.seed
        ExperimentSettings._epochs = args.epochs
        ExperimentSettings._episodes = args.episodes
        ExperimentSettings._lr = args.lr
        ExperimentSettings._use_attention = args.attention > 0
        ExperimentSettings._use_perceptron = args.perceptron > 0
        ExperimentSettings._path_KB = args.dskg
        ExperimentSettings._path_QA = args.dsqa
        ExperimentSettings._gamma = args.gamma
        ExperimentSettings._max_T = args.T
        ExperimentSettings._enable_cache = args.disable_cache == 0

