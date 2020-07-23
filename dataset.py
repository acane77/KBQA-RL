import torch
import pandas as pd
from knowledge_graph import KnowledgeGraph
from embedding import Embedder
from expeiment_settings import ExpSet
from utils import Utility

class Dataset:
    '''
    数据集：

        每一条数据为 [问题的字符串表示， 问题的embedding， 头实体， 答案]
    '''
    def __init__(self, path_KG, path_QA, split_ratio=0.8, using_cache=True):
        self.KG = KnowledgeGraph(path_KG)
        self.embedder = Embedder()
        self.training = True  # 指定是否是训练阶段
        self._iter_i = 0
        self._split_ratio = split_ratio

        # try to load from cache
        if using_cache and Utility.Binary.exists('dataset'):
            self.questions = Utility.Binary.load('dataset')
            print('{} questions loaded'.format(len(self.questions)))
            return

        # read the original questions
        questions = pd.read_csv(path_QA, sep='\t', header=None, names=['question_sentence', 'answer_set', 'answer_path'])
        questions['answer'] = questions['answer_set'].apply(lambda x: x.split('(')[0])
        questions['q_split'] = questions['question_sentence'].apply(lambda x: x.lower().split(' '))
        questions['answer'] = questions['answer_set'].apply(lambda x: x.split('(')[0])
        questions['e_s'] = questions['answer_path'].apply(lambda x: x.split('#')[0])
        # find head entity e_s, answer, and question_list by parsing the question_sentence
        questions['q_str'] = [self.parse_question(row['question_sentence'].split('?')[0], row['e_s'])
                              for idx, row in questions.iterrows()]

        # 对问题编码
        # NOTE: 这里是正对小数据集采取的空间换时间的方式，避免每一次都重新embed问题，对于大数据集需要单独处理数据
        questions['q'] = questions['q_str'].apply(lambda q: self.embed_question(q))

        question_list = questions[['q_str', 'q', 'e_s', 'answer']].values.tolist()
        question_list = [tuple(x) for x in question_list]
        self.questions = question_list
        print('{} questions loaded'.format(len(question_list)))

        if using_cache:
            Utility.Binary.save('dataset', question_list)

    def embed_question(self, question):
        n, idx = len(question), 0
        q_emb = torch.zeros((n, ExpSet.word_embedding_dimension))
        for word in question:
            if word == '<e>':
                continue
            w_emb = self.embedder.get_word_embedding(word)
            if w_emb is not None:
                q_emb[idx] = w_emb
                idx = idx + 1
        return q_emb[:idx]

    def embed_relation(self, relation):
        return self.embedder.get_relation_embedding(relation)

    def parse_question(self, question: str, e_s: str):
        '''
        将问题分为以单词为单位的分词列表，并找出头实体(e_s)【问题中在KG实体的单词中最长的单词】，并替换为<e>

        :param question: 问题字符串
        :param e_s: 头实体
        :return: 问题分词列表（字符串）
        '''
        modified_question_list = []
        for item in question.split(' '):
            if item == e_s:
                modified_question_list.append('<e>')
            else:
                if len(item.split('_')) > 0:
                    for x in item.split('_'):
                        if x != '':
                            modified_question_list.append(x)
                else:
                    modified_question_list.append(item)
        return modified_question_list

    def __iter__(self):
        return self

    def __next__(self):
        try:
            d = self[self._iter_i]
            self._iter_i = self._iter_i + 1
            return d
        except IndexError:
            self._iter_i = 0
            raise StopIteration()

    def __getitem__(self, item):
        if item >= self.size:
            raise IndexError('index out of bound, size={}, item={}, training={}'.format(self.size, item, self.training))
        if self.training:
            return self.questions[item]
        return self.questions[self.training_size + item]

    def __len__(self):
        return self.size

    @property
    def size(self):
        if self.training:
            return self.training_size
        return self.testing_size

    @property
    def data_size(self):
        return len(self.questions)

    @property
    def testing_size(self):
        return self.data_size - self.training_size

    @property
    def training_size(self):
        return int(self._split_ratio * self.data_size)

    def train(self, _train=True):
        self.training = _train
        self._iter_i = 0

if __name__ == '__main__':
    dataset = Dataset(ExpSet.path_KB, ExpSet.path_QA, 0.8)
    print(dataset[0])