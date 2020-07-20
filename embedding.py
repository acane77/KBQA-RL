import numpy as np
import pandas as pd
import csv
import json

class Embedder:
    def __init__(self):
        freebase_path = "./dataset/"
        glove_data_file = "{}glove.6B/glove.6B.50d.txt".format(freebase_path)
        relation2id_file = '{}knowledge_graphs/relation2id.txt'.format(freebase_path)
        embedding_relation_file = '{}embeddings/dimension_50/transe/relation2vec.bin'.format(freebase_path)

        print('Importing Glove Word Embeddings')
        words = pd.read_csv(glove_data_file, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
        self.word_embeddings = words

        print('Getting relation2id')
        self.relation2id = self.process_relation(relation2id_file)

        print('Getting relation Embeddings')
        self.relation_embedding = np.memmap(embedding_relation_file , dtype='float32', mode='r')

    def process_relation(self, relation2id_file: str):
        file = csv.reader(relation2id_file, delimiter='\t')
        relation2id = {}
        for row in file:
            item_list = row[0].split('.')
            # 由于数据集中只有 people.person.*** 的关系，所以只选择这一部分加入关系列表
            if 'people' in item_list:
                relation2id[item_list[-1]] = row[1]
        j = json.dumps(relation2id)
        fileObject = open("relation2id.json", 'w')
        fileObject.write(j)
        fileObject.close()
        return relation2id

    def get_word_embedding(self, word: str):
        '''
        从glove数据中获取词向量（用于问题的编码）

        :param word: word
        :return: word embedding
        '''
        return self.word_embeddings.loc[word].values

    def get_relation_embedding(self, rel: str):
        '''
        从freebase中获取关系向量
        :param rel: 关系字符串
        :return: 关系embedding
        '''
        try:
            index = int(self.relation2id[rel])
            vector_index = index * 50
            return self.relation_embedding[vector_index:vector_index+50]
        except Exception as e:
            return

