from knowledge_graph import *

class EntityLinker:
    def __init__(self, KG: KnowledgeGraph):
        self.KG = KG

    def parse_question(self, question: str):
        '''
        将问题分为以单词为单位的分词列表，并找出头实体(e_s)【问题中在KG实体的单词中最长的单词】，并替换为<e>

        :param question: 问题字符串
        :return: (问题分词列表（字符串），头实体e_s（字符串）)
        '''
        modified_question_list = []
        entity_list = []

        for idx, item in enumerate(question.split(' ')):
            if item in self.KG.entities:
                entity_list.append(item)

        entity = max(entity_list, key=len)

        for item in question.split(' '):
            if item == entity:
                modified_question_list.append('<e>')
            else:
                if len(item.split('_')) > 0:
                    for x in item.split('_'):
                        if x != '':
                            modified_question_list.append(x)
                else:
                    modified_question_list.append(item)
        return (modified_question_list, entity)

