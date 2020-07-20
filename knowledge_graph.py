import pandas as pd

class KnowledgeGraph:
    def __init__(self, path_KB, path_QA):
        # 图结构，用邻接表存储
        #   For each element: ent_name -> array{relations}
        self._graph = {}

        try:
            self.df_qa = pd.read_csv(path_QA, sep='\t', header=None, names=['question_sentence', 'answer_set', 'answer_path'])
            self.df_qa['answer'] = self.df_qa['answer_set'].apply(lambda x: x.split('(')[0])
            self.df_qa['q_split'] = self.df_qa['question_sentence'].apply(lambda x: x.lower().split(' '))
            self.df_kb = pd.read_csv(path_KB, sep='\s', header=None, names=['e_subject', 'relation', 'e_object'])
            self._create_entity_set()
            self._create_graph_structure()

        except Exception as e:
            print('File path is wrong, with path_kb={}, path_qa={}'.format(path_KB, path_QA))

        print('KG Loaded: {} entities loaded'.format(len(self._entities)))

    def _create_entity_set(self):
        subject_set = set(self.df_kb['e_subject'].unique())
        object_set = set(self.df_kb['e_object'].unique())
        self._entities = subject_set.union(object_set)

    def _create_graph_structure(self):
        for idx, triple in self.df_kb.iterrows():
            head_ent, rel, tail_ent = triple['e_subject'], triple['relation'], triple['e_object']
            if not head_ent in self._graph:
                self._graph[head_ent] = {}
            self._graph[head_ent][rel] = tail_ent

    def get_relations_of(self, entity_name):
        return [x for x in self._graph[entity_name].keys()]

    def get_tail_entity(self, head_entity, rel):
        return self._graph[head_entity][rel]

    def get_triple(self, head_entity, rel):
        return [head_entity, rel, self.get_tail_entity(head_entity, rel)]

    @property
    def entities(self):
        return self._entities