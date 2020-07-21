import pandas as pd

class KnowledgeGraph:
    def __init__(self, path_KB):
        # 图结构，用邻接表存储
        #   For each element: ent_name -> array{relations}
        self._graph = {}

        try:
            self.df_kb = pd.read_csv(path_KB, sep='\s', header=None, names=['e_subject', 'relation', 'e_object'], engine='python')
            self._create_entity_set()
            self._create_graph_structure()

        except Exception as e:
            print('File path is wrong, with path_kb={}'.format(path_KB))

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
        try:
            return [x for x in self._graph[entity_name].keys()]
        except KeyError:
            return None

    def get_tail_entity(self, head_entity, rel):
        return self._graph[head_entity][rel]

    def get_triple(self, head_entity, rel):
        return [head_entity, rel, self.get_tail_entity(head_entity, rel)]

    @property
    def entities(self):
        return self._entities
