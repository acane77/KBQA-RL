import torch

class State:
    def __init__(self,
                 q:   torch.Tensor,
                 e_s: str,
                 e_t: str,
                 t:   int,
                 q_t, H_t):
        '''
        :param q:   问题嵌入 :tensor[n x d]
        :param e_s: 头实体 :str
        :param e_t: 当前实体 :str
        :param t:   当前的跳数 :int (从0开始)
        :param q_t: 问题的历史记录 :tensor[T x n x d]
        :param H_t: 编码后的action历史记录 :tensor[T x n x d]
        '''
        self.q = q
        self.e_s = e_s
        self.e_t = e_t
        self.t = t
        self.q_t = q_t
        self.H_t = H_t

    @property
    def question(self):
        return self.q

    @property
    def head_entity(self):
        return self.e_s

    @property
    def current_entity(self):
        return self.e_t