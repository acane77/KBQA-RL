import pandas as pd
import torch
from knowledge_graph import *
from reward import *

class Environment:
    # ## action: relation
    #    state: entity
    def __init__(self, KG: KnowledgeGraph, reward_function :Reward = CosineSimiliarityReward(gamma=0.8)):
        self.KG = KG
        self.state = None
        self.start_state = None
        self.reward_function = reward_function
        self.answer = ''
        self.step_count = 0

    def new_question(self, start_state: State, answer: str):
        self.state = start_state
        self.start_state = start_state
        self.answer = answer
        self.step_count = 0
        return self

    def step(self, action :str, updated_questions: torch.Tensor, encoded_history: torch.Tensor):
        assert self.state is not None, "No answer and start state set"
        self.step_count = self.step_count + 1
        next_entity = self.KG.get_tail_entity(self.state.current_entity, action)
        next_state = State(self.state.q, self.state.e_s, next_entity,
                           self.state.t+1, q_t=updated_questions, H_t=encoded_history)
        reward = self.reward_function(next_state, self.answer)
        reach_answer = self.answer == next_entity
        return next_state, reward, reach_answer

    def get_possible_actions(self):
        action_space = KG.get_relations_of(self.state.current_entity)
        if len(action_space) > 0:
            return action_space
        return None

    @property
    def t(self):
        return self.step_count

if __name__ == '__main__':
    KG = KnowledgeGraph('dataset/2H-kb.txt', 'dataset/PQ-2H.txt')
    head_ent_list = [x for x in (KG._graph.keys())]
    env = Environment(KG)
    s_t = State(None, head_ent_list[0], head_ent_list[0], 0, torch.zeros((2,2,2)), torch.zeros((2,2,2)))
    env.new_question(s_t, 'maximilian_ii_of_bavaria\n')
    actions = (env.get_possible_actions())
    print(env.step(actions[0], torch.zeros((2,2)), torch.zeros((2,2))))
