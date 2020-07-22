from dataset import Dataset
from policy_network import PolicyNet
from nets.perceptron import Perceptron
from env import Environment
from expeiment_settings import ExpSet
from state import State
from utils import Utility
import torch
import numpy as np
from tqdm import tqdm

class ReinforcementLearning:
    def __init__(self, dataset: Dataset, policy_net: PolicyNet, num_epochs, num_episode, steps):
        self.dataset = dataset
        self.epochs = num_epochs
        self.episodes = num_episode
        self.training = True
        self.steps = steps
        self.env = Environment(self.dataset.KG)

        self.policy_net = policy_net
        self.embedder = self.dataset.embedder
        self.optimizer = torch.optim.Adam(policy_net.parameters(), lr=ExpSet.learning_rate)

    @property
    def KG(self):
        return self.dataset.KG

    @property
    def T(self):
        return self.steps

    def train(self):
        self.training = True
        self.run()

    def test(self):
        self.training = False
        self.run()

    def run(self):
        total_acc, total_loss = [], []
        validate_acc, validate_loss = [], []

        epochs = self.epochs if self.training else 1
        for epoch in range(epochs):
            self.dataset.train(self.training)
            print('\n>>>>>> {}'.format('TRAINING: EPOCH {}'.format(epoch) if self.training else 'TESTING'))
            # Train
            for d in tqdm(self.dataset):
                _, q, e_s, answer = d
                #print(i, _, answer)
                ## Only for display, they were tuned during each episode
                acc, loss = self.learn(q, e_s, answer, self.training)
                total_acc.append(acc)   # int
                total_loss.append(loss) # float

            print('## {} loss: {}, accuracy: {}'.format('Training' if self.training else 'Testing',
                                                        torch.stack(total_loss).mean(), np.mean(total_acc)))
            total_acc, total_loss = [], []

            # Validate
            if not self.training:
                continue
            self.dataset.train(False)
            for d in tqdm(self.dataset):
                _, q, e_s, answer = d
                ## Only for display, they were tuned during each episode
                acc, loss = self.learn(q, e_s, answer, False)
                validate_acc.append(acc)
                validate_loss.append(loss)

            print('## Validation loss: {}, accuracy: {}'.format(torch.stack(validate_loss).mean(), np.mean(validate_acc)))
            validate_acc, validate_loss = [], []

    def learn(self, q, e_s, answer, training):
        '''
        :param q: 问题 :n x d
        :param e_s: 头实体 :str
        :param answer: 答案 :str
        :param training:
        :return:
        '''
        self.optimizer.zero_grad()
        loss = torch.tensor(0.)
        correct_predictions = 0
        total_predictions = self.episodes
        for episode in range(self.episodes):
            T = ExpSet.max_T
            d = ExpSet.dim
            n = len(q)

            ## 历史信息
            state_pool = {}      # S_t: T x State
            q_t = []             # q_t: T x n x d  问题池
            H_t = []             # H_t: T x d      历史信息
            action_pool = []     # r_t: T x d
            attention_wenghted_question_pool = torch.zeros((T, d))  # q_t_star: T x d
            #action_history = []
            initial_action = torch.zeros(d)
            H_t.append(self.policy_net.gru(initial_action.unsqueeze(dim=0).unsqueeze(dim=0)))

            ## 初始化环境
            self.env.new_question(State(q, e_s, e_s, 0, q_t, H_t), answer)
            state_pool[0] = self.env.state
            episode_reward = torch.tensor(0.)
            rewards = []
            action_probs = []

            for t in range(T):
                # 更新问题

                question_t = self.policy_net.slp(q)
                q_t.append(question_t)
                # 从环境获取动作空间
                possible_actions = self.env.get_possible_actions()
                if possible_actions is None:
                    break
                action_space = self.beam_search(possible_actions)
                action_space.append('cfsdfsdfsdfsdfe')
                action_embeddings = [self.embedder.get_relation_embedding(action) for action in action_space]
                # 移除在找不到relation embedding的action
                action_space = [action for action in filter(lambda x: x is not None,
                                      [action_space[i] if x is not None else None for i, x in enumerate(action_embeddings)])]
                action_embeddings = [e for e in filter(lambda x: x is not None, action_embeddings)]
                action_embeddings = torch.stack(action_embeddings)

                action_space, action_distribution = self.policy_net(action_embeddings, q_t[t], H_t[t])
                if action_space is None:
                    break
                action = self.sample_action(action_space, action_distribution)
                #action_history.append(action)
                # TODO: 提前获取以适应Cuda
                action_pool.append(self.dataset.embedder.get_relation_embedding(action))
                H_t.append(self.policy_net.gru(action_pool[t].unsqueeze(dim=0).unsqueeze(dim=0)))
                # 从环境获取奖励dw
                next_state, reward, reach_answer = self.env.step(action, q_t, H_t)
                episode_reward = ExpSet.gamma * episode_reward + reward
                state_pool[t+1] = next_state
                rewards.append(reward)
                action_probs.append(action_distribution)
                #if training and reach_answer: # 只有训练的时候环境中有监督信息
                #    break
            prediction = state_pool[len(state_pool)-1].e_t
            #print(' -- correct: {}     prediction: {}    rewards: {}'.format(answer, prediction, rewards))
            if not rewards:
                continue
            # TODO: 如果这个有用，那么先考虑直接在每一个tensor上做运算，不能直接stack起来因为prob的维度不一样
            #action_probs = (action_probs)
            if prediction == answer:
                correct_predictions = correct_predictions + 1
            ## TODO: log_prossibility loss function if needed
            loss = loss - episode_reward
            if training:
                loss.backward()
        acc = correct_predictions / total_predictions
        if training:
            self.optimizer.step()
        return acc, loss

    def beam_search(self, possible_actions, beam_size=3):
        actions_scores = []
        for action in possible_actions:
            expected_reward = self.env.get_action_reward(action)
            actions_scores.append((action, expected_reward))

        sorted_actions = sorted(actions_scores, key=lambda x: x[1].item())[:beam_size]
        beamed_actions = [action_score[0] for action_score in sorted_actions]

        return beamed_actions

    def sample_action(self, action_space, action_distribution):
        idx = torch.multinomial(action_distribution, 1).item()
        return action_space[idx]

    def save_model(self, filename):
        torch.save(self.policy_net.state_dict(), filename)

    def load_model(self, filename):
        self.policy_net.load_state_dict(torch.load(filename))
        self.policy_net.eval()

