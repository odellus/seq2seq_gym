import gym
import datasets
import torch
import torch.nn as nn
from gym import spaces

from transformers import (
    LEDTokenizerFast,
    LEDForConditionalGeneration,
    LEDForSequenceClassification,
)

def reset_last_layer(critic):
    in_features = critic.classification_head.out_proj.in_features
    out_features = 1
    critic.classification_head.out_proj = nn.Linear(in_features, out_features)
    return critic

class ObservationPool:
    def __init__(self, dataset, tokenizer, cfg):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.size = len(dataset)
        self.seq_len = cfg['obs_seq_len']
        self.prepare_dataset()

    def prepare_dataset(self):
        self.dataset = self.dataset.shuffle()
        self.replay_buffer = []
        for x in self.dataset:
            self.replay_buffer.append(x)

    def get_observation(self):
        res = None
        if len(self.replay_buffer) > 0:
            input_text =  self.replay_buffer.pop().get('input_text')
            res = self.tokenizer(
                input_text, 
                padding = 'max_length',
                max_length = self.seq_len,
                return_tensors = 'pt',
            ).input_ids
            self.size -= 1
        return res

class SummarizeEnv(gym.Env):
    '''
    '''
    def __init__(self, tokenizer, dataset, critic, cfg):
        '''
        '''
        super(SummarizeEnv, self).__init__()
        # Pull in some configurations from cfg
        self.batch_size = cfg['batch_size']
        self.obs_seq_len = cfg['obs_seq_len']
        self.act_seq_len = cfg['act_seq_len']
        # Create a replay buffer out of the HF dataset
        self.observation_pool = ObservationPool(dataset, tokenizer, cfg)

        self.action_space = spaces.Box(
            low = 0,
            high = len(tokenizer) - 1,
            shape = [self.batch_size, self.act_seq_len]
        )

        self.observation_space = spaces.Box(
            low = 0,
            high = len(tokenizer) - 1,
            shape = [self.batch_size, self.obs_seq_len]
        )

        self.critic = critic.to('cuda')
    
    def get_observation(self):
        return self.observation_pool.get_observation()

    def step(self, action):
        observation = self.get_observation()
        with torch.no_grad():
            output = self.critic(action.to('cuda'))
            reward = output.logits.to('cpu').numpy()[0,0] # Get output from classifier as reward
        done = True if observation is None else False
        info_ = None
        return observation, reward, done, info_
    
    def reset(self):
        observation = self.get_observation()
        return observation
    
def main():
    dataset = datasets.Dataset.from_dict({'input_text': ['Some text', 'some more text']})
    cfg = {
        'obs_seq_len': 4096,
        'act_seq_len': 200,
        'batch_size': 1,
    }
    tokenizer = LEDTokenizerFast.from_pretrained('allenai/led-base-16384')
    critic = LEDForSequenceClassification.from_pretrained('allenai/led-base-16384')
    critic = critic.to('cuda')
    critic.eval()
    actor = LEDForConditionalGeneration.from_pretrained('allenai/led-base-16384')
    actor = actor.to('cuda')
    critic = reset_last_layer(critic)
    sum_env = SummarizeEnv(tokenizer, dataset, critic, cfg)
    obs = sum_env.reset()
    obs = obs.to('cuda')
    action = actor.generate(input_ids = obs)
    action = action.to('cuda')
    observation, reward, done, info_ = sum_env.step(action)
    print(tokenizer.decode(obs[0]))
    print(tokenizer.decode(observation[0]))
    print(done)
    print(tokenizer.decode(action[0]))
    print(reward)

if __name__ == '__main__':
    main()