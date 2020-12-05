import torch
from multiprocessing_env import SubprocVecEnv

from ACAgent import format_observations

def compute_returns(next_value, rewards, masks, gamma=0.99):
    
    '''Calculate the return $\\sum_{t=0}^T r_t + \\gamma^t  V_{\\omega}(s_{t+1})$'''

    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


# environment
def make_env(env_name, seed, rank):
    def _thunk():
        env = gym.make(env_name)
        env.seed(seed+rank)
        return env

    return _thunk



def save_checkpoint(
    fname,
    policy_model,
    optimizer,
    current_step_number,
    average_rewards,
    train_loss,
    download=False
):
    "saves the model and optimizer state_dict along with the losses and average_rewards per episode"

    checkpoint = {
        'current_step_number': current_step_number,
        'average_rewards': average_rewards,
        'train_loss': train_loss,
        'model_state_dict': policy_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }

    torch.save(checkpoint, path(fname))
    if download and Colab: files.download(path) # not working well


def dict_to_device(dic):
    for key, value in dic.items():
        dic[key] = dic[key].to(device)
    return dic

def train(env, agent, flags):
    """"""

    # set random seeds (for reproducibility)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    envs = [make_env(flags['env'], seed, i) for i in range(num_envs)]
    envs = SubprocVecEnv(envs)

    # instantiate the policy and optimiser
    num_inputs  = envs.observation_space.shape[0]
    num_outputs = envs.action_space.n
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    current_step_number = 0
    test_rewards = []
    state = envs.reset()

    
    while current_step_number < flags['max_steps']:
        
        log_probs = []
        values    = []
        rewards   = []
        masks     = []
        entropy = 0

        for _ in range(flags['num_step_td_update']):

            # sample an action from the distribution
            action = agent.act(state)
            # take a step in the environment
            next_state, reward, done, _ = envs.step(action.cpu().numpy())
                
            # compute the log probability
            log_prob = dist.log_prob(action)
            # compute the entropy
            entropy += dist.entropy().mean()
            
            # save the log probability, value and reward 
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
            masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))



            # if done, save episode rewards

            state = next_state
            current_step_number += 1
            
            if current_step_number % 1000 and flags['plot_test'] == 0:
                test_rewards.append(np.mean([test_env(model) for _ in range(10)]))
                plot(current_step_number, test_rewards)

        next_state = torch.FloatTensor(next_state).to(device)
        _, next_value = model(next_state)
   
        # calculate the discounted return of the episode
        returns = compute_returns(next_value, rewards, masks)

        log_probs = torch.cat(log_probs)
        returns   = torch.cat(returns).detach()
        values    = torch.cat(values)

        advantage = returns - values

        actor_loss  = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        # loss function
        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return rewards