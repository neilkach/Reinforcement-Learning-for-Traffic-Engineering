from collections import deque
import torch
torch.manual_seed(1)
import numpy as np
import ddpg
from importlib import reload
reload(ddpg)
from ddpg import DDPG
import matplotlib.pyplot as plt
import sys
import subprocess
import re

# problem parameters
num_tunnels = 4
state_dim = 3 * num_tunnels
action_dim = 1 * num_tunnels

num_agents = 20

# hyperparameters
hidden_dim = 12
batch_size = 256
actor_lr = 1e-4
critic_lr = 2e-4
tau = 0.001
gamma = 0.9
epsilon = 0.1
# use Gaussian noise
noise_scale = 0.5*torch.randn(action_dim)
noise_decay = 0.998

episodes = 300
max_steps = 300

#make it generalizable to any number of weights
def calc_reward(w, x, num_tunnels):
  cost = -0.1 * torch.dot(w, x[num_tunnels:num_tunnels*2])
  net_perf = torch.dot(w, x[0:num_tunnels]) - 1
  err = torch.abs(w - x[num_tunnels*2:num_tunnels*3])
  user_pref = -torch.sum(err)
  return net_perf + user_pref + cost

def plot_rewards(rewards):
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward per Episode')
    plt.show()

def plot_losses(actor_losses, critic_losses):
    plt.plot(actor_losses, label='Actor Loss')
    plt.plot(critic_losses, label='Critic Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Loss per Episode')
    plt.legend()
    plt.show()

def plot_weights(states, actions):
    states = np.array(states)
    actions = np.array(actions)
    for i in range(num_tunnels):
        plt.plot(states[:, i], label=f'State {i+1}')
    for i in range(num_tunnels):
        plt.plot(actions[:, i], label=f'Action {i+1}')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.title('Weights per Episode')
    plt.legend()
    plt.show()

def ping(ip):
    try:
        # Execute the ping command
        result = subprocess.run(['ping', '-c', '1', ip], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # Extract the time using regular expression
        match = re.search(r'time=(\d+\.\d+) ms', result.stdout)
        if match:
            return match.group(1)
        else:
            return None
    except Exception as e:
        print(f"Error pinging {ip}: {e}")
        return None

def main():
     # IP addresses to ping
    ips = ['192.168.0.0', '192.168.0.1', '192.168.0.2', '192.168.0.3']
    
    # Ping each IP address and store the times
    times = [ping(ip) for ip in ips]

    if None in times:
        print("Failed to retrieve ping times for one or more IP addresses.")
        sys.exit(1)
    
    # Print the times (or you can pass them to another function/script as required)
    print(f"Ping times: {times}")
    # intialize learning agent
    agent = DDPG(state_dim, action_dim, hidden_dim=hidden_dim, buffer_size=8000, batch_size=batch_size,
                    actor_lr=actor_lr, critic_lr=critic_lr, tau=tau, gamma=gamma)
    cost = torch.FloatTensor([50,10,10,10])
    user_pref  = torch.FloatTensor([0.4,0.4,0.1,0.1])
    fixed_x = torch.cat((cost, user_pref))

    rewards_window = deque(maxlen=100)
    rewards = []
    actor_losses = []
    critic_losses = []
    states = []
    actions = []

    for episode in range(episodes):
        agent_rewards = np.zeros(num_agents)
        performance_raw_weights = torch.rand(num_tunnels)
        state = torch.cat((performance_raw_weights, fixed_x))
        # state = performance_raw_weights
        actor_step_losses = []
        critic_step_losses = []
        past_reward = 0
        for steps in range(max_steps):
            action = agent.act(state, noise_scale)
            reward = calc_reward(action, state, num_tunnels=4) 
            done = int(steps == max_steps - 1)
            performance_raw_weights = torch.rand(num_tunnels)
            next_state = torch.cat((performance_raw_weights, fixed_x))
            # for i in range(num_agents):
            #      agent.store_transition(state, action, reward, next_state, done)
            # agent.store_transition(state, action, reward, next_state, done)
            # actor_loss, critic_loss = agent.learn()
            actor_loss = -reward * epsilon
            agent.actor.optimizer.zero_grad()
            actor_loss.backward()
            agent.actor.optimizer.step()

            if actor_loss is not None:
                actor_step_losses.append(actor_loss.detach().numpy())
                #critic_step_losses.append(critic_loss.detach().numpy())
            
            state = next_state
            agent_rewards += reward.detach().numpy()
            noise_scale *= noise_decay
        
        states.append(performance_raw_weights.detach().numpy())
        actions.append(action.detach().numpy())
        avg_reward = np.mean(agent_rewards)
        rewards_window.append(avg_reward)
        rewards.append(avg_reward)

        if actor_step_losses:
            actor_losses.append(np.mean(actor_step_losses))
            critic_losses.append(np.mean(critic_step_losses))

        if episode % 10 == 0:
            print(action)
            print(f'Episode {episode}, Reward: {avg_reward:.2f}, Avg Reward: {np.mean(rewards_window):.2f}')
        
    #save final network
    torch.save(agent.actor.state_dict(), 'final_models/actor_final.pth')
    torch.save(agent.critic.state_dict(), 'final_models/critic_final.pth')

if __name__ == '__main__':
    main()

