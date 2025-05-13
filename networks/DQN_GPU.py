import traci
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
import matplotlib.pyplot as plt
import os
import pickle

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define Transition structure
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

# Replay Memory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Neural Network
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

# DQN Agent for Traffic Lights
class TrafficLightDQN:
    def __init__(self):
        self.actions = [0, 1]  # 0 = prolong, 1 = change
        self.input_size = 5    # phase + 4 queues (north, south, east, west)
        self.output_size = len(self.actions)

        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.target_update = 100
        self.lr = 0.0005
        self.memory_capacity = 10000

        # Networks
        self.main_net = DQN(self.input_size, self.output_size).to(device)
        self.target_net = DQN(self.input_size, self.output_size).to(device)
        self.target_net.load_state_dict(self.main_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.main_net.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

        self.memory = ReplayMemory(self.memory_capacity)

        # Traffic parameters
        self.directions = {
            "north": ["-gneE8_0", "-gneE8_1", "-gneE8_2"],
            "south": ["gneE12_0", "gneE12_1", "gneE12_2"],
            "east": ["gneE7_0", "gneE7_1", "gneE7_2"],
            "west": ["-gneE10_0", "-gneE10_1", "-gneE10_2"]
        }

        self.queue_bins = [0, 3, 6, 9, 12]

        self.episode_rewards = []
        self.avg_wait_times = []

    def get_state(self):
        try:
            phase = traci.trafficlight.getPhase("gneJ2")
            queues = []
            for lanes in self.directions.values():
                queue = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in lanes)
                queues.append(queue)
            discretized = [np.digitize(q, self.queue_bins) for q in queues]
            return np.array([phase] + discretized, dtype=np.float32)
        except Exception as e:
            print(f"[Error in get_state] {e}")
            return np.zeros(self.input_size, dtype=np.float32)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.main_net(state_tensor)
        return q_values.argmax().item()

    def get_reward(self):
        try:
            total_wait = 0
            max_queue = 0
            total_vehicles = 0
            
            for lanes in self.directions.values():
                for lane in lanes:
                    total_wait += traci.lane.getWaitingTime(lane)
                    queue = traci.lane.getLastStepHaltingNumber(lane)
                    max_queue = max(max_queue, queue)
                    total_vehicles += traci.lane.getLastStepVehicleNumber(lane)
            
            # Normalize the reward components
            reward = - (total_wait / max(1, total_vehicles)) - (max_queue * 0.1)
            return float(reward)
        except Exception as e:
            print(f"[Error in get_reward] {e}")
            return 0.0

    def update_traffic_light(self, action, current_phase):
        new_phase = current_phase
        duration = 3
        try:
            if current_phase == 0:  # NS green
                new_phase = 0 if action == 0 else 1
                duration = 10 if action == 0 else 3
            elif current_phase == 1:  # NS yellow
                new_phase, duration = 4, 3
            elif current_phase == 4:  # EW green
                new_phase = 4 if action == 0 else 5
                duration = 10 if action == 0 else 3
            elif current_phase == 5:  # EW yellow
                new_phase, duration = 0, 3
            else:
                new_phase, duration = 0, 1
                
            traci.trafficlight.setPhase("gneJ2", new_phase)
            traci.trafficlight.setPhaseDuration("gneJ2", duration)
            return new_phase, duration
        except Exception as e:
            print(f"[Error in update_traffic_light] {e}")
            return current_phase, 1

    def train(self, episodes=400, max_steps=1000):
        step_counter = 0
        for episode in range(episodes):
            traci.start([
                "sumo-gui", "-c", "single.sumocfg",
                "--quit-on-end", "--waiting-time-memory", "1000",
                "--no-step-log", "--step-length", "1"
            ])
            traci.trafficlight.setPhase("gneJ2", 0)
            current_phase = 0
            total_reward = 0
            wait_times = []
            step = 0

            state = self.get_state()
            while traci.simulation.getMinExpectedNumber() > 0 and step < max_steps:
                action = self.choose_action(state)
                current_phase, duration = self.update_traffic_light(action, current_phase)

                for _ in range(int(duration)):
                    traci.simulationStep()
                    step += 1
                    if step >= max_steps or traci.simulation.getMinExpectedNumber() == 0:
                        break

                next_state = self.get_state()
                reward = self.get_reward()
                done = traci.simulation.getMinExpectedNumber() == 0 or step >= max_steps

                self.memory.push(state, action, next_state, reward, done)
                total_reward += reward
                wait_times.append(-reward)

                if len(self.memory) >= self.batch_size:
                    self.learn()

                state = next_state
                step_counter += 1

                if step_counter % self.target_update == 0:
                    self.target_net.load_state_dict(self.main_net.state_dict())

            traci.close()
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            avg_wait = np.mean(wait_times) if wait_times else 0
            self.episode_rewards.append(total_reward)
            self.avg_wait_times.append(avg_wait)

            print(f"Episode {episode+1}/{episodes} - "
                  f"Reward: {total_reward:.2f}, "
                  f"Avg Wait: {avg_wait:.2f}, "
                  f"Epsilon: {self.epsilon:.3f}")

            if (episode + 1) % 50 == 0:
                self.save_model()

        self.save_model()
        self.plot_results()

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
            
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Convert to tensors
        state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32).to(device)
        action_batch = torch.tensor(batch.action, dtype=torch.long).to(device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32).to(device)
        next_state_batch = torch.tensor(np.array(batch.next_state), dtype=torch.float32).to(device)
        done_batch = torch.tensor(batch.done, dtype=torch.float32).to(device)

        # Current Q values
        state_action_values = self.main_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Expected Q values
        with torch.no_grad():
            next_state_values = self.target_net(next_state_batch).max(1)[0]
            expected_state_action_values = reward_batch + (1 - done_batch) * self.gamma * next_state_values

        # Compute loss
        loss = self.loss_fn(state_action_values.squeeze(), expected_state_action_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.main_net.parameters(), 1.0)
        self.optimizer.step()

    def save_model(self):
        model_dir = "models/dqn"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        data = {
            'model_state': self.main_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'directions': self.directions,
            'stats': {
                'rewards': self.episode_rewards,
                'waits': self.avg_wait_times
            }
        }

        file_path = f"{model_dir}/traffic_model.pth"
        torch.save(data, file_path)
        print(f"Model saved at {file_path}")

    def plot_results(self):
        plt.figure(figsize=(15, 5))
        
        # Plot rewards
        plt.subplot(1, 2, 1)
        plt.plot(self.episode_rewards)
        plt.title("Rewards per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        
        # Plot wait times
        plt.subplot(1, 2, 2)
        plt.plot(self.avg_wait_times)
        plt.title("Average Wait Times")
        plt.xlabel("Episode")
        plt.ylabel("Wait Time (s)")
        
        plt.tight_layout()
        plt.savefig("dqn_results.png")
        plt.show()

if __name__ == "__main__":
    agent = TrafficLightDQN()
    print("=== Starting Training ===")
    agent.train(episodes=400)