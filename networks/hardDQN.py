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

# Replay memory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Dueling DQN (fully connected, no CNN)
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        # Shared feature layers
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        # Value stream
        self.value_fc = nn.Linear(128, 128)
        self.value_out = nn.Linear(128, 1)
        # Advantage stream
        self.advantage_fc = nn.Linear(128, 128)
        self.advantage_out = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        # Value stream
        value = F.elu(self.value_fc(x))
        value = self.value_out(value)
        # Advantage stream
        advantage = F.elu(self.advantage_fc(x))
        advantage = self.advantage_out(advantage)
        # Combine with dueling: Q = V + (A - mean(A))
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

# Traffic Light DQN Agent
class TrafficLightDQN:
    def __init__(self, tls_id="gneJ2", cv_rate=0.4):
        self.tls_id = tls_id
        self.actions = [0, 1]  # 0 = prolong, 1 = change
        # Lane configuration (from provided code, extended for multiple lanes)
        self.directions = {
            "north": ["-gneE8_0", "-gneE8_1", "-gneE8_2"],
            "south": ["gneE12_0", "gneE12_1", "gneE12_2"],
            "east": ["gneE7_0", "gneE7_1", "gneE7_2"],
            "west": ["-gneE10_0", "-gneE10_1", "-gneE10_2"]
        }
        self.num_lanes = sum(len(lanes) for lanes in self.directions.values())  # 12 lanes
        self.segments_per_lane = 20  # 8m cells, assuming 160m detection range
        self.input_size = (self.num_lanes * self.segments_per_lane * 3) + 1  # Positions, speeds, signals + phase
        self.output_size = len(self.actions)
        self.cv_rate = cv_rate  # Connected vehicle penetration rate (40%)

        # Hyperparameters (aligned with paper)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.target_update = 50
        self.lr = 1e-4  # From paper

        # Networks
        self.main_net = DQN(self.input_size, self.output_size).to(device)
        self.target_net = DQN(self.input_size, self.output_size).to(device)
        self.target_net.load_state_dict(self.main_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.main_net.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

        self.memory = ReplayMemory(20000)  # Larger memory for stability

        # Phase configuration (acyclic, from paper: NS green, EW green, transitions)
        self.phases = {
            0: "GGGGrrrrrrrr",  # NS green
            1: "yyyyrrrrrrrr",  # NS yellow
            4: "rrrrrrrrGGGG",  # EW green
            5: "rrrrrrrryyyy"   # EW yellow
        }

        # Metrics
        self.episode_rewards = []
        self.avg_wait_times = []
        self.max_delay = 1.0  # For reward normalization

    def get_state(self, training=True):
        """Partial DTSE: Flattened vector of CV positions, speeds, signals, and phase."""
        try:
            phase = traci.trafficlight.getPhase(self.tls_id)
            positions = np.zeros((self.num_lanes, self.segments_per_lane), dtype=np.float32)
            speeds = np.zeros((self.num_lanes, self.segments_per_lane), dtype=np.float32)
            signals = np.zeros((self.num_lanes, self.segments_per_lane), dtype=np.float32)
            lane_idx = 0
            for dir_lanes in self.directions.values():
                for lane in dir_lanes:
                    vehicles = traci.lane.getLastStepVehicleIDs(lane)
                    for veh in vehicles:
                        # Partial observability: sample CVs during deployment
                        if training or random.random() < self.cv_rate:
                            pos = traci.vehicle.getLanePosition(veh)  # Meters from stop line
                            segment = min(int(pos / 8), self.segments_per_lane - 1)  # 8m cells
                            speed = traci.vehicle.getSpeed(veh) / 13.89  # Normalize by 50 km/h
                            if segment >= 0:
                                positions[lane_idx, segment] = 1.0
                                speeds[lane_idx, segment] = speed
                    # Signal state: 1 for green, 0 for yellow/red
                    signals[lane_idx, :] = 1.0 if phase in [0, 4] else 0.0
                    lane_idx += 1
            # Flatten and append phase
            state = np.concatenate([
                positions.flatten(),
                speeds.flatten(),
                signals.flatten(),
                [phase / 5.0]  # Normalize phase (0-5)
            ])
            return state
        except Exception as e:
            print(f"[Error get_state] {e}")
            return np.zeros(self.input_size, dtype=np.float32)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.main_net(state_tensor)
        return q_values.argmax().item()

    def get_reward(self):
        """Total squared delay: Sum of squared speed-based delays, normalized."""
        try:
            v_max = 13.89  # 50 km/h in m/s
            total_squared_delay = 0.0
            for lanes in self.directions.values():
                for lane in lanes:
                    vehicles = traci.lane.getLastStepVehicleIDs(lane)
                    for veh in vehicles:
                        speed = traci.vehicle.getSpeed(veh)
                        delay = 1.0 - (speed / v_max)
                        total_squared_delay += delay ** 2
            # Normalize reward
            self.max_delay = max(self.max_delay, total_squared_delay) if total_squared_delay > 0 else self.max_delay
            reward = 1.0 - (total_squared_delay / self.max_delay)
            return reward
        except Exception as e:
            print(f"[Error get_reward] {e}")
            return 0.0

    def update_traffic_light(self, action, current_phase):
        """Update phase and duration based on action (acyclic cycle)."""
        new_phase = current_phase
        duration = 3
        try:
            if current_phase == 0:  # NS green
                new_phase = 0 if action == 0 else 1
                duration = 10 if action == 0 else 3
            elif current_phase == 1:  # NS yellow
                new_phase, duration = 4, 33
            elif current_phase == 4:  # EW green
                new_phase = 4 if action == 0 else 5
                duration = 10 if action == 0 else 3
            elif current_phase == 5:  # EW yellow
                new_phase, duration = 0, 33
            else:
                new_phase, duration = 0, 1
            traci.trafficlight.setPhase(self.tls_id, new_phase)
            traci.trafficlight.setPhaseDuration(self.tls_id, duration)
        except Exception as e:
            print(f"[Error update_traffic_light] {e}")
            new_phase, duration = 0, 1
        return new_phase, duration

    def train(self, episodes=400, max_steps=1000, sumo_config="single.sumocfg"):
        step_counter = 0
        for episode in range(episodes):
            traci.start([
                "sumo-gui", "-c", sumo_config,
                "--quit-on-end", "--waiting-time-memory", "1000",
                "--no-step-log", "--step-length", "1"
            ])
            traci.trafficlight.setPhase(self.tls_id, 0)
            current_phase = 0
            total_reward = 0
            wait_times = []
            step = 0

            state = self.get_state(training=True)
            while traci.simulation.getMinExpectedNumber() > 0 and step < max_steps:
                action = self.choose_action(state)
                current_phase, duration = self.update_traffic_light(action, current_phase)

                for _ in range(int(duration)):
                    traci.simulationStep()
                    step += 1
                    if step >= max_steps or traci.simulation.getMinExpectedNumber() == 0:
                        break

                next_state = self.get_state(training=True)
                reward = self.get_reward()
                done = traci.simulation.getMinExpectedNumber() == 0 or step >= max_steps

                self.memory.push(state, action, next_state, reward, done)
                total_reward += reward
                # Compute average waiting time for metrics
                total_wait = sum(sum(traci.lane.getWaitingTime(lane) for lane in lanes) for lanes in self.directions.values())
                wait_times.append(total_wait / max(1, len(wait_times) + 1))

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

            print(f"Episode {episode+1}/{episodes} — "
                  f"Reward: {total_reward:.2f}, "
                  f"Average Wait Time: {avg_wait:.2f}s, "
                  f"Epsilon: {self.epsilon:.3f}")

            if (episode + 1) % 50 == 0:
                self.save_model(episode=episode + 1)

        self.save_model()
        self.plot_results()

    def learn(self):
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.tensor(batch.state, dtype=torch.float32).to(device)
        action_batch = torch.tensor(batch.action, dtype=torch.long).to(device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32).to(device)
        next_state_batch = torch.tensor(batch.next_state, dtype=torch.float32).to(device)
        done_batch = torch.tensor(batch.done, dtype=torch.float32).to(device)

        state_action_values = self.main_net(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_state_values = self.target_net(next_state_batch).max(1)[0]
            expected_values = reward_batch + (1 - done_batch) * self.gamma * next_state_values

        loss = self.loss_fn(state_action_values, expected_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self, model_type="dqn", episode=None):
        model_dir = f"models/{model_type}"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        file_path = f"{model_dir}/traffic_model_ep{episode}.pkl" if episode else f"{model_dir}/traffic_model_final.pkl"
        data = {
            'main_net_state': self.main_net.state_dict(),
            'target_net_state': self.target_net.state_dict(),
            'directions': self.directions,
            'stats': {
                'rewards': self.episode_rewards,
                'waits': self.avg_wait_times
            }
        }
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
        print(f"{model_type.capitalize()} model saved at {file_path}")

    def plot_results(self):
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.episode_rewards)
        plt.title("Rewards per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.subplot(1, 2, 2)
        plt.plot(self.avg_wait_times)
        plt.title("Average Waiting Time per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Waiting Time (s)")
        plt.tight_layout()
        plt.savefig("dqn_results.png")
        plt.close()

# Run training
if __name__ == "__main__":
    agent = TrafficLightDQN(tls_id="gneJ2", cv_rate=0.0)
    print("=== Starting Training ===")
    agent.train(episodes=400, sumo_config="single.sumocfg")