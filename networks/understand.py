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

#instruction pour utiliser GPU a la
# Définir la structure Transition
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

# Classe ReplayMemory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Classe DQN
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Classe TrafficLightDQN (adaptation de TrafficLightQLearning pour DQN)
class TrafficLightDQN:
    def __init__(self):
        self.actions = [0, 1]  # 0 = prolonger, 1 = changer
        self.input_size = 5  # phase + 4 directions (north, south, east, west)
        self.output_size = len(self.actions)

        # Hyperparamètres
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32
        self.target_update = 100  # Mettre à jour le réseau cible toutes les 100 étapes

        # Réseaux DQN
        self.main_net = DQN(self.input_size, self.output_size)
        self.target_net = DQN(self.input_size, self.output_size)
        self.target_net.load_state_dict(self.main_net.state_dict())
        self.target_net.eval()  # Mode évaluation pour le réseau cible
        self.optimizer = optim.Adam(self.main_net.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

        # Mémoire de replay
        self.memory = ReplayMemory(10000)

        # Directions (comme dans TrafficLightQLearning)
        self.directions = {
            "north": ["-gneE8_0", "-gneE8_1", "-gneE8_2"],
            "south": ["gneE12_0", "gneE12_1", "gneE12_2"],
            "east": ["gneE7_0", "gneE7_1", "gneE7_2"],
            "west": ["-gneE10_0", "-gneE10_1", "-gneE10_2"]
    }

        self.queue_bins = [0, 3, 6, 9, 12]

        # Statistiques
        self.episode_rewards = []
        self.avg_wait_times = []

    def get_state(self):
        try:
            phase = traci.trafficlight.getPhase("gneJ2")
            queues = []
            for lanes in self.directions.values():
                queue = 0
                for lane in lanes:
                    queue += traci.lane.getLastStepHaltingNumber(lane)
                queues.append(queue)
            discretized = [np.digitize(q, self.queue_bins) for q in queues]
            state = [phase] + discretized
            return np.array(state, dtype=np.float32)
        except Exception as e:
            print(f"Erreur get_state: {e}")
            return np.zeros(5, dtype=np.float32)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.main_net(state_tensor)
        return q_values.argmax().item()

    def get_reward(self):
        try:
            total_wait = 0
            max_queue = 0
            for lanes in self.directions.values():
                for lane in lanes:
                    total_wait += traci.lane.getWaitingTime(lane)
                    max_queue = max(max_queue, traci.lane.getLastStepHaltingNumber(lane))
            reward = -(total_wait + max_queue * 2)
            return reward / 100.0  # Normalisation
        except Exception as e:
            print(f"Erreur get_reward: {e}")
            return 0.0

    def update_traffic_light(self, action, current_phase):
        new_phase = current_phase
        duration = 3
        try:
            if current_phase == 0:  # NS vert
                if action == 0:
                    duration = 10
                else:
                    new_phase = 1  # NS jaune
                    duration = 3
            elif current_phase == 1:  # NS jaune
                new_phase = 4  # EW vert
                duration = 33
            elif current_phase == 4:  # EW vert
                if action == 0:
                    duration = 10
                else:
                    new_phase = 5  # EW jaune
                    duration = 3
            elif current_phase == 5:  # EW jaune
                new_phase = 0  # retour à NS vert
                duration = 33
            else:
                new_phase = 0
                duration = 1
            traci.trafficlight.setPhase("gneJ2", new_phase)
            traci.trafficlight.setPhaseDuration("gneJ2", duration)
        except Exception as e:
            print(f"Erreur update_traffic_light: {e}")
            new_phase = 0
            duration = 1
        return new_phase, duration

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

                self.memory.push(
                    state, action, next_state, reward, done
                )
                total_reward += reward
                wait_times.append(-reward * 100)  # Dé-normalisation pour stats

                if len(self.memory) >= self.batch_size:
                    transitions = self.memory.sample(self.batch_size)
                    batch = Transition(*zip(*transitions))

                    state_batch = torch.tensor(batch.state, dtype=torch.float32)
                    action_batch = torch.tensor(batch.action, dtype=torch.long)
                    reward_batch = torch.tensor(batch.reward, dtype=torch.float32)
                    next_state_batch = torch.tensor(batch.next_state, dtype=torch.float32)
                    done_batch = torch.tensor(batch.done, dtype=torch.float32)

                    # Calcul des valeurs Q
                    state_action_values = self.main_net(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
                    with torch.no_grad():
                        next_state_values = self.target_net(next_state_batch).max(1)[0]
                        expected_values = reward_batch + (1 - done_batch) * self.gamma * next_state_values

                    # Calcul de la perte
                    loss = self.loss_fn(state_action_values, expected_values)

                    # Optimisation
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                state = next_state
                step_counter += 1

                # Mise à jour du réseau cible
                if step_counter % self.target_update == 0:
                    self.target_net.load_state_dict(self.main_net.state_dict())

            traci.close()
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            avg_wait = np.mean(wait_times) if wait_times else 0
            self.episode_rewards.append(total_reward)
            self.avg_wait_times.append(avg_wait)

            print(f"Episode {episode+1}/{episodes}: "
                  f"Reward={total_reward:.1f}, "
                  f"Avg Wait={avg_wait:.1f}s, "
                  f"Epsilon={self.epsilon:.3f}")

            if (episode + 1) % 50 == 0:
                self.save_model()

        self.save_model()
        self.plot_results()

    def save_model(self):
        if not os.path.exists('models'):
            os.makedirs('models')
        torch.save({
            'main_net': self.main_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'rewards': self.episode_rewards,
            'waits': self.avg_wait_times
        }, "models/dqn_traffic_model.pt")
        print("Modèle sauvegardé")

    def plot_results(self):
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.episode_rewards)
        plt.title("Récompenses par épisode")
        plt.subplot(1, 2, 2)
        plt.plot(self.avg_wait_times)
        plt.title("Attente moyenne")
        plt.tight_layout()
        plt.savefig("dqn_results.png")
        plt.show()

if __name__ == "__main__":
    agent = TrafficLightDQN()
    print("=== Début de l'entraînement ===")
    agent.train(episodes=400)