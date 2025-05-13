import traci
import numpy as np
import random
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
import os

class TrafficLightQLearning:
    def __init__(self):
        self.actions = [0, 1]  # 0 = étendre, 1 = changer
        self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))

        # Hyperparamètres
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        # Directions (VALIDÉES)
        # self.directions = {
        #     "north": ["-gneE8_0", "-gneE8_1", "-gneE8_2"],
        #     "south": ["gneE12_0", "gneE12_1", "gneE12_2"],
        #     "east": ["gneE10_0", "gneE10_1"],
        #     "west": ["-gneE7_0", "-gneE7_1"]
        # }

        self.directions = {
            "north": ["-gneE8_0", "-gneE8_1", "-gneE8_2"],
            "south": ["gneE12_0", "gneE12_1", "gneE12_2"],
            "east": ["gneE7_0", "gneE7_1", "gneE7_2"],
            "west": ["-gneE10_0", "-gneE10_1", "-gneE10_2"]
    }




        self.verify_lanes()
        self.queue_bins = [0, 3, 6, 9, 12]

        self.episode_rewards = []
        self.avg_wait_times = []
        self.cumulative_wait_times = []

    def verify_lanes(self):
        traci.start(["sumo", "-c", "single.sumocfg", "--quit-on-end"])
        existing_lanes = set(traci.lane.getIDList())
        all_found = True
        for dir_name, lanes in self.directions.items():
            for lane in lanes:
                if lane not in existing_lanes:
                    print(f" Voie introuvable: {lane} ({dir_name})")
                    all_found = False
        if all_found:
            print(" Toutes les voies définies existent dans le réseau.")
        traci.close()
    def get_state(self):
        try:
            phase = traci.trafficlight.getPhase("gneJ2")
            queues = []
            for lanes in self.directions.values():
                queue = 0
                for lane in lanes:
                    try:
                        queue += traci.lane.getLastStepHaltingNumber(lane)
                    except:
                        continue
                queues.append(queue)
            discretized = [np.digitize(q, self.queue_bins) for q in queues]
            return (phase,) + tuple(discretized)
        except Exception as e:
            print(f"Erreur get_state(): {str(e)}")
            return (0, 0, 0, 0, 0)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        return np.argmax(self.q_table[state])

    def get_reward(self):
        try:
            total_wait = 0
            max_queue = 0
            for lanes in self.directions.values():
                for lane in lanes:
                    try:
                        total_wait += traci.lane.getWaitingTime(lane)
                        max_queue = max(max_queue, traci.lane.getLastStepHaltingNumber(lane))
                    except:
                        continue
            return -(total_wait + max_queue * 2)
        except:
            return 0

    def update_traffic_light(self, action, current_phase):
        """Correction de la logique de changement des feux"""
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
                print(f"[] Phase inconnue : {current_phase}, retour à 0")
                new_phase = 0
                duration = 1
            traci.trafficlight.setPhase("gneJ2", new_phase)
            traci.trafficlight.setPhaseDuration("gneJ2", duration)
        except Exception as e:
            print(f"[ update_traffic_light] Erreur avec traci : {e}")
            new_phase = 0
            duration = 1
        return new_phase, duration

    def train(self, episodes=100):
        for episode in range(episodes):
            traci.start([
                "sumo-gui"  ,
                "-c", "single.sumocfg",
                "--quit-on-end",
                "--waiting-time-memory", "1000",
                "--no-step-log",
             
                "--step-length", "1"
            ])

            traci.trafficlight.setPhase("gneJ2", 0)
            current_phase = 0
            total_reward = 0
            wait_times = []

            while traci.simulation.getMinExpectedNumber() > 0:
                state = self.get_state()
                action = self.choose_action(state)

                current_phase, duration = self.update_traffic_light(action, current_phase)

                for _ in range(duration):
                    traci.simulationStep()
                    reward = self.get_reward()
                    total_reward += reward
                    wait_times.append(-reward)

                next_state = self.get_state()
                best_next = np.max(self.q_table[next_state])
                td_target = reward + self.gamma * best_next
                td_error = td_target - self.q_table[state][action]
                self.q_table[state][action] += self.alpha * td_error

            traci.close()

            avg_wait = np.mean(wait_times) if wait_times else 0
            self.episode_rewards.append(total_reward)
            self.avg_wait_times.append(avg_wait)
            self.cumulative_wait_times.append(sum(wait_times))
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            print(f"Episode {episode+1}: "
                  f"REcompense={total_reward:.1f}, "
                  f"Attente={avg_wait:.1f}s, "
                  f"epsilon={self.epsilon:.3f}")

            if (episode + 1) % 50 == 0:
                self.save_model()

        self.save_model()
        self.plot_results()

    def save_model(self, model_type="qlearning"):
        model_dir = f"models/{model_type}"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        data = {
            'q_table': dict(self.q_table),  # For Q-learning
            'directions': self.directions,
            'stats': {
                'rewards': self.episode_rewards,
                'waits': self.avg_wait_times
            }
        }

        file_path = f"{model_dir}/traffic_model.pkl"
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
        print(f"{model_type.capitalize()} model saved at {file_path}")


    def plot_results(self):
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.plot(self.episode_rewards)
        plt.title("Récompenses par épisode")
        plt.subplot(1, 3, 2)
        plt.plot(self.avg_wait_times)
        plt.title("Attente moyenne")
        plt.subplot(1, 3, 3)
        plt.plot(self.cumulative_wait_times)
        plt.title("Attente cumulée")
        plt.tight_layout()
        plt.savefig("results.png")
        plt.show()


if __name__ == "__main__":
    print("=== Vérification du réseau ===")
    agent = TrafficLightQLearning()
    print("\n=== Début de l'entraînement ===")
    agent.train(episodes=400)
