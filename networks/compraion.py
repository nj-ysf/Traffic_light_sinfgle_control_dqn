import matplotlib as plt
import pickle


def load_results(model_type="qlearning"):
    with open(f"models/{model_type}/traffic_model.pkl", "rb") as f:
        data = pickle.load(f)
    return data['stats']['rewards'], data['stats']['waits']

def plot_comparison():
    qlearning_rewards, qlearning_wait_times = load_results("qlearning")
    dqn_rewards, dqn_wait_times = load_results("dqn")

    plt.figure(figsize=(15, 6))

    # Comparaison des récompenses par épisode
    plt.subplot(1, 2, 1)
    plt.plot(qlearning_rewards, label='Q-learning')
    plt.plot(dqn_rewards, label='DQN')
    plt.title("Récompenses par épisode")
    plt.xlabel("Épisodes")
    plt.ylabel("Récompense")
    plt.legend()

    # Comparaison du temps d'attente moyen
    plt.subplot(1, 2, 2)
    plt.plot(qlearning_wait_times, label='Q-learning')
    plt.plot(dqn_wait_times, label='DQN')
    plt.title("Temps d'attente moyen")
    plt.xlabel("Épisodes")
    plt.ylabel("Temps d'attente (s)")
    plt.legend()

    plt.tight_layout()
    plt.show()

# Exécution de la comparaison
plot_comparison()
