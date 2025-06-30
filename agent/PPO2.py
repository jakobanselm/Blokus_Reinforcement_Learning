import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
import os
import time

# Importiere deine Environment-Klasse.
# Stelle sicher, dass die Datei, die BlokusMultiAgentEnv enthält,
# im selben Verzeichnis liegt oder im Python-Pfad ist.
# Ich nenne sie hier mal "blokus_env.py".
from env.blokus_env_multi_agent_ray_rllib import BlokusMultiAgentEnv

def main():
    """
    Hauptfunktion zum Initialisieren von Ray und zum Starten des Trainings.
    """
    # Initialisiere Ray. 'ignore_reinit_error' ist nützlich, wenn man das Skript
    # interaktiv in einem Notebook ausführt.
    ray.init(ignore_reinit_error=True)

    # Registriere deine custom Environment unter einem Namen.
    # Dies ist der empfohlene Weg, um custom Envs in RLlib zu verwenden.
    register_env("blokus_multi_agent", lambda config: BlokusMultiAgentEnv(config))

    # === Multi-Agent Konfiguration ===
    # Erstelle eine Dummy-Instanz der Umgebung, um die Spaces und Agenten-IDs zu erhalten.
    # Das ist ein robuster Weg, um die Konfiguration dynamisch zu erstellen.
    dummy_env = BlokusMultiAgentEnv()
    
    # Definiere die Policies. In diesem Fall verwenden wir eine einzige, geteilte Policy.
    # Alle Agenten werden dieselben Gewichte lernen und verwenden.
    # Die Policy-Spezifikation (PolicySpec) ist der neue Weg, dies zu definieren.
    policies = {
        "shared_policy": PolicySpec(
            observation_space=dummy_env.observation_spaces["player_0"],
            action_space=dummy_env.action_spaces["player_0"],
            # Keine spezielle Konfiguration für die Policy selbst notwendig
            config={} 
        )
    }

    # Definiere eine Mapping-Funktion, die jedem Agenten die 'shared_policy' zuweist.
    # Diese Funktion wird bei jedem Schritt aufgerufen.
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        return "shared_policy"

    # === PPO AlgorithmConfig ===
    # Dies ist die neue, objektorientierte Art, Algorithmen zu konfigurieren.
    config = (
        PPOConfig()
        .environment(
            "blokus_multi_agent",  # Der registrierte Name deiner Umgebung
            env_config={},         # Keine spezielle Konfiguration für den Env-Konstruktor nötig
        )
        .framework("torch")  # Oder "tf2", "torch" ist der Standard
        .resources(
            num_gpus=0  # Setze auf 1, wenn du eine GPU verwenden möchtest
        )
        .rollouts(
            num_rollout_workers=4,  # Anzahl der parallelen Worker zum Sammeln von Erfahrungen
            rollout_fragment_length='auto'
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
        )
        .training(
            # WICHTIG: RLlib erkennt die "action_mask" im Info-Dict automatisch
            # und wendet sie an. Es ist keine spezielle Modellkonfiguration
            # wie "use_masking" mehr nötig.
            
            # Hyperparameter für PPO
            gamma=0.99,
            lr=5e-5,
            train_batch_size=4096,  # Größe der Daten, die pro Trainingsiteration verwendet werden
            sgd_minibatch_size=256, # Größe der Minibatches für das stochastische Gradientenverfahren
            num_sgd_iter=10,        # Anzahl der Epochen über die gesammelten Daten
            vf_loss_coeff=0.5,
            entropy_coeff=0.01,
            model={
                # Standard-Feedforward-Netzwerk. Gut für den Anfang.
                "fcnet_hiddens": [256, 256], 
            }
        )
    )

    # === Training starten ===
    # Baue den Algorithmus basierend auf der Konfiguration.
    algo = config.build()

    # Definiere, wo Checkpoints gespeichert werden sollen.
    checkpoint_dir = os.path.join(os.path.expanduser("~"), "ray_results", "blokus_ppo")
    print(f"Checkpoints werden in '{checkpoint_dir}' gespeichert.")

    # Trainingsschleife
    num_iterations = 200
    for i in range(num_iterations):
        start_time = time.time()
        result = algo.train()
        end_time = time.time()

        # Gib nützliche Informationen zum Trainingsfortschritt aus.
        print(f"Iteration: {i+1}/{num_iterations}")
        print(f"  Dauer: {end_time - start_time:.2f}s")
        print(f"  Timesteps insgesamt: {result['timesteps_total']}")
        
        # 'episode_reward_mean' ist eine der wichtigsten Metriken.
        if 'episode_reward_mean' in result['hist_stats']:
            # Für Multi-Agent müssen wir in die historien-Statistiken schauen
            mean_reward = result['hist_stats']['episode_reward_mean'][0]
            print(f"  Mittlere Episoden-Belohnung (shared_policy): {mean_reward:.2f}")
        else:
            # Manchmal ist es direkt im Result-Dict
            mean_reward = result.get('episode_reward_mean', float('nan'))
            print(f"  Mittlere Episoden-Belohnung: {mean_reward:.2f}")

        # Speichere alle 10 Iterationen einen Checkpoint.
        if (i + 1) % 10 == 0:
            checkpoint_path = algo.save(checkpoint_dir)
            print(f"  Checkpoint in Iteration {i+1} gespeichert unter: {checkpoint_path.checkpoint.path}")
        
        print("-" * 30)


    # Gib den finalen Checkpoint-Pfad aus.
    final_checkpoint = algo.save(checkpoint_dir)
    print(f"\nTraining abgeschlossen. Finaler Checkpoint gespeichert unter: {final_checkpoint.checkpoint.path}")

    # Ressourcen freigeben
    algo.stop()
    ray.shutdown()

if __name__ == "__main__":
    main()
