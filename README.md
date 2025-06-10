# Blokus Reinforcement Learning

Dieses Projekt untersucht Reinforcement-Learning-Ansätze für das Brettspiel **Blokus**. Die Umgebung ist in `gymnasium` implementiert und ermöglicht maskierte Aktionen.

## Installation

Das Projekt benötigt Python 3.11 oder neuer. Die Abhängigkeiten befinden sich in `requirements.txt` und lassen sich beispielsweise mit `pip` installieren:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Tests

Zur Verifikation der Umgebung und der Agenten stehen einige PyTest-Tests bereit. Sie können diese mit folgendem Befehl ausführen:

```bash
pytest
```

## Beispielskripte

- `agent/multiagent_selfplay.py` trainiert zwei PPO-Agenten im Selbstspiel gegeneinander.
- `agent/train_rllib_selfplay.py` verwendet RLlib, um vier Agenten parallel im Selbstspiel zu trainieren.
- `agent/train_multiagent_rllib_v3.py` nutzt die verbesserte Multi-Agent-Umgebung und trainiert separate Politiken per PPO.

Die Skripte lassen sich direkt als Python-Module ausführen, sobald alle Abhängigkeiten installiert sind, z.B.:

```bash
python -m agent.multiagent_selfplay
```

Weitere Details zur Verwendung der Agenten finden sich in [Agent.md](Agent.md).
