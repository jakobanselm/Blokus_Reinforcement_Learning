# Agenten-Dokumentation

Dieses Dokument beschreibt die vorhandenen Trainingsskripte und deren Verwendung.

## Zwei-Agenten-Selbstspiel

`agent/multiagent_selfplay.py` trainiert zwei unabhängige PPO-Agenten, die abwechselnd gegeneinander antreten. Jeder Agent aktualisiert seine Gewichte, während der andere als Gegner fungiert. Starten lässt sich das Training mit:

```bash
python -m agent.multiagent_selfplay
```

## Vier-Agenten-Selbstspiel (RLlib)

Mit `agent/train_rllib_selfplay.py` können vier Agenten parallel mittels RLlib trainiert werden. Das Skript speichert die erlernten Politiken nach dem Training im Ordner `rllib_blokus_agents`.

```bash
python -m agent.train_rllib_selfplay
```

## Tests und Beispielumgebung

Zum schnellen Testen der Umgebung steht `agent/test.py` zur Verfügung. Darin wird ein zufälliger Agent über 100 Schritte ausgeführt und das Spielfeld nach jedem Zug ausgegeben:

```bash
python -m agent.test
```

Für zusätzliche Sicherungen der Umgebung existieren weitere PyTest-Tests im Verzeichnis `agent/`.
