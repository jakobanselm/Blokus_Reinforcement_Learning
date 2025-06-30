# Blokus Reinforcement Learning

This project implements a reinforcement learning solution for the Blokus game using the Proximal Policy Optimization (PPO) algorithm from the RLlib library. The implementation supports multi-agent training with action masking.

## Project Structure

- `agent/PPO2.py`: Contains the implementation of the PPO algorithm using the RLlib library for multi-agent training. It registers the Blokus multi-agent environment, initializes Ray, configures the PPO algorithm with action masking, and includes a training loop that saves checkpoints.
  
- `env/blokus_env_multi_agent_ray_rllib.py`: Defines the Blokus multi-agent environment compatible with Ray's RLlib. It implements the necessary methods for observation and action spaces, as well as the logic for the multi-agent setup.

- `requirements.txt`: Lists the dependencies required for the project, including Ray and any other necessary libraries for reinforcement learning and the Blokus environment.

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Blokus_Reinforcement_Learning
   ```

2. **Install dependencies**:
   Make sure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the training**:
   Execute the PPO training script:
   ```bash
   python agent/PPO2.py
   ```

## Usage

The project is designed to train agents to play the Blokus game. The training process will output the mean reward of the agents over iterations and save checkpoints every 10 iterations.

## Additional Information

This implementation leverages the capabilities of Ray and RLlib for efficient multi-agent training. The Blokus game rules are encapsulated in the custom environment defined in `blokus_env_multi_agent_ray_rllib.py`. 

For further details on the Blokus game and RLlib, please refer to their respective documentation.