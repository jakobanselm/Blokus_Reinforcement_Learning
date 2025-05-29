
# Import MaskablePPO for masked action support and ActionMasker wrapper
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

# Import MaskableEvalCallback and evaluation helper for proper masked evaluation
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.evaluation import evaluate_policy

# Import your custom Blokus environment and game logic
from env.blokus_env_masked import Blokus_Env_Masked
from game.game import Game
from global_constants import BOARD_SIZE, PLAYER_COLORS

# Import MaskableCategorical to monkey-patch masking behavior
from sb3_contrib.common.maskable.distributions import MaskableCategorical
import torch

# -----------------------------------------------------------------------------
# Monkey-patch to normalize masked logits into a valid probability simplex
# -----------------------------------------------------------------------------

def _normalized_apply_masking(self, masks):
    """
    Override the default apply_masking to:
      1) Handle None masks by creating a standard Categorical distribution
      2) Convert mask to tensor of correct type/device
      3) Replace invalid-action logits with -inf
      4) Recreate Categorical distribution via logits => softmax normalization
    """
    # 1) If no mask provided, build distribution directly
    if masks is None:
        self.distribution = torch.distributions.Categorical(logits=self.logits)
        return

    # 2) Convert mask to torch Tensor on correct device/dtype
    if not torch.is_tensor(masks):
        masks = torch.tensor(masks, device=self.logits.device)
    masks = masks.to(self.logits.dtype)

    # 3) Create new_logits: keep valid, set invalid to -inf
    neginf = torch.finfo(self.logits.dtype).min
    new_logits = torch.where(masks.bool(), self.logits, neginf)

    # 4) Build new Categorical distribution from new_logits
    self.distribution = torch.distributions.Categorical(logits=new_logits)
    self.logits = new_logits

# Apply monkey-patch to MaskableCategorical
MaskableCategorical.apply_masking = _normalized_apply_masking

# -----------------------------------------------------------------------------
# Environment setup
# -----------------------------------------------------------------------------

game = Game(BOARD_SIZE, PLAYER_COLORS)
env = Blokus_Env_Masked(game=game)
wrapped_env = ActionMasker(env, lambda e: e.get_action_mask())

# For evaluation, create a separate but identical masked env
eval_game = Game(BOARD_SIZE, PLAYER_COLORS)
eval_env = Blokus_Env_Masked(game=eval_game)
eval_env = ActionMasker(eval_env, lambda e: e.get_action_mask())

# -----------------------------------------------------------------------------
# Model configuration, evaluation, and training
# -----------------------------------------------------------------------------

# Set up MaskableEvalCallback for proper masked evaluation
eval_callback = MaskableEvalCallback(
    eval_env,
    best_model_save_path="./best_model",
    log_path="./eval_logs",
    eval_freq=5000,
    n_eval_episodes=10,
    deterministic=True
)

# Create MaskablePPO model using MultiInputPolicy for dict observations
model = MaskablePPO(
    policy="MultiInputPolicy",
    env=wrapped_env,
    verbose=1,
    learning_rate=1.5e-4,    # halbierte Lernrate
    clip_range=0.1,          # engeres Clip-Fenster
    ent_coef=0.01,           # erhöhte Entropie
    tensorboard_log="./tb_blokus/"
)

# Train the model for 100k timesteps with masked evaluation callback
model.learn(total_timesteps=100_000, callback=eval_callback)

# Save the trained model for later usage
model.save("ppo_blokus")

# (Optional) Manual evaluation after training
# mean_reward, std_reward = evaluate_policy(
#     model,
#     eval_env,
#     n_eval_episodes=20,
#     deterministic=True
# )
# print(f"Mean reward: {mean_reward:.2f} \u00B1 {std_reward:.2f}")

