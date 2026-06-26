import torch
from typing import Literal, Callable

def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    ) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Args:
    • reward_fn: Callable[[str, str], dict[str, float]] Scores the rollout responses against the
    ground truths, producing a dict with keys "reward", "format_reward", and "answer_reward".
    • rollout_responses: list[str] Rollouts from the policy. The length of this list is
    rollout_batch_size = n_prompts_per_rollout_batch * group_size.
    • repeated_ground_truths: list[str] The ground truths for the examples. The length of this
    list is rollout_batch_size, because the ground truth for each example is repeated group_size
    times.
    Returns:
    • tuple[torch.Tensor, dict[str, float]].
    ‣ raw_rewards shape (rollout_batch_size,). Unnormalized rewards for each rollout response.
    ‣ metadata Reward statistics to log. At minimum, include the mean total and format rewards
    over the rollout batch
    """
    rewards = []
    correct_series, correct_format_only_series, wrong_series = 0, 0, 0
    for i, response in enumerate(rollout_responses):
        grading = reward_fn(response, repeated_ground_truths[i])
        rewards.append(grading['answer_reward'])
        is_correct = grading['format_reward'] == 1 and grading['answer_reward'] == 1
        correct_series += is_correct
        correct_format_only = grading['format_reward'] == 1 and grading['answer_reward'] == 0
        correct_format_only_series += correct_format_only
        is_wrong = grading['format_reward'] == 0
        wrong_series += is_wrong
    metadata = {
        'mean_rewards': correct_series / len(rollout_responses),
        'mean_correct_format_only':  correct_format_only_series / len(rollout_responses),
        'mean_wrong':  wrong_series / len(rollout_responses),
        'sequence_length': len(rollout_responses)
        }
    rewards = torch.tensor(rewards)
    print(rewards, metadata)
    return torch.tensor(rewards), metadata

"""
def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean" , "none"] = advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std" , "mean" , "none" , "mean"] = "std"):
    Compute advantages by applying the requested baseline and normalization within each group.
    Args:
    • raw_rewards: torch.Tensor shape (rollout_batch_size,). Unnormalized rewards for each
    rollout response, where rollout_batch_size = n_prompts_per_rollout_batch * group_size.
    • group_size: int Number of responses per question (group).
    • baseline: Literal["mean", "none"] For this problem, support mean, which subtracts the per-
    group mean reward. Later, none will mean no baseline subtraction.
    • advantage_eps: float Small constant to avoid division by zero in normalization.
    • advantage_normalizer: Literal["std", "none", "mean"] For this problem, support std, which
    divides by the per-group standard deviation. Later, none will mean no normalization and mean
    will mean divide by the per-group mean reward.
    Returns:
    • tuple[torch.Tensor, dict[str, float]].
    ‣ advantages shape (rollout_batch_size,). Group-normalized rewards for each rollout
    response.
    ‣ metadata your choice of other statistics to log (e.g.
    mean, std, max/min of rewards).
    To test your code, implement [adapters.
   """

