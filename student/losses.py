"""Student one-step plus rollout loss."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from .rollout import open_loop_rollout


# def one_step_delta_loss(model, states: torch.Tensor, actions: torch.Tensor, normalizer) -> torch.Tensor:
#     obs = states[:, :-1].reshape(-1, states.shape[-1])
#     act = actions.reshape(-1, actions.shape[-1])
#     target_delta = (states[:, 1:] - states[:, :-1]).reshape(-1, states.shape[-1])
#     obs_norm = normalizer.normalize_obs(obs)
#     act_norm = normalizer.normalize_act(act)
#     target_norm = normalizer.normalize_delta(target_delta)
#     pred_norm, _ = model(obs_norm, act_norm, None)
#     return F.mse_loss(pred_norm, target_norm)

def one_step_delta_loss(model, states: torch.Tensor, actions: torch.Tensor,
                        normalizer, clean_states: torch.Tensor = None) -> torch.Tensor:
    if clean_states is None:
        clean_states = states

    obs = states[:, :-1].reshape(-1, states.shape[-1])
    act = actions.reshape(-1, actions.shape[-1])

    # targets computed from clean states
    target_delta = (clean_states[:, 1:] - clean_states[:, :-1]).reshape(-1, clean_states.shape[-1])

    obs_norm    = normalizer.normalize_obs(obs)
    act_norm    = normalizer.normalize_act(act)
    target_norm = normalizer.normalize_delta(target_delta)
    pred_norm, _ = model(obs_norm, act_norm, None)
    return F.mse_loss(pred_norm, target_norm)

def rollout_loss(model, states: torch.Tensor, actions: torch.Tensor, normalizer, warmup_steps: int, horizon: int) -> torch.Tensor:
    needed_states = int(warmup_steps) + int(horizon) + 1
    if states.shape[1] < needed_states:
        raise ValueError(
            "training.train_sequence_length is too short for rollout loss: "
            f"need at least {needed_states - 1} actions for warmup={warmup_steps}, horizon={horizon}."
        )
    max_start = states.shape[1] - needed_states
    if max_start > 0:
        start = int(torch.randint(0, max_start + 1, (), device=states.device).item())
    else:
        start = 0
    sub_states  = states[:, start : start + needed_states]
    sub_actions = actions[:, start : start + int(warmup_steps) + int(horizon)]
    preds = open_loop_rollout(model, sub_states, sub_actions, normalizer,
                               warmup_steps=warmup_steps, horizon=horizon)
    targets     = sub_states[:, warmup_steps + 1 : warmup_steps + 1 + horizon]
    pred_norm   = normalizer.normalize_obs(preds)
    target_norm = normalizer.normalize_obs(targets)
    return F.mse_loss(pred_norm, target_norm)


def compute_stages(min_horizon, max_horizon, num_stages, total_updates):
    """
    Compute staged curriculum boundaries automatically.

    Returns list of (start_update, horizon) pairs.
    """
    updates_per_stage = total_updates / num_stages
    horizon_step = (max_horizon - min_horizon) / (num_stages - 1)

    stages = []
    for i in range(num_stages):
        start_update = int(i * updates_per_stage)
        horizon = int(min_horizon + i * horizon_step)
        stages.append((start_update, horizon))

    return stages


def get_staged_horizon(update, stages):
    """
    Return the current horizon given update count and stage boundaries.
    """
    horizon = stages[0][1]
    for start_update, h in stages:
        if update >= start_update:
            horizon = h
    return horizon


# def compute_loss(model, batch: dict[str, torch.Tensor], normalizer, cfg: dict,
#                  update: int = 0, total_updates: int = 1):
#     loss_cfg     = cfg["loss"]
#     states       = batch["states"]
#     actions      = batch["actions"]

#     one = one_step_delta_loss(model, states, actions, normalizer)

#     warmup       = int(cfg["eval"].get("warmup_steps", 10))
#     min_horizon  = int(loss_cfg.get("rollout_min_horizon", 5))
#     max_horizon  = int(loss_cfg.get("rollout_train_horizon", 60))
#     num_stages   = int(loss_cfg.get("rollout_num_stages", 6))

#     stages  = compute_stages(min_horizon, max_horizon, num_stages, total_updates)
#     horizon = get_staged_horizon(update, stages)

#     roll = rollout_loss(model, states, actions, normalizer,
#                         warmup_steps=warmup, horizon=horizon)

#     total = float(loss_cfg.get("one_step_weight", 1.0)) * one \
#           + float(loss_cfg.get("rollout_weight", 0.7)) * roll

#     return total, {
#         "loss/total":       float(total.detach().cpu()),
#         "loss/one_step":    float(one.detach().cpu()),
#         "loss/rollout":     float(roll.detach().cpu()),
#         "loss/horizon":     float(horizon),
#         "loss/max_horizon": float(max_horizon),
#     }

def compute_loss(model, batch: dict[str, torch.Tensor], normalizer, cfg: dict,
                 update: int = 0, total_updates: int = 1):
    loss_cfg = cfg["loss"]
    states  = batch["states"]
    actions = batch["actions"]

    # noise augmentation — inputs are noisy, targets stay clean
    obs_noise_sigma = float(loss_cfg.get("obs_noise_sigma", 0.0))
    act_noise_sigma = float(loss_cfg.get("act_noise_sigma", 0.0))

    if obs_noise_sigma > 0.0 or act_noise_sigma > 0.0:
        noisy_states  = states  + torch.randn_like(states)  * obs_noise_sigma
        noisy_actions = actions + torch.randn_like(actions) * act_noise_sigma
        noisy_actions = noisy_actions.clamp(-3.0, 3.0)
    else:
        noisy_states  = states
        noisy_actions = actions

    one = one_step_delta_loss(model, noisy_states, noisy_actions, normalizer, states)

    warmup       = int(cfg["eval"].get("warmup_steps", 10))
    min_horizon  = int(loss_cfg.get("rollout_min_horizon", 5))
    max_horizon  = int(loss_cfg.get("rollout_train_horizon", 70))
    num_stages   = int(loss_cfg.get("rollout_num_stages", 23))

    stages  = compute_stages(min_horizon, max_horizon, num_stages, total_updates)
    horizon = get_staged_horizon(update, stages)

    roll = rollout_loss(model, noisy_states, noisy_actions, normalizer,
                        warmup_steps=warmup, horizon=horizon)

    total = float(loss_cfg.get("one_step_weight", 1.0)) * one \
          + float(loss_cfg.get("rollout_weight", 0.7)) * roll

    return total, {
        "loss/total":       float(total.detach().cpu()),
        "loss/one_step":    float(one.detach().cpu()),
        "loss/rollout":     float(roll.detach().cpu()),
        "loss/horizon":     float(horizon),
        "loss/max_horizon": float(max_horizon),
    }