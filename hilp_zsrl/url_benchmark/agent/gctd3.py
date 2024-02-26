import typing as tp
from collections import OrderedDict
import dataclasses
import logging

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
import omegaconf

from url_benchmark.dmc import TimeStep
from url_benchmark.in_memory_replay_buffer import ReplayBuffer
from url_benchmark import utils
from .fb_modules import mlp, Actor


logger = logging.getLogger(__name__)
MetaDict = tp.Mapping[str, np.ndarray]


@dataclasses.dataclass
class GCTD3AgentConfig:
    # @package agent
    _target_: str = "url_benchmark.agent.gctd3.GCTD3Agent"
    name: str = "gctd3"
    obs_type: str = omegaconf.MISSING  # to be specified later
    image_wh: int = omegaconf.MISSING  # to be specified later
    obs_shape: tp.Tuple[int, ...] = omegaconf.MISSING  # to be specified later
    action_shape: tp.Tuple[int, ...] = omegaconf.MISSING  # to be specified later
    device: str = omegaconf.II("device")  # ${device}
    lr: float = 1e-4
    critic_target_tau: float = 0.01
    update_every_steps: float = 1
    use_tb: bool = omegaconf.II("use_tb")  # ${use_tb}
    use_wandb: bool = omegaconf.II("use_wandb")  # ${use_wandb}
    num_expl_steps: int = omegaconf.MISSING
    num_inference_steps: int = 10000
    hidden_dim: int = 1024
    feature_dim: int = 512
    stddev_schedule: str = "0.2"  # "linear(1,0.2,200000)"
    stddev_clip: float = 0.3  # 1.0
    nstep: int = 1
    batch_size: int = 1024  # 256 for pixels
    init_critic: bool = True
    future_ratio: float = 0
    preprocess: bool = True
    add_trunk: bool = False
    gc_negative: bool = True
    use_done: bool = True


cs = ConfigStore.instance()
cs.store(group="agent", name="gctd3", node=GCTD3AgentConfig)


class Critic(nn.Module):
    def __init__(self, obs_dim, z_dim, action_dim, feature_dim, hidden_dim,
                 preprocess=False, add_trunk=True) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.z_dim = z_dim
        self.preprocess = preprocess

        if self.preprocess:
            self.obs_action_net = mlp(self.obs_dim + self.action_dim, hidden_dim, "ntanh", feature_dim, "irelu")
            self.obs_z_net = mlp(self.obs_dim + self.z_dim, hidden_dim, "ntanh", feature_dim, "irelu")
            if not add_trunk:
                self.trunk: nn.Module = nn.Identity()
                feature_dim = 2 * feature_dim
            else:
                self.trunk = mlp(2 * feature_dim, hidden_dim, "irelu")
                feature_dim = hidden_dim
        else:
            self.trunk = mlp(self.obs_dim + self.z_dim + self.action_dim, hidden_dim, "ntanh")
            feature_dim = hidden_dim

        seq = [feature_dim, hidden_dim, "irelu", self.z_dim]
        self.F1 = mlp(*seq)
        self.F2 = mlp(*seq)

        self.apply(utils.weight_init)

    def forward(self, obs, z, action):
        assert z.shape[-1] == self.z_dim

        if self.preprocess:
            obs_action = self.obs_action_net(torch.cat([obs, action], dim=-1))
            obs_z = self.obs_z_net(torch.cat([obs, z], dim=-1))
            h = torch.cat([obs_action, obs_z], dim=-1)
        else:
            h = torch.cat([obs, z, action], dim=-1)

        if hasattr(self, "trunk"):
            h = self.trunk(h)
        F1 = self.F1(h)
        F2 = self.F2(h)
        return F1, F2


class GCTD3Agent:
    # pylint: disable=unused-argument

    def __init__(self,
                 **kwargs: tp.Any
                 ):
        cfg = GCTD3AgentConfig(**kwargs)
        self.cfg = cfg
        assert len(cfg.action_shape) == 1
        self.action_dim = cfg.action_shape[0]
        self.solved_meta: tp.Any = None

        self.obs_dim = cfg.obs_shape[0]
        self.goal_dim = self.obs_dim

        self.actor = Actor(self.obs_dim, self.goal_dim, self.action_dim,
                           cfg.feature_dim, cfg.hidden_dim,
                           preprocess=cfg.preprocess, add_trunk=self.cfg.add_trunk).to(cfg.device)

        self.critic: nn.Module = Critic(self.obs_dim, self.goal_dim, self.action_dim,
                                        cfg.feature_dim, cfg.hidden_dim,
                                        preprocess=cfg.preprocess, add_trunk=self.cfg.add_trunk).to(cfg.device)
        self.critic_target: nn.Module = Critic(self.obs_dim, self.goal_dim, self.action_dim,
                                               cfg.feature_dim, cfg.hidden_dim,
                                               preprocess=cfg.preprocess, add_trunk=self.cfg.add_trunk).to(cfg.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)

        self.reward_model: tp.Optional[torch.nn.Module] = None
        self.reward_opt: tp.Optional[torch.optim.Adam] = None

        self.train()
        self.critic_target.train()

    def train(self, training: bool = True) -> None:
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def init_from(self, other) -> None:
        # copy parameters over
        utils.hard_update_params(other.actor, self.actor)
        utils.hard_update_params(other.critic, self.critic)

    def init_meta(self) -> MetaDict:
        return OrderedDict()

    def update_meta(
        self,
        meta: MetaDict,
        global_step: int,
        time_step: TimeStep,
        finetune: bool = False,
        replay_loader: tp.Optional[ReplayBuffer] = None
    ) -> MetaDict:
        return meta

    def get_goal_meta(self, goal_array: np.ndarray) -> MetaDict:
        meta = OrderedDict()
        meta['g'] = goal_array
        return meta

    def infer_meta_from_obs_and_rewards(self, obs: torch.Tensor, reward: torch.Tensor, next_obs: torch.Tensor):
        g = next_obs[torch.argmax(reward)]
        meta = OrderedDict()
        meta['g'] = g.squeeze().cpu().numpy()
        return meta

    def act(self, obs, meta, step, eval_mode) -> np.ndarray:
        device = torch.device(self.cfg.device)
        obs = torch.as_tensor(obs, device=device).unsqueeze(0)
        goals = []
        for value in meta.values():
            value = torch.as_tensor(value, device=device).unsqueeze(0)
            goals.append(value)
        goal = torch.cat(goals, dim=-1)
        stddev = utils.schedule(self.cfg.stddev_schedule, step)
        dist = self.actor(obs, goal, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.cfg.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self,
                      obs: torch.Tensor,
                      goal: torch.Tensor,
                      action: torch.Tensor,
                      reward: torch.Tensor,
                      discount: torch.Tensor,
                      next_obs: torch.Tensor,
                      step: int) -> tp.Dict[str, float]:
        metrics = {}

        with torch.no_grad():
            stddev = utils.schedule(self.cfg.stddev_schedule, step)
            dist = self.actor(next_obs, goal, stddev)
            next_action = dist.sample(clip=self.cfg.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, goal, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, goal, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.cfg.use_tb or self.cfg.use_wandb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()
            metrics['stdev'] = stddev

        # optimize critic
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        return metrics

    def update_actor(self,
                     obs: torch.Tensor,
                     goal: torch.Tensor,
                     step: int) -> tp.Dict[str, float]:
        metrics = {}

        stddev = utils.schedule(self.cfg.stddev_schedule, step)
        dist = self.actor(obs, goal, stddev)
        action = dist.sample(clip=self.cfg.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, goal, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.cfg.use_tb or self.cfg.use_wandb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update(self, replay_loader: ReplayBuffer, step: int) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}

        if step % self.cfg.update_every_steps != 0:
            return metrics

        batch = replay_loader.sample(self.cfg.batch_size)
        batch = batch.to(self.cfg.device)

        obs = batch.obs
        action = batch.action
        discount = batch.discount
        next_obs = batch.next_obs
        goal = batch.future_obs
        done = ((obs - goal).abs().sum(dim=-1, keepdim=True) < 1e-8)
        if self.cfg.gc_negative:
            reward = done.to(torch.float32) - 1.
        else:
            reward = done.to(torch.float32)
        if self.cfg.use_done:
            discount = torch.where(done, torch.zeros_like(discount), discount)

        # update critic
        metrics.update(self.update_critic(obs=obs, goal=goal, action=action, reward=reward,
                                          discount=discount, next_obs=next_obs, step=step))

        # update actor
        metrics.update(self.update_actor(obs=obs, goal=goal, step=step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target, self.cfg.critic_target_tau)

        return metrics
