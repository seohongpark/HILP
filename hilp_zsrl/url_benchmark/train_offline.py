import platform
import os

if 'mac' in platform.platform():
    pass
else:
    os.environ['MUJOCO_GL'] = 'egl'
    if 'SLURM_STEP_GPUS' in os.environ:
        os.environ['EGL_DEVICE_ID'] = os.environ['SLURM_STEP_GPUS']

from pathlib import Path
import sys
base = Path(__file__).absolute().parents[1]
for fp in [base, base / "url_benchmark"]:
    assert fp.exists()
    if str(fp) not in sys.path:
        sys.path.append(str(fp))

import logging
import torch
import warnings

logger = logging.getLogger(__name__)
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore', category=DeprecationWarning)

import json
import dataclasses
import tempfile
import typing as tp
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
import numpy as np
import torch
import wandb
import omegaconf as omgcf

from url_benchmark import dmc
from dm_env import specs
from url_benchmark import utils
from url_benchmark import agent as agents
from url_benchmark.logger import Logger
from url_benchmark.in_memory_replay_buffer import ReplayBuffer
from url_benchmark.video import VideoRecorder
from url_benchmark.my_utils import record_video


@dataclasses.dataclass
class Config:
    agent: tp.Any
    # misc
    run_group: str = "Debug"
    seed: int = 1
    device: str = "cuda"
    save_video: bool = True
    use_tb: bool = False
    use_wandb: bool = True
    # experiment
    experiment: str = "offline"
    # task settings
    task: str = "walker_run"
    obs_type: str = "states"  # [states, pixels]
    frame_stack: int = 3  # only works if obs_type=pixels
    image_wh: int = 64
    action_repeat: int = 1
    discount: float = 0.98
    future: float = 0.99  # discount of future sampling, future=1 means no future sampling
    p_currgoal: float = 0  # current goal ratio
    p_randomgoal: float = 0  # random goal ratio
    # eval
    num_eval_episodes: int = 10
    eval_every_steps: int = 10000
    num_final_eval_episodes: int = 50
    video_every_steps: int = 100000
    num_skip_frames: int = 2
    custom_reward: tp.Optional[str] = None  # activates custom eval if not None
    # checkpoint
    snapshot_at: tp.Tuple[int, ...] = ()
    checkpoint_every: int = 100000
    load_model: tp.Optional[str] = None
    # training
    num_grad_steps: int = 1000000
    log_every_steps: int = 1000
    num_seed_frames: int = 0
    replay_buffer_episodes: int = 5000
    update_encoder: bool = True
    batch_size: int = omgcf.II("agent.batch_size")
    goal_eval: bool = False
    # dataset
    load_replay_buffer: tp.Optional[str] = None
    expl_agent: str = "rnd"
    replay_buffer_dir: str = omgcf.SI("../../../../datasets")


ConfigStore.instance().store(name="workspace_config", node=Config)


class BaseReward:
    def __init__(self, seed: tp.Optional[int] = None) -> None:
        self._env: dmc.EnvWrapper  # to be instantiated in subclasses
        self._rng = np.random.RandomState(seed)

    def get_goal(self, goal_space: str) -> np.ndarray:
        raise NotImplementedError

    def from_physics(self, physics: np.ndarray) -> float:
        "careful this is not threadsafe"
        with self._env.physics.reset_context():
            self._env.physics.set_state(physics)
        return self.from_env(self._env)

    def from_env(self, env: dmc.EnvWrapper) -> float:
        raise NotImplementedError


class DmcReward(BaseReward):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        env_name, task_name = name.split("_", maxsplit=1)
        from dm_control import suite  # import
        from url_benchmark import custom_dmc_tasks as cdmc
        if 'jaco' not in env_name:
            make = suite.load if (env_name, task_name) in suite.ALL_TASKS else cdmc.make
            self._env = make(env_name, task_name)
        else:
            self._env = cdmc.make_jaco(task_name, obs_type='states', seed=0)

    def from_env(self, env: dmc.EnvWrapper) -> float:
        return float(self._env.task.get_reward(env.physics))


def make_agent(
        obs_type: str, image_wh, obs_spec, action_spec, num_expl_steps: int, cfg: omgcf.DictConfig
) -> tp.Union[agents.FBDDPGAgent, agents.DDPGAgent]:
    cfg.obs_type = obs_type
    cfg.image_wh = image_wh
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = (action_spec.num_values, ) if isinstance(action_spec, specs.DiscreteArray) \
        else action_spec.shape
    cfg.num_expl_steps = num_expl_steps
    return hydra.utils.instantiate(cfg)


def _init_eval_meta(workspace, custom_reward: BaseReward = None):
    num_steps = workspace.agent.cfg.num_inference_steps
    obs_list, reward_list, next_obs_list = [], [], []
    batch_size = 0
    while batch_size < num_steps:
        batch = workspace.replay_loader.sample(workspace.cfg.batch_size, custom_reward=custom_reward)
        batch = batch.to(workspace.cfg.device)
        if isinstance(workspace.agent, agents.FBDDPGAgent) or (isinstance(workspace.agent, agents.SFAgent) and workspace.agent.cfg.feature_type == 'state'):
            obs_list.append(batch.next_obs)
            next_obs_list.append(batch.next_obs)
        else:
            obs_list.append(batch.obs)
            next_obs_list.append(batch.next_obs)
        reward_list.append(batch.reward)
        batch_size += batch.next_obs.size(0)
    obs, reward, next_obs = torch.cat(obs_list, 0), torch.cat(reward_list, 0), torch.cat(next_obs_list, 0)
    obs_t, reward_t, next_obs_t = obs[:num_steps], reward[:num_steps], next_obs[:num_steps]
    return workspace.agent.infer_meta_from_obs_and_rewards(obs_t, reward_t, next_obs_t)


class Workspace:
    def __init__(self, cfg: Config) -> None:
        self.work_dir = Path.cwd()
        print(f'Workspace: {self.work_dir}')
        print(f'Running code in : {Path(__file__).parent.resolve().absolute()}')
        logger.info(f'Workspace: {self.work_dir}')
        logger.info(f'Running code in : {Path(__file__).parent.resolve().absolute()}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        if not torch.cuda.is_available():
            if cfg.device != "cpu":
                logger.warning(f"Falling back to cpu as {cfg.device} is not available")
                cfg.device = "cpu"
                cfg.agent.device = "cpu"
        self.device = torch.device(cfg.device)

        task = cfg.task
        self.domain = task.split('_', maxsplit=1)[0]

        self.train_env = self._make_env()
        self.eval_env = self._make_env()
        # create agent
        self.train_env.reset()
        self.agent = make_agent(cfg.obs_type,
                                cfg.image_wh,
                                self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                cfg.num_seed_frames // cfg.action_repeat,
                                cfg.agent)

        # create logger
        self.logger = Logger(self.work_dir,
                             use_tb=cfg.use_tb,
                             use_wandb=cfg.use_wandb)

        if cfg.use_wandb:
            exp_name = ''
            exp_name += f'sd{cfg.seed:03d}_'
            if 'SLURM_JOB_ID' in os.environ:
                exp_name += f's_{os.environ["SLURM_JOB_ID"]}.'
            if 'SLURM_PROCID' in os.environ:
                exp_name += f'{os.environ["SLURM_PROCID"]}.'
            exp_name += '_'.join([
                cfg.run_group, cfg.agent.name, self.domain,
            ])
            wandb_output_dir = tempfile.mkdtemp()
            wandb.init(project='hilp_zsrl', group=cfg.run_group, name=exp_name,
                       config=omgcf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
                       dir=wandb_output_dir)

        self.replay_loader = ReplayBuffer(max_episodes=cfg.replay_buffer_episodes, discount=cfg.discount, future=cfg.future)

        cam_id = 0 if 'quadruped' not in self.domain else 2

        self.video_recorder = VideoRecorder(self.work_dir if cfg.save_video else None, task=self.cfg.task,
                                            camera_id=cam_id, use_wandb=self.cfg.use_wandb)

        self.timer = utils.Timer()
        self.global_step = 0
        self.global_episode = 0
        self.eval_rewards_history: tp.List[float] = []
        self._checkpoint_filepath = self.work_dir / "models" / "latest.pt"
        if self._checkpoint_filepath.exists():
            self.load_checkpoint(self._checkpoint_filepath)
        elif cfg.load_model is not None:
            self.load_checkpoint(cfg.load_model, exclude=["replay_loader"])

        datasets_dir = self.work_dir / cfg.replay_buffer_dir
        replay_dir = datasets_dir.resolve() / self.domain / cfg.expl_agent / 'buffer'
        print(f'replay dir: {replay_dir}')

        print("loading Replay from %s", self.cfg.load_replay_buffer)
        self.load_checkpoint(self.cfg.load_replay_buffer, only=["replay_loader"], num_episodes=cfg.replay_buffer_episodes, use_pixels=(cfg.obs_type == 'pixels'))

        self.replay_loader._future = cfg.future
        self.replay_loader._discount = cfg.discount
        self.replay_loader._p_currgoal = cfg.p_currgoal
        self.replay_loader._p_randomgoal = cfg.p_randomgoal
        self.replay_loader._frame_stack = cfg.frame_stack if cfg.obs_type == 'pixels' else None
        self.replay_loader._max_episodes = len(self.replay_loader._storage["discount"])

    def _make_env(self) -> dmc.EnvWrapper:
        cfg = self.cfg
        return dmc.make(cfg.task, cfg.obs_type, cfg.frame_stack, cfg.action_repeat, cfg.seed, image_wh=cfg.image_wh)

    @property
    def global_frame(self) -> int:
        return self.global_step * self.cfg.action_repeat

    def _make_custom_reward(self) -> tp.Optional[BaseReward]:
        if self.cfg.custom_reward is None:
            return None
        return DmcReward(self.cfg.custom_reward)

    def get_argmax_goal(self, custom_reward):
        num_steps = self.agent.cfg.num_inference_steps
        reward_list, next_obs_list = [], []
        batch_size = 0
        while batch_size < num_steps:
            batch = self.replay_loader.sample(self.cfg.batch_size, custom_reward=custom_reward)
            batch = batch.to(self.cfg.device)
            next_obs_list.append(batch.next_obs)
            reward_list.append(batch.reward)
            batch_size += batch.next_obs.size(0)
        reward, next_obs = torch.cat(reward_list, 0), torch.cat(next_obs_list, 0)
        reward_t, next_obs_t = reward[:num_steps], next_obs[:num_steps]
        return next_obs_t[torch.argmax(reward_t)].detach().cpu().numpy()

    def train(self):
        train_until_step = utils.Until(self.cfg.num_grad_steps)
        eval_every_step = utils.Every(self.cfg.eval_every_steps)
        log_every_step = utils.Every(self.cfg.log_every_steps)

        while train_until_step(self.global_step):
            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(), self.global_step)
                self.eval()

            metrics = self.agent.update(self.replay_loader, self.global_step)
            self.logger.log_metrics(metrics, self.global_step, ty='train')
            if log_every_step(self.global_step):
                elapsed_time, total_time = self.timer.reset()
                with self.logger.log_and_dump_ctx(self.global_step, ty='train') as log:
                    log('fps', self.cfg.log_every_steps / elapsed_time)
                    log('total_time', total_time)
                    log('step', self.global_step)
            self.global_step += 1
            # try to save snapshot
            if self.global_frame in self.cfg.snapshot_at:
                self.save_checkpoint(self._checkpoint_filepath.with_name(f'snapshot_{self.global_frame}.pt'),
                                     exclude=["replay_loader"])
            # save checkpoint to reload
            if self.cfg.checkpoint_every != 0 and self.global_frame % self.cfg.checkpoint_every == 0:
                self.save_checkpoint(self._checkpoint_filepath, exclude=["replay_loader"])
        if self.cfg.checkpoint_every != 0:
            self.save_checkpoint(self._checkpoint_filepath, exclude=["replay_loader"])  # make sure we save the final checkpoint
        self.finalize()

    def eval(self, final_eval=False):
        step, episode = 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        physics_agg = dmc.PhysicsAggregator()
        rewards: tp.List[float] = []
        custom_reward = self._make_custom_reward()  # not None only if final_eval
        meta = _init_eval_meta(self, custom_reward)
        videos = []
        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            if self.cfg.goal_eval:
                goal = self.get_argmax_goal(custom_reward)
                meta = self.agent.get_goal_meta(goal_array=goal, obs_array=time_step.observation)

            total_reward = 0.0
            video_enabled = (episode < 2) and (self.global_frame % self.cfg.video_every_steps == 0)
            self.video_recorder.init(self.eval_env, enabled=video_enabled)
            while not time_step.last():
                if self.cfg.goal_eval and self.cfg.agent.name == 'sf' and self.cfg.agent.feature_learner == 'hilp':
                    # Recompute z every step
                    meta = self.agent.get_goal_meta(goal_array=goal, obs_array=time_step.observation)
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation, meta, self.global_step, eval_mode=True)
                time_step = self.eval_env.step(action)
                physics_agg.add(self.eval_env)
                if step % self.cfg.num_skip_frames == 0:
                    self.video_recorder.record(self.eval_env)
                if custom_reward is not None:
                    time_step.reward = custom_reward.from_env(self.eval_env)
                total_reward += time_step.reward
                step += 1
            if video_enabled:
                videos.append(self.video_recorder.frames)
            rewards.append(total_reward)
            episode += 1
            self.video_recorder.save(f'{self.global_frame}.mp4')

        self.eval_rewards_history.append(float(np.mean(rewards)))
        if final_eval:
            return {
                'episode_reward': self.eval_rewards_history[-1],
            }, videos

        if len(videos) > 0:
            video = record_video(f'TrajVideo_{self.global_frame}', videos, skip_frames=2)
            wandb.log({'TrajVideo': video}, step=self.global_frame)
        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', self.eval_rewards_history[-1])
            if len(rewards) > 1:
                log('episode_reward#std', float(np.std(rewards)))
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

    _CHECKPOINTED_KEYS = ('agent', 'global_step', 'global_episode', "replay_loader")

    def save_checkpoint(self, fp: tp.Union[Path, str], exclude: tp.Sequence[str] = ()) -> None:
        logger.info(f"Saving checkpoint to {fp}")
        exclude = list(exclude)
        assert all(x in self._CHECKPOINTED_KEYS for x in exclude)
        fp = Path(fp)
        fp.parent.mkdir(exist_ok=True, parents=True)
        assert isinstance(self.replay_loader, ReplayBuffer), "Is this buffer designed for checkpointing?"
        # this is just a dumb security check to not forget about it
        payload = {k: self.__dict__[k] for k in self._CHECKPOINTED_KEYS if k not in exclude}
        with fp.open('wb') as f:
            torch.save(payload, f, pickle_protocol=4)

    def load_checkpoint(self, fp: tp.Union[Path, str], only: tp.Optional[tp.Sequence[str]] = None, exclude: tp.Sequence[str] = (), num_episodes=None, use_pixels=False) -> None:
        print(f"loading checkpoint from {fp}")
        fp = Path(fp)
        with fp.open('rb') as f:
            payload = torch.load(f)

        if num_episodes is not None:
            payload._episodes_length = payload._episodes_length[:num_episodes]
            payload._max_episodes = min(payload._max_episodes, num_episodes)
            for key, value in payload._storage.items():
                payload._storage[key] = value[:num_episodes]
        if use_pixels:
            payload._storage['observation'] = payload._storage['pixel']
            del payload._storage['pixel']
            payload._batch_names.remove('pixel')

        if isinstance(payload, ReplayBuffer):  # compatibility with pure buffers pickles
            payload = {"replay_loader": payload}
        if only is not None:
            only = list(only)
            assert all(x in self._CHECKPOINTED_KEYS for x in only)
            payload = {x: payload[x] for x in only}
        exclude = list(exclude)
        assert all(x in self._CHECKPOINTED_KEYS for x in exclude)
        for x in exclude:
            payload.pop(x, None)
        for name, val in payload.items():
            logger.info("Reloading %s from %s", name, fp)
            if name == "agent":
                self.agent.init_from(val)
            elif name == "replay_loader":
                assert isinstance(val, ReplayBuffer)
                # pylint: disable=protected-access
                # drop unecessary meta which could make a mess
                val._current_episode.clear()  # make sure we can start over
                val._future = self.cfg.future
                val._discount = self.cfg.discount
                val._max_episodes = len(val._storage["discount"])
                self.replay_loader = val
            else:
                assert hasattr(self, name)
                setattr(self, name, val)
                if name == "global_episode":
                    logger.warning(f"Reloaded agent at global episode {self.global_episode}")

    def finalize(self) -> None:
        print("Running final test", flush=True)

        domain_tasks = {
            "cheetah": ['walk', 'walk_backward', 'run', 'run_backward'],
            "quadruped": ['stand', 'walk', 'run', 'jump'],
            "walker": ['stand', 'walk', 'run', 'flip'],
            "jaco": ['reach_top_left', 'reach_top_right', 'reach_bottom_left', 'reach_bottom_right'],
        }
        if self.domain not in domain_tasks:
            return
        eval_hist = self.eval_rewards_history
        rewards = {}
        videos = {}
        infos = {}
        for name in domain_tasks[self.domain]:
            task = "_".join([self.domain, name])
            self.cfg.task = task
            self.cfg.custom_reward = task  # for the replay buffer
            self.cfg.seed += 10000000  # for the sake of avoiding similar seeds
            self.eval_env = self._make_env()
            self.eval_rewards_history = []
            self.cfg.num_eval_episodes = self.cfg.num_final_eval_episodes
            info, video = self.eval(final_eval=True)
            rewards[name] = self.eval_rewards_history
            infos[name] = info
            videos[name] = video
        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            for name in domain_tasks[self.domain]:
                video = record_video(f'Final_{name}', videos[name], skip_frames=2)
                wandb.log({f'Final_{name}': video}, step=self.global_frame)
                for k, v in infos[name].items():
                    log(f'final/{name}/{k}', v)
        self.eval_rewards_history = eval_hist  # restore
        with (self.work_dir / "test_rewards.json").open("w") as f:
            json.dump(rewards, f)


@hydra.main(config_path='.', config_name='base_config')
def main(cfg: omgcf.DictConfig) -> None:
    workspace = Workspace(cfg)
    workspace.train()


if __name__ == '__main__':
    main()
