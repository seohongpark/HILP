import glob
import os
import time
import platform
from datetime import datetime

if 'mac' in platform.platform():
    pass
else:
    os.environ['MUJOCO_GL'] = 'egl'
    if 'SLURM_STEP_GPUS' in os.environ:
        os.environ['EGL_DEVICE_ID'] = os.environ['SLURM_STEP_GPUS']

from absl import app, flags
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import flax

import tqdm

from src import d4rl_utils, d4rl_ant, ant_diagnostics, viz_utils
from src.agents import hilp as learner
from src.dataset_utils import GCDataset

from jaxrl_m.wandb import setup_wandb, default_wandb_config
import wandb
from jaxrl_m.evaluation import evaluate_with_trajectories, EpisodeMonitor

from ml_collections import config_flags
import pickle

from src.utils import record_video, CsvLogger

FLAGS = flags.FLAGS
flags.DEFINE_string('agent_name', 'hilp', '')
flags.DEFINE_string('env_name', 'antmaze-large-diverse-v2', '')

flags.DEFINE_string('save_dir', 'exp/', '')
flags.DEFINE_string('restore_path', None, '')
flags.DEFINE_integer('restore_epoch', None, '')
flags.DEFINE_string('run_group', 'Debug', '')
flags.DEFINE_integer('seed', 0, '')
flags.DEFINE_integer('eval_episodes', 50, '')
flags.DEFINE_integer('num_video_episodes', 2, '')
flags.DEFINE_integer('log_interval', 1000, '')
flags.DEFINE_integer('eval_interval', 100000, '')
flags.DEFINE_integer('save_interval', 1000000, '')
flags.DEFINE_integer('batch_size', 1024, '')
flags.DEFINE_integer('train_steps', 1000000, '')

flags.DEFINE_float('lr', 3e-4, '')
flags.DEFINE_integer('value_hidden_dim', 512, '')
flags.DEFINE_integer('value_num_layers', 3, '')
flags.DEFINE_integer('actor_hidden_dim', 512, '')
flags.DEFINE_integer('actor_num_layers', 3, '')
flags.DEFINE_float('discount', 0.99, '')
flags.DEFINE_float('tau', 0.005, '')
flags.DEFINE_float('expectile', 0.95, '')
flags.DEFINE_integer('use_layer_norm', 1, '')
flags.DEFINE_integer('skill_dim', 32, '')
flags.DEFINE_float('skill_expectile', 0.9, '')
flags.DEFINE_float('skill_temperature', 10, '')
flags.DEFINE_float('skill_discount', 0.99, '')

flags.DEFINE_float('p_currgoal', 0.0, '')
flags.DEFINE_float('p_trajgoal', 0.625, '')
flags.DEFINE_float('p_randomgoal', 0.375, '')

flags.DEFINE_integer('planning_num_recursions', 0, '')
flags.DEFINE_integer('planning_num_states', 50000, '')
flags.DEFINE_integer('planning_num_knns', 50, '')

flags.DEFINE_string('encoder', None, '')
flags.DEFINE_float('p_aug', None, '')

flags.DEFINE_string('algo_name', None, '')  # Not used, only for logging

config_flags.DEFINE_config_dict('wandb', default_wandb_config(), lock_config=False)


@jax.jit
def get_traj_v(agent, trajectory):
    def get_v(s, g):
        v1, v2 = agent.network(jax.tree_map(lambda x: x[None], s), jax.tree_map(lambda x: x[None], g), method='value')
        return (v1 + v2) / 2
    observations = trajectory['observations']
    all_values = jax.vmap(jax.vmap(get_v, in_axes=(None, 0)), in_axes=(0, None))(observations, observations)
    return {
        'dist_to_beginning': all_values[:, 0],
        'dist_to_end': all_values[:, -1],
        'dist_to_middle': all_values[:, all_values.shape[1] // 2],
    }


@jax.jit
def get_v_goal(agent, goal, observations):
    goal = jnp.tile(goal, (observations.shape[0], 1))
    v1, v2 = agent.network(observations, goal, method='value')
    return (v1 + v2) / 2


def get_env_and_dataset():
    aux_env = {}
    goal_info = {}
    if 'antmaze' in FLAGS.env_name:
        env_name = FLAGS.env_name

        if 'ultra' in FLAGS.env_name:
            import d4rl_ext
            import gym
            env = gym.make(env_name)
            env = EpisodeMonitor(env)
        else:
            env = d4rl_utils.make_env(env_name)

        dataset = d4rl_utils.get_dataset(env, FLAGS.env_name, goal_conditioned=True)
        dataset = dataset.copy({'rewards': dataset['rewards'] - 1.0})

        env.render(mode='rgb_array', width=200, height=200)
        if 'large' in FLAGS.env_name:
            env.viewer.cam.lookat[0] = 18
            env.viewer.cam.lookat[1] = 12
            env.viewer.cam.distance = 50
            env.viewer.cam.elevation = -90

            viz_env, viz_dataset = d4rl_ant.get_env_and_dataset(env_name)
            viz = ant_diagnostics.Visualizer(env_name, viz_env, viz_dataset, discount=FLAGS.discount)
            init_state = np.copy(viz_dataset['observations'][0])
            init_state[:2] = (12.5, 8)
            aux_env = {
                'viz_env': viz_env,
                'viz_dataset': viz_dataset,
                'viz': viz,
            }
        elif 'ultra' in FLAGS.env_name:
            env.viewer.cam.lookat[0] = 26
            env.viewer.cam.lookat[1] = 18
            env.viewer.cam.distance = 70
            env.viewer.cam.elevation = -90
        else:
            raise NotImplementedError
    elif 'kitchen' in FLAGS.env_name:
        if 'visual' in FLAGS.env_name:
            orig_env_name = FLAGS.env_name.split('visual-')[1]
            env = d4rl_utils.make_env(orig_env_name)
            dataset = dict(np.load(f'data/d4rl_kitchen_rendered/{orig_env_name}.npz'))
            dataset = d4rl_utils.get_dataset(env, FLAGS.env_name, dataset=dataset, filter_terminals=True)
            if 'partial' in FLAGS.env_name:
                # Precomputed index for a goal state
                goal_info = {
                    'ob': dataset['observations'][118319],
                }
            elif 'mixed' in FLAGS.env_name:
                from src.d4rl_utils import kitchen_render
                state = env.reset()
                # This is dataset['observations'][118319] of kitchen-partial-v0
                goal_state = [-2.3403780e+00, -1.3053924e+00, 1.1021180e+00, -1.8613019e+00, 1.5087037e-01, 1.7687809e+00, 1.2525779e+00, 2.9698312e-02, 3.0899283e-02, 3.9908718e-04, 4.9550228e-05, -1.9946630e-05, 2.7519276e-05, 4.8786267e-05, 3.2835731e-05, 2.6504624e-05, 3.8422750e-05, -6.9888681e-01, -5.0150707e-02, 3.4855098e-01, -9.8701166e-03, -7.6958216e-03, -8.0031347e-01, -1.9142720e-01, 7.2064394e-01, 1.6191028e+00, 1.0021452e+00, -3.2998802e-04, 3.7205056e-05, 5.3616576e-02]
                # Set the goal state for kitchen-mixed-v0
                goal_state[9:] = state[39:]
                env.sim.set_state(np.concatenate([goal_state, env.init_qvel]))
                env.sim.forward()
                goal_info = {
                    'ob': kitchen_render(env).astype(np.float32),
                }
                env.reset()
        else:
            env = d4rl_utils.make_env(FLAGS.env_name)
            dataset = d4rl_utils.get_dataset(env, FLAGS.env_name, filter_terminals=True)
            dataset = dataset.copy({'observations': dataset['observations'][:, :30], 'next_observations': dataset['next_observations'][:, :30]})
    else:
        raise NotImplementedError

    return env, dataset, aux_env, goal_info


def main(_):
    g_start_time = int(datetime.now().timestamp())

    exp_name = ''
    exp_name += f'sd{FLAGS.seed:03d}_'
    if 'SLURM_JOB_ID' in os.environ:
        exp_name += f's_{os.environ["SLURM_JOB_ID"]}.'
    if 'SLURM_PROCID' in os.environ:
        exp_name += f'{os.environ["SLURM_PROCID"]}.'
    if 'SLURM_RESTART_COUNT' in os.environ:
        exp_name += f'rs_{os.environ["SLURM_RESTART_COUNT"]}.'
    exp_name += f'{g_start_time}'
    exp_name += f'_{FLAGS.wandb["name"]}'


    # Create wandb logger
    FLAGS.wandb['project'] = 'hilp_gcrl'
    FLAGS.wandb['name'] = FLAGS.wandb['exp_descriptor'] = exp_name
    FLAGS.wandb['group'] = FLAGS.wandb['exp_prefix'] = FLAGS.run_group
    setup_wandb(dict(), **FLAGS.wandb)

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, wandb.config.exp_prefix, wandb.config.experiment_id)
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    env, dataset, aux_env, goal_info = get_env_and_dataset()

    base_observation = jax.tree_map(lambda arr: arr[0], dataset['observations'])
    env.reset()

    train_dataset = GCDataset(
        dataset,
        p_currgoal=FLAGS.p_currgoal, p_trajgoal=FLAGS.p_trajgoal, p_randomgoal=FLAGS.p_randomgoal,
        discount=FLAGS.discount, p_aug=FLAGS.p_aug,
    )

    total_steps = FLAGS.train_steps
    example_batch = dataset.sample(1)
    agent = learner.create_learner(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        lr=FLAGS.lr,
        value_hidden_dims=(FLAGS.value_hidden_dim,) * FLAGS.value_num_layers,
        actor_hidden_dims=(FLAGS.actor_hidden_dim,) * FLAGS.actor_num_layers,
        discount=FLAGS.discount,
        tau=FLAGS.tau,
        expectile=FLAGS.expectile,
        use_layer_norm=FLAGS.use_layer_norm,
        skill_dim=FLAGS.skill_dim,
        skill_expectile=FLAGS.skill_expectile,
        skill_temperature=FLAGS.skill_temperature,
        skill_discount=FLAGS.skill_discount,
        encoder=FLAGS.encoder,
    )

    if FLAGS.restore_path is not None:
        restore_path = FLAGS.restore_path
        candidates = glob.glob(restore_path)
        if len(candidates) == 0:
            raise Exception(f'Path does not exist: {restore_path}')
        if len(candidates) > 1:
            raise Exception(f'Multiple matching paths exist for: {restore_path}')
        if FLAGS.restore_epoch is None:
            restore_path = candidates[0] + '/params.pkl'
        else:
            restore_path = candidates[0] + f'/params_{FLAGS.restore_epoch}.pkl'
        with open(restore_path, "rb") as f:
            load_dict = pickle.load(f)
        agent = flax.serialization.from_state_dict(agent, load_dict['agent'])
        print(f'Restored from {restore_path}')

    if 'antmaze' in FLAGS.env_name:
        example_trajectory = train_dataset.sample(50, indx=np.arange(1000, 1050), evaluation=True)
    else:
        example_trajectory = train_dataset.sample(50, indx=np.arange(0, 50), evaluation=True)

    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()
    for i in tqdm.tqdm(range(1, total_steps + 1), smoothing=0.1, dynamic_ncols=True):
        batch = train_dataset.sample(FLAGS.batch_size)
        agent, update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = (time.time() - first_time)
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        if i == 1 or i % FLAGS.eval_interval == 0:
            eval_metrics = {}
            trajs_dict = {}
            for policy_type in ['goal_skill', 'goal_skill_planning'] if FLAGS.planning_num_recursions > 0 else ['goal_skill']:
                num_episodes = FLAGS.eval_episodes
                num_video_episodes = FLAGS.num_video_episodes

                if policy_type == 'goal_skill_planning':
                    planning_info = dict(
                        num_recursions=FLAGS.planning_num_recursions,
                        num_knns=FLAGS.planning_num_knns,
                        examples=dataset.sample(FLAGS.planning_num_states),
                    )
                else:
                    planning_info = None
                eval_info, cur_trajs, renders = evaluate_with_trajectories(
                    agent, env, goal_info=goal_info, env_name=FLAGS.env_name, num_episodes=num_episodes,
                    base_observation=base_observation, num_video_episodes=num_video_episodes,
                    policy_type=policy_type, planning_info=planning_info,
                )
                eval_metrics.update({f'{policy_type}/{k}': v for k, v in eval_info.items()})
                trajs_dict[policy_type] = cur_trajs

            if FLAGS.num_video_episodes > 0:
                video = record_video('Video', i, renders=renders)
                eval_metrics['video'] = video

            traj_metrics = get_traj_v(agent, example_trajectory)
            value_viz = viz_utils.make_visual_no_image(
                traj_metrics,
                [partial(viz_utils.visualize_metric, metric_name=k) for k in traj_metrics.keys()]
            )
            eval_metrics['value_traj_viz'] = wandb.Image(value_viz)

            if 'antmaze' in FLAGS.env_name and 'large' in FLAGS.env_name:
                trajs = trajs_dict['goal_skill']
                viz_env, viz_dataset, viz = aux_env['viz_env'], aux_env['viz_dataset'], aux_env['viz']
                traj_image = d4rl_ant.trajectory_image(viz_env, viz_dataset, trajs)
                eval_metrics['trajectories'] = wandb.Image(traj_image)

                new_metrics_dist = viz.get_distance_metrics(trajs)
                eval_metrics.update({f'debugging/{k}': v for k, v in new_metrics_dist.items()})

                image_goal = d4rl_ant.gcvalue_image(
                    viz_env,
                    viz_dataset,
                    partial(get_v_goal, agent),
                )
                eval_metrics['v_goal'] = wandb.Image(image_goal)

            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

        if i % FLAGS.save_interval == 0:
            save_dict = dict(
                agent=flax.serialization.to_state_dict(agent),
            )

            fname = os.path.join(FLAGS.save_dir, f'params_{i}.pkl')
            print(f'Saving to {fname}')
            with open(fname, "wb") as f:
                pickle.dump(save_dict, f)
    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':
    app.run(main)
