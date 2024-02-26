from absl import app, flags
from src import d4rl_utils
import d4rl
import numpy as np
import tqdm


FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', 'kitchen-mixed-v0', '')


def kitchen_render(kitchen_env, wh=64):
    from dm_control.mujoco import engine
    camera = engine.MovableCamera(kitchen_env.sim, wh, wh)
    camera.set_pose(distance=1.86, lookat=[-0.3, .5, 2.], azimuth=90, elevation=-60)
    img = camera.render()
    return img


def main(_):
    if 'kitchen' in FLAGS.env_name:
        env = d4rl_utils.make_env(FLAGS.env_name)
        dataset = d4rl.qlearning_dataset(env)

        env.reset()

        pixel_dataset = dataset.copy()
        pixel_obs = []
        pixel_next_obs = []
        for i in tqdm.tqdm(range(len(dataset['observations']))):
            ob = dataset['observations'][i]
            next_ob = dataset['next_observations'][i]

            env.sim.set_state(np.concatenate([ob[:30], np.zeros(29)]))
            env.sim.forward()
            pixel_ob = kitchen_render(env, wh=64)

            env.sim.set_state(np.concatenate([next_ob[:30], np.zeros(29)]))
            env.sim.forward()
            pixel_next_ob = kitchen_render(env, wh=64)

            pixel_obs.append(pixel_ob)
            pixel_next_obs.append(pixel_next_ob)
        pixel_dataset['observations'] = np.array(pixel_obs)
        pixel_dataset['next_observations'] = np.array(pixel_next_obs)

        np.savez_compressed(f'data/d4rl_kitchen_rendered/{FLAGS.env_name}.npz', **pixel_dataset)


if __name__ == '__main__':
    app.run(main)
