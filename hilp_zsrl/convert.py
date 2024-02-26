import os
import platform

if 'mac' in platform.platform():
    pass
else:
    os.environ['MUJOCO_GL'] = 'egl'
    if 'SLURM_STEP_GPUS' in os.environ:
        os.environ['EGL_DEVICE_ID'] = os.environ['SLURM_STEP_GPUS']


from absl import app, flags
from pathlib import Path
import numpy as np
from pickle import HIGHEST_PROTOCOL
import torch
from tqdm import tqdm
from url_benchmark import dmc
from url_benchmark.in_memory_replay_buffer import ReplayBuffer


FLAGS = flags.FLAGS
flags.DEFINE_string('env', 'walker', '')
flags.DEFINE_string('task', 'run', '')
flags.DEFINE_string('method', 'aps', '')
flags.DEFINE_string('save_path', None, '')
flags.DEFINE_integer('num_episodes', 5000, '')
flags.DEFINE_integer('use_pixels', 0, '')
flags.DEFINE_integer('image_wh', 64, '')


def main(_):
    env = FLAGS.env
    method = FLAGS.method
    task = FLAGS.task
    buffer_dir = Path(f"{FLAGS.save_path}/datasets/{env}/{method}/buffer/")
    train_env = dmc.make(f'{env}_{task}')
    replay_loader = ReplayBuffer(max_episodes=FLAGS.num_episodes, discount=0.99, future=0.99)
    replay_loader.load(train_env, buffer_dir, relabel=True)
    if FLAGS.use_pixels:
        replay_loader._batch_names.add('pixel')
        replay_loader._storage['pixel'] = np.zeros((*replay_loader._storage['action'].shape[:2], 3, FLAGS.image_wh, FLAGS.image_wh), dtype=np.uint8)
        for i in tqdm(range(len(replay_loader))):
            for j in range(replay_loader._storage['pixel'][i].shape[0]):
                with train_env.physics.reset_context():
                    train_env.physics.set_state(replay_loader._storage['physics'][i][j])
                camera_id = dict(quadruped=2).get(env, 0)
                pixel = train_env.physics.render(height=FLAGS.image_wh, width=FLAGS.image_wh, camera_id=camera_id)
                replay_loader._storage['pixel'][i][j] = pixel.transpose(2, 0, 1)
    if not FLAGS.use_pixels:
        file_name = 'replay'
    else:
        file_name = f'replay_pixel{FLAGS.image_wh}'
    with Path(f"{FLAGS.save_path}/datasets/{env}/{method}/{file_name}.pt").open('wb') as f:
        torch.save(replay_loader, f, pickle_protocol=HIGHEST_PROTOCOL)


if __name__ == '__main__':
    app.run(main)
