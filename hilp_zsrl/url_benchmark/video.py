import typing as tp
from pathlib import Path
import imageio
import numpy as np
import wandb


class VideoRecorder:
    def __init__(self,
                 root_dir: tp.Optional[tp.Union[str, Path]],
                 task: str = None,
                 render_size: int = 96,
                 fps: int = 20,
                 camera_id: int = 0,
                 use_wandb: bool = False) -> None:
        self.save_dir: tp.Optional[Path] = None
        if root_dir is not None:
            self.save_dir = Path(root_dir) / 'eval_video'
            self.save_dir.mkdir(exist_ok=True)
        self.task = task
        self.enabled = False
        self.render_size = render_size
        self.fps = fps
        self.frames: tp.List[np.ndarray] = []
        self.camera_id = camera_id
        self.use_wandb = use_wandb

    def init(self, env, enabled: bool = True) -> None:
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(env)

    def record(self, env) -> None:
        if self.enabled:
            if hasattr(env, 'physics'):
                if env.physics is not None:
                    frame = env.physics.render(height=self.render_size,
                                               width=self.render_size,
                                               camera_id=self.camera_id)
                else:
                    frame = env.base_env.render()
            else:
                frame = env.render()
            self.frames.append(frame)

    def log_to_wandb(self) -> None:
        frames = np.transpose(np.array(self.frames), (0, 3, 1, 2))
        fps, skip = 6, 8
        wandb.log({
            'eval/video':
            wandb.Video(frames[::skip, :, ::2, ::2], fps=fps, format="gif")
        })

    def save(self, file_name: str) -> None:
        if self.enabled:
            assert self.save_dir is not None
            path = self.save_dir / file_name
            imageio.mimsave(str(path), self.frames, fps=self.fps)  # type: ignore
