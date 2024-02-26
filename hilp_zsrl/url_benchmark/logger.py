import csv
import logging
import typing as tp
from pathlib import Path
import datetime
from collections import defaultdict

import torch
import wandb
from termcolor import colored
from torch.utils.tensorboard import SummaryWriter


Formating = tp.List[tp.Tuple[str, str, str]]
COMMON_TRAIN_FORMAT = [('frame', 'F', 'int'), ('step', 'S', 'int'),
                       ('episode', 'E', 'int'), ('episode_length', 'L', 'int'),
                       ('episode_reward', 'R', 'float'),
                       ('fps', 'FPS', 'float'), ('total_time', 'T', 'time')]

COMMON_EVAL_FORMAT = [('frame', 'F', 'int'), ('step', 'S', 'int'),
                      ('episode', 'E', 'int'), ('episode_length', 'L', 'int'),
                      ('episode_reward', 'R', 'float'),
                      ('total_time', 'T', 'time')]


pylogger = logging.getLogger(__name__)


class AverageMeter:
    def __init__(self) -> None:
        self._sum = 0.0
        self._count = 0

    def update(self, value: float, n: int = 1) -> None:
        self._sum += value
        self._count += n

    def value(self) -> float:
        return self._sum / max(1, self._count)


Metrics = tp.Dict[str, float]


class MetersGroup:
    def __init__(self, csv_file_name: tp.Union[Path, str], formating: Formating, use_wandb: bool) -> None:
        self._csv_file_name = Path(csv_file_name)
        self._formating = formating
        self._meters: tp.Dict[str, AverageMeter] = defaultdict(AverageMeter)
        self._csv_file: tp.Optional[tp.TextIO] = None
        self._csv_writer: tp.Optional[csv.DictWriter[str]] = None
        self.use_wandb = use_wandb

    def log(self, key: str, value: float, n: int = 1) -> None:
        self._meters[key].update(value, n)

    def _prime_meters(self) -> Metrics:
        data = {}
        for key, meter in self._meters.items():
            if key.startswith('train'):
                key = key[len('train') + 1:]
            else:
                key = key[len('eval') + 1:]
            key = key.replace('/', '_')
            data[key] = meter.value()
        return data

    def _remove_old_entries(self, data: Metrics) -> None:
        rows = []
        with self._csv_file_name.open('r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if float(row['episode']) >= data['episode']:
                    break
                rows.append(row)
        with self._csv_file_name.open('w') as f:
            writer = csv.DictWriter(f,
                                    fieldnames=sorted(data.keys()),
                                    restval=0.0)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _dump_to_csv(self, data: Metrics) -> None:
        if self._csv_writer is None:
            should_write_header = True
            if self._csv_file_name.exists():
                self._remove_old_entries(data)
                should_write_header = False

            self._csv_file = self._csv_file_name.open('a')
            self._csv_writer = csv.DictWriter(self._csv_file,
                                              fieldnames=sorted(data.keys()),
                                              restval=0.0)
            if should_write_header:
                self._csv_writer.writeheader()
        if self._csv_writer is None or self._csv_file is None:
            raise RuntimeError("CSV writer and file should have been instantiated")

        self._csv_writer.writerow(data)
        self._csv_file.flush()

    @staticmethod
    def _format(key: str, value: float, ty: str) -> str:
        if ty == 'int':
            value = int(value)
            return f'{key}: {value}'
        elif ty == 'float':
            return f'{key}: {value:.04f}'
        elif ty == 'time':
            value_ = str(datetime.timedelta(seconds=int(value)))
            return f'{key}: {value_}'
        raise ValueError(f'invalid format type: {ty}')

    def _dump_to_console(self, data: Metrics, prefix: str) -> None:
        prefix = colored(prefix, 'yellow' if prefix == 'train' else 'green')
        pieces = [f'| {prefix: <14}']
        for key, disp_key, ty in self._formating:
            value = data.get(key, 0)
            pieces.append(self._format(disp_key, value, ty))
        print(' | '.join(pieces))

    @staticmethod
    def _dump_to_wandb(data: Metrics, step: int) -> None:
        wandb.log(data, step=step)

    def dump(self, step: int, prefix: str) -> None:
        if len(self._meters) == 0:
            return
        data = self._prime_meters()
        data['frame'] = step
        if self.use_wandb:
            wandb_data = {prefix + '/' + key: val for key, val in data.items()}
            self._dump_to_wandb(data=wandb_data, step=step)
        self._dump_to_csv(data)
        self._dump_to_console(data, prefix)
        self._meters.clear()


class Logger:
    def __init__(self, log_dir: Path, use_tb: bool, use_wandb: bool) -> None:
        self._log_dir = log_dir

        self._train_mg = MetersGroup(log_dir / 'train.csv',
                                     formating=COMMON_TRAIN_FORMAT,
                                     use_wandb=use_wandb)
        self._eval_mg = MetersGroup(log_dir / 'eval.csv',
                                    formating=COMMON_EVAL_FORMAT,
                                    use_wandb=use_wandb)
        self._sw: tp.Optional[SummaryWriter] = None
        if use_tb:
            self._sw = SummaryWriter(str(log_dir / 'tb'))
        self.use_wandb = use_wandb

    def _try_sw_log(self, key, value, step) -> None:
        if self._sw is not None:
            self._sw.add_scalar(key, value, step)

    def log(self, key: str, value: tp.Union[float, torch.Tensor], step: int) -> None:
        assert key.startswith('train') or key.startswith('eval')
        if isinstance(value, torch.Tensor):
            value = value.item()
        self._try_sw_log(key, value, step)
        mg = self._train_mg if key.startswith('train') else self._eval_mg
        mg.log(key, value)

    def log_metrics(self, metrics: tp.Dict[str, float], step: int, ty: str) -> None:
        for key, value in metrics.items():
            self.log(f'{ty}/{key}', value, step)

    def dump(self, step, ty=None) -> None:
        try:
            if ty is None or ty == 'eval':
                self._eval_mg.dump(step, 'eval')
            if ty is None or ty == 'train':
                self._train_mg.dump(step, 'train')
        except ValueError as e:
            pylogger.warning(f"Could not dump metrics: {e}")

    def log_and_dump_ctx(self, step: int, ty: str) -> "LogAndDumpCtx":
        return LogAndDumpCtx(self, step, ty)


class LogAndDumpCtx:
    def __init__(self, logger: Logger, step: int, ty: str) -> None:
        self._logger = logger
        self._step = step
        self._ty = ty

    def __enter__(self) -> "LogAndDumpCtx":
        return self

    def __call__(self, key: str, value: float) -> None:
        self._logger.log(f'{self._ty}/{key}', value, self._step)

    def __exit__(self, *args: tp.Any) -> None:
        self._logger.dump(self._step, self._ty)
