import elements
import embodied
import numpy as np

from dreamerv3.market.data import load_split_npz


class Market(embodied.Env):

  def __init__(
      self,
      task,
      dataset_dir,
      split='',
      shuffle=True,
      seed=0,
      reward_mode='zero',
  ):
    split = split or task
    if split not in ('train', 'val', 'test'):
      raise ValueError(f'Expected split in train/val/test, got: {split}')

    payload = load_split_npz(dataset_dir, split)
    required = {'feat', 'ret_1', 'timestamp_ns'}
    missing = required - set(payload.keys())
    if missing:
      raise ValueError(f'Market split missing keys: {sorted(missing)}')

    self.split = split
    self.shuffle = bool(shuffle)
    self.reward_mode = reward_mode
    self.rng = np.random.default_rng(seed)

    self.feat = payload['feat'].astype(np.float32)
    self.ret_1 = payload['ret_1'].astype(np.float32)
    self.timestamp_ns = payload['timestamp_ns'].astype(np.int64)
    self.close = payload['close'].astype(np.float32) if 'close' in payload else None

    if self.feat.ndim != 3:
      raise ValueError(f'Expected feat shape [episodes, length, feat_dim], got {self.feat.shape}')
    if self.ret_1.shape[:2] != self.feat.shape[:2]:
      raise ValueError(f'feat and ret_1 shape mismatch: {self.feat.shape} vs {self.ret_1.shape}')

    self.episodes = int(self.feat.shape[0])
    self.length = int(self.feat.shape[1])
    self.feat_dim = int(self.feat.shape[2])

    self.order = np.arange(self.episodes, dtype=np.int64)
    if self.split == 'train' and self.shuffle:
      self.rng.shuffle(self.order)
    self.cursor = 0

    self.ep = 0
    self.t = 0
    self.done = True

  @property
  def obs_space(self):
    spaces = {
        'feat': elements.Space(np.float32, (self.feat_dim,)),
        'ret_1': elements.Space(np.float32, (1,)),
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
        'log/timestamp_ns': elements.Space(np.int64),
    }
    if self.close is not None:
      spaces['log/close'] = elements.Space(np.float32)
    return spaces

  @property
  def act_space(self):
    return {
        'reset': elements.Space(bool),
        'action': elements.Space(np.float32, (1,), -1.0, 1.0),
    }

  def step(self, action):
    if action['reset'] or self.done:
      self._start_episode()
      is_last = bool(self.length == 1)
      self.done = is_last
      return self._obs(is_first=True, is_last=is_last, is_terminal=is_last)

    self.t += 1
    self.done = bool(self.t >= self.length - 1)
    return self._obs(is_first=False, is_last=self.done, is_terminal=self.done)

  def _start_episode(self):
    if self.cursor >= self.episodes:
      self.cursor = 0
      if self.split == 'train' and self.shuffle:
        self.rng.shuffle(self.order)
    self.ep = int(self.order[self.cursor])
    self.cursor += 1
    self.t = 0
    self.done = False

  def _obs(self, is_first=False, is_last=False, is_terminal=False):
    reward = np.float32(0.0)
    if self.reward_mode == 'ret':
      reward = np.float32(self.ret_1[self.ep, self.t, 0])

    obs = {
        'feat': self.feat[self.ep, self.t],
        'ret_1': self.ret_1[self.ep, self.t],
        'reward': reward,
        'is_first': bool(is_first),
        'is_last': bool(is_last),
        'is_terminal': bool(is_terminal),
        'log/timestamp_ns': np.int64(self.timestamp_ns[self.ep, self.t]),
    }
    if self.close is not None:
      obs['log/close'] = np.float32(self.close[self.ep, self.t])
    return obs
