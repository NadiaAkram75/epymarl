"""
Pixel Observation Wrapper for EPyMARL
======================================
Converts GymmaWrapper state-vector observations to stacked pixel frames.
Handles all four benchmark environments with environment-specific backends.

Rendering backends:
  MPE        → PettingZooWrapper._env.render() with render_mode='rgb_array'
               set at construction time in GymmaWrapper
  RWARE      → inner.renderer.render(inner, return_rgb_array=True)
  LBF        → pyglet headless: viewer.window buffer capture
  Overcooked → StateVisualizer → pygame Surface → numpy

Literature:
  Frame stacking        : Mnih et al. (2015) DQN
  84x84 resolution      : standard visual RL
  Random shift (RAD)    : Laskin et al. (2020), Yarats et al. (2021) DrQ
  Shared CNN encoder    : Chu & Ye (2018) PSMADDPG, Zhu et al. (2022)
"""

import numpy as np
import cv2
from collections import deque
from gymnasium.spaces import Box


# ---------------------------------------------------------------------------
# Render backends
# ---------------------------------------------------------------------------

def _render_mpe(gymma_env):
    """
    MPE: GymmaWrapper creates MPE with render_mode='rgb_array'.
    Walk to PettingZooWrapper then call ._env.render() on the
    aec_to_parallel_wrapper directly — no mode argument needed.
    """
    try:
        obj = gymma_env._env
        while obj is not None and type(obj).__name__ != 'PettingZooWrapper':
            obj = getattr(obj, 'env', None)
        if obj is None:
            return None
        # obj._env is aec_to_parallel_wrapper
        frame = obj._env.render()
        if isinstance(frame, np.ndarray) and frame.ndim == 3:
            return frame
    except Exception:
        pass
    return None


def _render_rware(gymma_env):
    """
    RWARE: bypass gymnasium wrapper, call Viewer directly with
    return_rgb_array=True to avoid needing a display connection.
    """
    try:
        inner = gymma_env._env
        while hasattr(inner, 'env'):
            inner = inner.env
        if not hasattr(inner, 'renderer') or inner.renderer is None:
            from rware.rendering import Viewer
            inner.renderer = Viewer(inner.grid_size)
        frame = inner.renderer.render(inner, return_rgb_array=True)
        if isinstance(frame, np.ndarray) and frame.ndim == 3:
            return frame
    except Exception:
        pass
    return None


def _render_lbf(gymma_env):
    """
    LBF: pyglet headless capture.
    LBF uses pyglet not pygame. We dispatch window events and flip
    before capturing the colour buffer. Image is flipped vertically
    because pyglet stores frames bottom-to-top.
    """
    try:
        import pyglet
        inner = gymma_env._env
        while hasattr(inner, 'env'):
            inner = inner.env

        if not getattr(inner, '_rendering_initialized', False):
            pyglet.options['headless'] = True
            inner.render()

        viewer = getattr(inner, 'viewer', None)
        if viewer is None:
            return None
        win = getattr(viewer, 'window', None)
        if win is None:
            return None

        inner.render()
        win.switch_to()
        win.dispatch_events()
        win.dispatch_event('on_draw')
        win.flip()

        buf = pyglet.image.get_buffer_manager().get_color_buffer()
        img = buf.get_image_data()
        raw = np.frombuffer(
            img.get_data('RGB', img.width * 3), dtype=np.uint8
        )
        frame = raw.reshape(img.height, img.width, 3)
        return np.flipud(frame).copy()
    except Exception:
        return None


def _render_overcooked(gymma_env):
    """
    Overcooked: StateVisualizer returns a pygame Surface.
    Convert to (H, W, 3) numpy array.
    pygame surfarray gives (W, H, 3) so we transpose.
    """
    try:
        import pygame
        from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
        inner = gymma_env._env
        base_env = getattr(inner, 'base_env', None)
        if base_env is None:
            return None
        state   = base_env.state
        mdp     = base_env.mdp
        viz     = StateVisualizer()
        surface = viz.render_state(state, mdp.terrain_mtx)
        if not isinstance(surface, pygame.Surface):
            return None
        arr = pygame.surfarray.array3d(surface)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    except Exception:
        return None


_RENDER_BACKENDS = {
    "mpe":        _render_mpe,
    "spread":     _render_mpe,
    "pz-mpe":     _render_mpe,
    "foraging":   _render_lbf,
    "lbf":        _render_lbf,
    "rware":      _render_rware,
    "warehouse":  _render_rware,
    "overcooked": _render_overcooked,
}


def _get_render_fn(env_key: str):
    key_lower = env_key.lower()
    for substr, fn in _RENDER_BACKENDS.items():
        if substr in key_lower:
            return fn
    return _render_mpe


# ---------------------------------------------------------------------------
# PixelWrapper
# ---------------------------------------------------------------------------

class PixelWrapper:
    """
    Wraps a GymmaWrapper to return stacked pixel observations (C*F, H, W).

    Args:
        env          : GymmaWrapper instance
        env_key      : env key string to select render backend
        height       : frame height (default 84)
        width        : frame width  (default 84)
        n_frames     : frames to stack (default 3)
        grayscale    : use 1-channel grayscale instead of RGB
        random_shift : RAD augmentation shift in pixels (default 4)
        training     : True = random shift aug, False = centre crop only
    """

    def __init__(
        self,
        env,
        env_key: str = "",
        height: int = 84,
        width:  int = 84,
        n_frames: int = 3,
        grayscale: bool = False,
        random_shift: int = 4,
        training: bool = True,
    ):
        self.env          = env
        self.env_key      = env_key
        self.height       = height
        self.width        = width
        self.n_frames     = n_frames
        self.grayscale    = grayscale
        self.random_shift = random_shift
        self.training     = training

        self._render_fn   = _get_render_fn(env_key)
        self.n_channels   = 1 if grayscale else 3
        self.n_agents     = env.n_agents
        self.episode_limit = env.episode_limit

        self._frames = [deque(maxlen=n_frames) for _ in range(self.n_agents)]

        # channels-first layout for CNN: (n_frames * C, H, W)
        self.obs_shape = (n_frames * self.n_channels, height, width)
        self.observation_space = Box(
            low=0.0, high=1.0, shape=self.obs_shape, dtype=np.float32
        )

    # ------------------------------------------------------------------ #
    #  Frame processing                                                    #
    # ------------------------------------------------------------------ #

    def _capture(self) -> np.ndarray:
        """Capture raw frame. Returns (H, W, C) uint8, blank on failure."""
        frame = self._render_fn(self.env)
        if frame is None or not isinstance(frame, np.ndarray) or frame.ndim != 3:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame = cv2.resize(
            frame, (self.width, self.height), interpolation=cv2.INTER_AREA
        )
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = frame[:, :, np.newaxis]
        return frame

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """(H,W,C) uint8 -> (C,H,W) float32 in [0,1]."""
        return np.transpose(frame.astype(np.float32) / 255.0, (2, 0, 1))

    def _augment(self, frame: np.ndarray) -> np.ndarray:
        """
        RAD random shift augmentation (Laskin et al. 2020).
        Pads by random_shift pixels, then randomly crops back.
        Training: random crop. Eval: centre crop.
        """
        if self.random_shift == 0:
            return frame
        p = self.random_shift
        C, H, W = frame.shape
        padded = np.pad(frame, ((0, 0), (p, p), (p, p)), mode="edge")
        top  = np.random.randint(0, 2 * p) if self.training else p
        left = np.random.randint(0, 2 * p) if self.training else p
        return padded[:, top:top + H, left:left + W]

    def _process(self) -> np.ndarray:
        return self._augment(self._preprocess(self._capture()))

    def _fill(self):
        """Fill all agents' frame buffers with the current frame."""
        frame = self._process()
        for i in range(self.n_agents):
            for _ in range(self.n_frames):
                self._frames[i].append(frame.copy())

    def _push(self):
        """Push one new frame to all agents' buffers."""
        frame = self._process()
        for i in range(self.n_agents):
            self._frames[i].append(frame.copy())

    def _obs(self, agent_id: int) -> np.ndarray:
        """Stack frame buffer for agent -> (n_frames*C, H, W)."""
        return np.concatenate(list(self._frames[agent_id]), axis=0)

    # ------------------------------------------------------------------ #
    #  EPyMARL MultiAgentEnv interface                                     #
    # ------------------------------------------------------------------ #

    def reset(self, seed=None, options=None):
        self.env.reset(seed=seed, options=options)
        self._fill()
        return [self._obs(i) for i in range(self.n_agents)], {}

    def step(self, actions):
        _, reward, done, truncated, info = self.env.step(actions)
        self._push()
        return [self._obs(i) for i in range(self.n_agents)], reward, done, truncated, info

    def get_obs(self):
        return [self._obs(i) for i in range(self.n_agents)]

    def get_obs_agent(self, agent_id):
        return self._obs(agent_id)

    def get_obs_size(self):
        C, H, W = self.obs_shape
        return C * H * W

    def get_state(self):
        return np.concatenate(
            [self._obs(i).flatten() for i in range(self.n_agents)]
        ).astype(np.float32)

    def get_state_size(self):
        return self.n_agents * self.get_obs_size()

    def get_avail_actions(self):
        return self.env.get_avail_actions()

    def get_avail_agent_actions(self, agent_id):
        return self.env.get_avail_agent_actions(agent_id)

    def get_total_actions(self):
        return self.env.get_total_actions()

    def get_stats(self):
        return self.env.get_stats()

    def save_replay(self):
        self.env.save_replay()

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)

    def render(self):
        return self._capture()

    def set_training_mode(self, training: bool):
        """Switch between training (random aug) and eval (centre crop)."""
        self.training = training

    @property
    def n_actions(self):
        return self.get_total_actions()