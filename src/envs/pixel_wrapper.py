import gymnasium as gym
import numpy as np
import cv2


class PixelObservationWrapper(gym.ObservationWrapper):
    """
    Wraps a single-agent gymnasium environment to return 84x84x3 RGB pixel observations.
    Converts environment renders to CHW format tensors for PyTorch.
    """
    def __init__(self, env, width=84, height=84):
        super().__init__(env)
        self.width = width
        self.height = height
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(3, height, width),
            dtype=np.uint8
        )

    def observation(self, obs):
        frame = self.env.render()
        if frame is None or not isinstance(frame, np.ndarray):
            return np.zeros((3, self.height, self.width), dtype=np.uint8)
        if frame.ndim == 2:
            frame = np.stack([frame, frame, frame], axis=-1)
        elif frame.shape[2] == 4:
            frame = frame[:, :, :3]
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        frame = frame.transpose(2, 0, 1)
        return frame.astype(np.uint8)


class MultiAgentPixelWrapper:
    """
    Wraps a multi-agent environment to return per-agent 84x84x3 pixel observations.
    """
    def __init__(self, env, n_agents, width=84, height=84):
        self.env = env
        self.n_agents = n_agents
        self.width = width
        self.height = height

    def get_pixel_obs(self):
        frame = self.env.render()
        if frame is None or not isinstance(frame, np.ndarray):
            return [np.zeros((3, self.height, self.width), dtype=np.uint8)] * self.n_agents
        if frame.ndim == 2:
            frame = np.stack([frame, frame, frame], axis=-1)
        elif frame.shape[2] == 4:
            frame = frame[:, :, :3]
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        frame = frame.transpose(2, 0, 1).astype(np.uint8)
        return [frame.copy() for _ in range(self.n_agents)]


def preprocess_pixel_obs(obs, device):
    """Convert numpy pixel obs to normalized PyTorch tensor."""
    import torch
    tensor = torch.from_numpy(obs).float() / 255.0
    return tensor.unsqueeze(0).to(device)