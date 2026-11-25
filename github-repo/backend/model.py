"""Inference-only port of the thesis FNO in fourier_2d_3.py."""
from collections import OrderedDict
import inspect
import pickle
import time
import types

import numpy as np
import torch
from torch import nn


class CheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("collections", "OrderedDict"):
            return OrderedDict
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return torch._utils._rebuild_tensor_v2
        if module == "torch" and name in ("FloatStorage", "ComplexFloatStorage"):
            return getattr(torch, name)
        raise pickle.UnpicklingError("Unexpected checkpoint type: {}.{}".format(module, name))


def read_weights(path):
    if "weights_only" in inspect.signature(torch.load).parameters:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    reader = types.ModuleType("gravity_tensor_pickle")
    reader.Unpickler = CheckpointUnpickler
    reader.load = lambda handle, **kwargs: CheckpointUnpickler(handle, **kwargs).load()
    return torch.load(str(path), map_location="cpu", pickle_module=reader)


class SpectralConv3d(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.modes = modes
        for i in range(1, 5):
            self.register_parameter("weights" + str(i), nn.Parameter(
                torch.empty(width, width, *modes, dtype=torch.cfloat)
            ))

    def forward(self, x):
        a, b, c = self.modes
        dims = (-3, -2, -1)
        freq = torch.fft.rfftn(x.contiguous(), dim=dims)
        spectrum = torch.zeros_like(freq)
        for i, (sx, sy) in enumerate(((slice(a), slice(b)), (slice(-a, None), slice(b)),
                                     (slice(a), slice(-b, None)), (slice(-a, None), slice(-b, None))), 1):
            spectrum[:, :, sx, sy, :c] = torch.einsum(
                "bixyz,ioxyz->boxyz", freq[:, :, sx, sy, :c], getattr(self, "weights" + str(i))
            )
        return torch.fft.irfftn(spectrum, s=x.shape[-3:], dim=dims)


class GravityFNO(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.fc0 = nn.Linear(4, width)
        for i in range(4):
            setattr(self, "conv" + str(i), SpectralConv3d(width, modes))
            setattr(self, "w" + str(i), nn.Conv1d(width, width, 1))
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc0(x).permute(0, 4, 1, 2, 3)
        for i in range(4):
            local = getattr(self, "w" + str(i))(x.flatten(2)).reshape(x.shape)
            x = getattr(self, "conv" + str(i))(x) + local
            if i < 3:
                x = x.relu()
        x = x.permute(0, 2, 3, 4, 1)
        # The projection is pointwise; splitting the depth dimension reduces peak memory.
        slices = [self.fc2(self.fc1(chunk).relu()) for chunk in x.split(8, dim=3)]
        return torch.cat(slices, dim=3).mean(dim=3).squeeze(-1)


def load_model(path, device):
    state = read_weights(path)
    width = state["fc0.weight"].shape[0]
    modes = tuple(state["conv0.weights1"].shape[2:])
    model = GravityFNO(width, modes)
    model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False).to(device)


@torch.no_grad()
def predict(model, density, stats, device):
    n = density.shape[0]
    axes = (torch.arange(n, dtype=torch.float32, device=device) + .5) / n
    coordinates = torch.stack(torch.meshgrid(axes, axes, axes, indexing="ij"), dim=-1)
    encoded = (torch.from_numpy(density).to(device) - float(stats[0, 0])) / (float(stats[0, 1]) + 1e-5)
    features = torch.cat((encoded.unsqueeze(-1), coordinates), dim=-1).unsqueeze(0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = model(features)[0] * (float(stats[1, 1]) + 1e-5) + float(stats[1, 0])
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000
    return result.cpu().numpy().astype(np.float32, copy=False), elapsed_ms
