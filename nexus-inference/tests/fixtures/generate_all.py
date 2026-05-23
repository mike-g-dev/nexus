#!/usr/bin/env python3
"""Generate safetensors fixtures and expected outputs for nexus-inference integration tests.

Uses explicit deterministic weights (torch.linspace) instead of random
initialization so that outputs are identical across torch versions and
platforms.

Install dependencies:
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install safetensors packaging numpy
"""

import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from safetensors.torch import save_file

FIXTURES_DIR = Path(__file__).parent


def init_linspace(param, lo=-0.2, hi=0.2):
    with torch.no_grad():
        param.copy_(torch.linspace(lo, hi, param.numel(), dtype=param.dtype).reshape(param.shape))


def generate_lstm():
    input_size, hidden_size, output_size = 3, 4, 2

    lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
    fc = nn.Linear(hidden_size, output_size)

    with torch.no_grad():
        init_linspace(lstm.weight_ih_l0, -0.2, 0.2)
        init_linspace(lstm.weight_hh_l0, -0.1, 0.1)
        lstm.bias_ih_l0.fill_(0.01)
        lstm.bias_hh_l0.fill_(-0.01)
        init_linspace(fc.weight, -0.3, 0.3)
        fc.bias.fill_(0.0)

    inputs = [
        [0.5, -0.3, 0.8],
        [1.0, 0.2, -0.5],
        [-0.7, 0.4, 0.1],
        [0.3, -0.9, 0.6],
        [0.0, 0.7, -0.2],
    ]

    state = {}
    for k, v in lstm.state_dict().items():
        state[f"lstm.{k}"] = v
    for k, v in fc.state_dict().items():
        state[f"fc.{k}"] = v
    save_file(state, FIXTURES_DIR / "lstm.safetensors")

    outputs = []
    with torch.no_grad():
        h = torch.zeros(1, 1, hidden_size)
        c = torch.zeros(1, 1, hidden_size)
        for inp in inputs:
            x = torch.tensor([[inp]])
            out, (h, c) = lstm(x, (h, c))
            y = fc(out.squeeze(0)).squeeze(0)
            outputs.append(y.tolist())

    with open(FIXTURES_DIR / "lstm_expected.json", "w") as f:
        json.dump(
            {
                "rnn_prefix": "lstm",
                "proj_prefix": "fc",
                "inputs": inputs,
                "outputs": outputs,
                "tolerance": 1e-5,
            },
            f,
            indent=2,
        )
        f.write("\n")

    print(f"  lstm: {len(inputs)} steps, output_size={output_size}")


def generate_gru():
    input_size, hidden_size, output_size = 3, 4, 1

    gru = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
    fc = nn.Linear(hidden_size, output_size)

    with torch.no_grad():
        init_linspace(gru.weight_ih_l0, -0.2, 0.2)
        init_linspace(gru.weight_hh_l0, -0.1, 0.1)
        gru.bias_ih_l0.fill_(0.01)
        gru.bias_hh_l0.fill_(-0.01)
        init_linspace(fc.weight, -0.3, 0.3)
        fc.bias.fill_(0.0)

    inputs = [
        [0.5, -0.3, 0.8],
        [1.0, 0.2, -0.5],
        [-0.7, 0.4, 0.1],
        [0.3, -0.9, 0.6],
        [0.0, 0.7, -0.2],
    ]

    state = {}
    for k, v in gru.state_dict().items():
        state[f"gru.{k}"] = v
    for k, v in fc.state_dict().items():
        state[f"fc.{k}"] = v
    save_file(state, FIXTURES_DIR / "gru.safetensors")

    outputs = []
    with torch.no_grad():
        h = torch.zeros(1, 1, hidden_size)
        for inp in inputs:
            x = torch.tensor([[inp]])
            out, h = gru(x, h)
            y = fc(out.squeeze(0)).squeeze(0)
            outputs.append(y.tolist())

    with open(FIXTURES_DIR / "gru_expected.json", "w") as f:
        json.dump(
            {
                "rnn_prefix": "gru",
                "proj_prefix": "fc",
                "inputs": inputs,
                "outputs": outputs,
                "tolerance": 1e-5,
            },
            f,
            indent=2,
        )
        f.write("\n")

    print(f"  gru: {len(inputs)} steps, output_size={output_size}")


def generate_mlp_f32():
    mlp = nn.Sequential(
        nn.Linear(3, 8),
        nn.ReLU(),
        nn.Linear(8, 4),
        nn.ReLU(),
        nn.Linear(4, 2),
    )

    with torch.no_grad():
        for module in mlp:
            if isinstance(module, nn.Linear):
                init_linspace(module.weight, -0.2, 0.2)
                module.bias.fill_(0.01)

    inputs = [
        [0.5, -0.3, 0.8],
        [1.0, 0.0, -1.0],
        [-0.5, 0.5, 0.5],
    ]

    state = {}
    for k, v in mlp.state_dict().items():
        state[f"mlp.{k}"] = v
    save_file(state, FIXTURES_DIR / "mlp_f32.safetensors")

    outputs = []
    with torch.no_grad():
        for inp in inputs:
            y = mlp(torch.tensor(inp))
            outputs.append(y.tolist())

    with open(FIXTURES_DIR / "mlp_f32_expected.json", "w") as f:
        json.dump(
            {
                "prefix": "mlp",
                "activation": "relu",
                "inputs": inputs,
                "outputs": outputs,
                "tolerance": 1e-5,
            },
            f,
            indent=2,
        )
        f.write("\n")

    print(f"  mlp_f32: {len(inputs)} predictions, n_outputs=2")


def generate_mlp_f64():
    mlp = nn.Sequential(
        nn.Linear(2, 4),
        nn.ReLU(),
        nn.Linear(4, 1),
    ).double()

    with torch.no_grad():
        for module in mlp:
            if isinstance(module, nn.Linear):
                init_linspace(module.weight, -0.2, 0.2)
                module.bias.fill_(0.01)

    inputs = [
        [3.0, 4.0],
        [-1.0, 2.0],
        [0.0, 0.0],
    ]

    state = {}
    for k, v in mlp.state_dict().items():
        state[f"net.{k}"] = v
    save_file(state, FIXTURES_DIR / "mlp_f64.safetensors")

    outputs = []
    with torch.no_grad():
        for inp in inputs:
            y = mlp(torch.tensor(inp, dtype=torch.float64))
            outputs.append(y.tolist())

    with open(FIXTURES_DIR / "mlp_f64_expected.json", "w") as f:
        json.dump(
            {
                "prefix": "net",
                "activation": "relu",
                "inputs": inputs,
                "outputs": outputs,
                "tolerance": 1e-10,
            },
            f,
            indent=2,
        )
        f.write("\n")

    print(f"  mlp_f64: {len(inputs)} predictions, n_outputs=1")


def generate_conv1d():
    input_ch, kernel_size, filters, output_size = 2, 3, 4, 1

    conv = nn.Conv1d(input_ch, filters, kernel_size)
    proj = nn.Linear(filters, output_size)

    with torch.no_grad():
        init_linspace(conv.weight, -0.2, 0.2)
        conv.bias.fill_(0.01)
        init_linspace(proj.weight, -0.3, 0.3)
        proj.bias.fill_(0.0)

    inputs = [
        [0.5, -0.3],
        [1.0, 0.2],
        [-0.7, 0.4],
        [0.3, -0.9],
        [0.0, 0.7],
    ]

    state = {}
    for k, v in conv.state_dict().items():
        state[f"conv.{k}"] = v
    for k, v in proj.state_dict().items():
        state[f"proj.{k}"] = v
    save_file(state, FIXTURES_DIR / "conv1d.safetensors")

    # Causal padding: prepend kernel_size-1 zeros to match our
    # circular buffer starting state (all zeros).
    outputs = []
    with torch.no_grad():
        x = torch.tensor(inputs, dtype=torch.float32).T.unsqueeze(0)  # (1, C, L)
        x_padded = F.pad(x, (kernel_size - 1, 0))
        conv_out = conv(x_padded)  # (1, F, L)
        for t in range(len(inputs)):
            activated = F.relu(conv_out[0, :, t])
            y = proj(activated)
            outputs.append(y.tolist())

    with open(FIXTURES_DIR / "conv1d_expected.json", "w") as f:
        json.dump(
            {
                "conv_prefix": "conv",
                "proj_prefix": "proj",
                "activation": "relu",
                "inputs": inputs,
                "outputs": outputs,
                "tolerance": 1e-5,
            },
            f,
            indent=2,
        )
        f.write("\n")

    print(f"  conv1d: {len(inputs)} steps, filters={filters}")


if __name__ == "__main__":
    print("Generating fixtures...")
    generate_lstm()
    generate_gru()
    generate_mlp_f32()
    generate_mlp_f64()
    generate_conv1d()
    print("Done.")
