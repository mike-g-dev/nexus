#!/usr/bin/env python3
"""Generate safetensors fixtures and expected outputs for nexus-inference integration tests.

Uses explicit deterministic weights (linspace, sinusoidal) instead of random
initialization so that outputs are identical across torch versions and platforms.

Install dependencies:
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install safetensors packaging numpy
"""

import json
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from safetensors.torch import save_file

FIXTURES_DIR = Path(__file__).parent


# ---- weight initialization (deterministic, no RNG) ----


def init_linspace(param, lo=-0.2, hi=0.2):
    with torch.no_grad():
        param.copy_(
            torch.linspace(lo, hi, param.numel(), dtype=param.dtype).reshape(param.shape)
        )


def init_sinusoidal(param, lo=-0.3, hi=0.3):
    with torch.no_grad():
        n = param.numel()
        t = torch.linspace(0, 4 * math.pi, n, dtype=param.dtype)
        scale = (hi - lo) / 2
        mid = (hi + lo) / 2
        param.copy_((mid + scale * torch.sin(t)).reshape(param.shape))


def make_inputs(n_steps, n_features, seed=1):
    inputs = []
    for i in range(n_steps):
        row = []
        for j in range(n_features):
            val = 0.5 * math.sin(seed * (i + 1) * (j + 1) * 0.7)
            row.append(round(val, 8))
        inputs.append(row)
    return inputs


# ---- LSTM generators ----


def generate_rnn(name, rnn_cls, gate_mult, input_size, hidden_size, output_size,
                 inputs, init_fn, rnn_prefix, proj_prefix):
    rnn = rnn_cls(input_size, hidden_size, num_layers=1, batch_first=True)
    fc = nn.Linear(hidden_size, output_size)

    with torch.no_grad():
        init_fn(rnn.weight_ih_l0)
        init_fn(rnn.weight_hh_l0, -0.15, 0.15)
        rnn.bias_ih_l0.fill_(0.01)
        rnn.bias_hh_l0.fill_(-0.01)
        init_fn(fc.weight)
        fc.bias.fill_(0.0)

    state = {}
    for k, v in rnn.state_dict().items():
        state[f"{rnn_prefix}.{k}"] = v
    for k, v in fc.state_dict().items():
        state[f"{proj_prefix}.{k}"] = v
    save_file(state, FIXTURES_DIR / f"{name}.safetensors")

    outputs = []
    with torch.no_grad():
        if isinstance(rnn, nn.LSTM):
            h = torch.zeros(1, 1, hidden_size)
            c = torch.zeros(1, 1, hidden_size)
            for inp in inputs:
                x = torch.tensor([[inp]])
                out, (h, c) = rnn(x, (h, c))
                y = fc(out.squeeze(0)).squeeze(0)
                outputs.append(y.tolist())
        else:
            h = torch.zeros(1, 1, hidden_size)
            for inp in inputs:
                x = torch.tensor([[inp]])
                out, h = rnn(x, h)
                y = fc(out.squeeze(0)).squeeze(0)
                outputs.append(y.tolist())

    with open(FIXTURES_DIR / f"{name}_expected.json", "w") as f:
        json.dump(
            {
                "rnn_prefix": rnn_prefix,
                "proj_prefix": proj_prefix,
                "inputs": inputs,
                "outputs": outputs,
                "tolerance": 1e-5,
            },
            f,
            indent=2,
        )
        f.write("\n")

    print(f"  {name}: {len(inputs)} steps, I={input_size} H={hidden_size} O={output_size}")


def generate_lstm():
    generate_rnn("lstm", nn.LSTM, 4,
                 input_size=3, hidden_size=4, output_size=2,
                 inputs=make_inputs(5, 3, seed=1),
                 init_fn=init_linspace,
                 rnn_prefix="lstm", proj_prefix="fc")


def generate_lstm_large():
    generate_rnn("lstm_large", nn.LSTM, 4,
                 input_size=5, hidden_size=16, output_size=3,
                 inputs=make_inputs(15, 5, seed=2),
                 init_fn=init_sinusoidal,
                 rnn_prefix="encoder.lstm", proj_prefix="encoder.fc")


def generate_lstm_single_output():
    generate_rnn("lstm_single_output", nn.LSTM, 4,
                 input_size=4, hidden_size=8, output_size=1,
                 inputs=make_inputs(10, 4, seed=3),
                 init_fn=init_sinusoidal,
                 rnn_prefix="rnn", proj_prefix="head")


def generate_gru():
    generate_rnn("gru", nn.GRU, 3,
                 input_size=3, hidden_size=4, output_size=1,
                 inputs=make_inputs(5, 3, seed=1),
                 init_fn=init_linspace,
                 rnn_prefix="gru", proj_prefix="fc")


def generate_gru_large():
    generate_rnn("gru_large", nn.GRU, 3,
                 input_size=4, hidden_size=12, output_size=2,
                 inputs=make_inputs(15, 4, seed=4),
                 init_fn=init_sinusoidal,
                 rnn_prefix="gru", proj_prefix="proj")


def generate_gru_multi_output():
    generate_rnn("gru_multi_output", nn.GRU, 3,
                 input_size=3, hidden_size=6, output_size=3,
                 inputs=make_inputs(8, 3, seed=5),
                 init_fn=init_linspace,
                 rnn_prefix="seq.gru", proj_prefix="seq.fc")


# ---- MLP generators ----


def generate_mlp(name, layer_sizes, activation_cls, activation_name, dtype,
                 inputs, prefix, init_fn, tolerance, activation_param=None):
    layers = []
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        if i < len(layer_sizes) - 2:
            layers.append(activation_cls())
    mlp = nn.Sequential(*layers)
    if dtype == torch.float64:
        mlp = mlp.double()

    with torch.no_grad():
        for module in mlp:
            if isinstance(module, nn.Linear):
                init_fn(module.weight)
                module.bias.fill_(0.01)

    state = {}
    for k, v in mlp.state_dict().items():
        key = f"{prefix}.{k}" if prefix else k
        state[key] = v
    save_file(state, FIXTURES_DIR / f"{name}.safetensors")

    outputs = []
    with torch.no_grad():
        for inp in inputs:
            t = torch.tensor(inp, dtype=dtype)
            y = mlp(t)
            outputs.append(y.tolist())

    with open(FIXTURES_DIR / f"{name}_expected.json", "w") as f:
        meta = {
            "prefix": prefix,
            "activation": activation_name,
            "inputs": inputs,
            "outputs": outputs,
            "tolerance": tolerance,
        }
        if activation_param is not None:
            meta["activation_param"] = activation_param
        json.dump(meta, f, indent=2)
        f.write("\n")

    sizes_str = "->".join(str(s) for s in layer_sizes)
    print(f"  {name}: {sizes_str}, {activation_name}, {len(inputs)} inputs")


def generate_mlp_f32():
    generate_mlp("mlp_f32", [3, 8, 4, 2], nn.ReLU, "relu", torch.float32,
                 inputs=make_inputs(3, 3, seed=10),
                 prefix="mlp", init_fn=init_linspace, tolerance=1e-5)


def generate_mlp_f32_tanh():
    generate_mlp("mlp_f32_tanh", [3, 6, 2], nn.Tanh, "tanh", torch.float32,
                 inputs=make_inputs(4, 3, seed=11),
                 prefix="net", init_fn=init_sinusoidal, tolerance=1e-5)


def generate_mlp_f32_sigmoid():
    generate_mlp("mlp_f32_sigmoid", [4, 8, 4, 1], nn.Sigmoid, "sigmoid", torch.float32,
                 inputs=make_inputs(4, 4, seed=12),
                 prefix="model", init_fn=init_linspace, tolerance=1e-5)


def generate_mlp_f32_gelu():
    generate_mlp("mlp_f32_gelu", [3, 8, 2], nn.GELU, "gelu", torch.float32,
                 inputs=make_inputs(4, 3, seed=13),
                 prefix="fc", init_fn=init_sinusoidal, tolerance=1e-5)


def generate_mlp_f32_single_layer():
    generate_mlp("mlp_f32_single_layer", [3, 2], nn.ReLU, "relu", torch.float32,
                 inputs=make_inputs(3, 3, seed=14),
                 prefix="linear", init_fn=init_linspace, tolerance=1e-5)


def generate_mlp_f32_deep():
    generate_mlp("mlp_f32_deep", [2, 4, 6, 4, 3], nn.ReLU, "relu", torch.float32,
                 inputs=make_inputs(5, 2, seed=15),
                 prefix="deep", init_fn=init_sinusoidal, tolerance=1e-5)


def generate_mlp_f64():
    generate_mlp("mlp_f64", [2, 4, 1], nn.ReLU, "relu", torch.float64,
                 inputs=[[3.0, 4.0], [-1.0, 2.0], [0.0, 0.0]],
                 prefix="net", init_fn=init_linspace, tolerance=1e-10)


def generate_mlp_f64_no_prefix():
    generate_mlp("mlp_f64_no_prefix", [3, 6, 2], nn.ReLU, "relu", torch.float64,
                 inputs=make_inputs(4, 3, seed=16),
                 prefix="", init_fn=init_sinusoidal, tolerance=1e-10)


def generate_mlp_f64_tanh():
    generate_mlp("mlp_f64_tanh", [2, 8, 4, 1], nn.Tanh, "tanh", torch.float64,
                 inputs=make_inputs(5, 2, seed=17),
                 prefix="model", init_fn=init_linspace, tolerance=1e-10)


def generate_mlp_f32_swish():
    generate_mlp("mlp_f32_swish", [3, 6, 2], nn.SiLU, "swish", torch.float32,
                 inputs=make_inputs(4, 3, seed=30),
                 prefix="silu", init_fn=init_sinusoidal, tolerance=1e-5)


def generate_mlp_f32_elu():
    generate_mlp("mlp_f32_elu", [4, 8, 3], nn.ELU, "elu", torch.float32,
                 inputs=make_inputs(4, 4, seed=31),
                 prefix="elu_net", init_fn=init_linspace, tolerance=1e-5,
                 activation_param=1.0)


def generate_mlp_f32_leaky_relu():
    generate_mlp("mlp_f32_leaky_relu", [3, 6, 4, 2], nn.LeakyReLU, "leaky_relu", torch.float32,
                 inputs=make_inputs(5, 3, seed=32),
                 prefix="lrelu", init_fn=init_sinusoidal, tolerance=1e-5,
                 activation_param=0.01)


# ---- Conv1d generators ----


def generate_conv(name, input_ch, kernel_size, filters, output_size,
                  activation_cls, activation_name, inputs, conv_prefix,
                  proj_prefix, init_fn, activation_param=None):
    conv = nn.Conv1d(input_ch, filters, kernel_size)
    proj = nn.Linear(filters, output_size)

    with torch.no_grad():
        init_fn(conv.weight)
        conv.bias.fill_(0.01)
        init_fn(proj.weight)
        proj.bias.fill_(0.0)

    state = {}
    for k, v in conv.state_dict().items():
        state[f"{conv_prefix}.{k}"] = v
    for k, v in proj.state_dict().items():
        state[f"{proj_prefix}.{k}"] = v
    save_file(state, FIXTURES_DIR / f"{name}.safetensors")

    _base_fns = {
        "relu": F.relu,
        "tanh": torch.tanh,
        "sigmoid": torch.sigmoid,
        "identity": lambda x: x,
        "swish": F.silu,
    }
    if activation_name in _base_fns:
        activation_fn = _base_fns[activation_name]
    elif activation_name == "gelu":
        activation_fn = lambda x: F.gelu(x, approximate='tanh')
    elif activation_name == "elu":
        _p = activation_param if activation_param is not None else 1.0
        activation_fn = lambda x, p=_p: F.elu(x, alpha=p)
    elif activation_name == "leaky_relu":
        _p = activation_param if activation_param is not None else 0.01
        activation_fn = lambda x, p=_p: F.leaky_relu(x, negative_slope=p)
    else:
        raise ValueError(f"unknown activation: {activation_name}")

    outputs = []
    with torch.no_grad():
        x = torch.tensor(inputs, dtype=torch.float32).T.unsqueeze(0)
        x_padded = F.pad(x, (kernel_size - 1, 0))
        conv_out = conv(x_padded)
        for t in range(len(inputs)):
            activated = activation_fn(conv_out[0, :, t])
            y = proj(activated)
            outputs.append(y.tolist())

    with open(FIXTURES_DIR / f"{name}_expected.json", "w") as f:
        meta = {
            "conv_prefix": conv_prefix,
            "proj_prefix": proj_prefix,
            "activation": activation_name,
            "inputs": inputs,
            "outputs": outputs,
            "tolerance": 1e-5,
        }
        if activation_param is not None:
            meta["activation_param"] = activation_param
        json.dump(meta, f, indent=2)
        f.write("\n")

    print(f"  {name}: C={input_ch} K={kernel_size} F={filters} O={output_size}, "
          f"{activation_name}, {len(inputs)} steps")


def generate_conv1d():
    generate_conv("conv1d", input_ch=2, kernel_size=3, filters=4, output_size=1,
                  activation_cls=nn.ReLU, activation_name="relu",
                  inputs=make_inputs(5, 2, seed=20),
                  conv_prefix="conv", proj_prefix="proj", init_fn=init_linspace)


def generate_conv1d_tanh():
    generate_conv("conv1d_tanh", input_ch=3, kernel_size=4, filters=2, output_size=1,
                  activation_cls=nn.Tanh, activation_name="tanh",
                  inputs=make_inputs(10, 3, seed=21),
                  conv_prefix="conv", proj_prefix="fc", init_fn=init_sinusoidal)


def generate_conv1d_identity():
    generate_conv("conv1d_identity", input_ch=1, kernel_size=2, filters=3, output_size=2,
                  activation_cls=nn.Identity, activation_name="identity",
                  inputs=make_inputs(8, 1, seed=22),
                  conv_prefix="layer.conv", proj_prefix="layer.proj", init_fn=init_linspace)


def generate_conv1d_large():
    generate_conv("conv1d_large", input_ch=4, kernel_size=5, filters=8, output_size=2,
                  activation_cls=nn.ReLU, activation_name="relu",
                  inputs=make_inputs(15, 4, seed=23),
                  conv_prefix="enc.conv", proj_prefix="enc.proj", init_fn=init_sinusoidal)


def generate_conv1d_sigmoid():
    generate_conv("conv1d_sigmoid", input_ch=2, kernel_size=3, filters=4, output_size=1,
                  activation_cls=nn.Sigmoid, activation_name="sigmoid",
                  inputs=make_inputs(8, 2, seed=24),
                  conv_prefix="sig_conv", proj_prefix="sig_proj", init_fn=init_linspace)


def generate_conv1d_swish():
    generate_conv("conv1d_swish", input_ch=3, kernel_size=3, filters=4, output_size=2,
                  activation_cls=nn.SiLU, activation_name="swish",
                  inputs=make_inputs(6, 3, seed=33),
                  conv_prefix="swish_conv", proj_prefix="swish_proj", init_fn=init_sinusoidal)


def generate_conv1d_elu():
    generate_conv("conv1d_elu", input_ch=2, kernel_size=4, filters=3, output_size=1,
                  activation_cls=None, activation_name="elu",
                  inputs=make_inputs(8, 2, seed=34),
                  conv_prefix="elu_conv", proj_prefix="elu_proj", init_fn=init_linspace,
                  activation_param=1.5)


def generate_conv1d_leaky_relu():
    generate_conv("conv1d_leaky_relu", input_ch=2, kernel_size=3, filters=4, output_size=1,
                  activation_cls=None, activation_name="leaky_relu",
                  inputs=make_inputs(6, 2, seed=35),
                  conv_prefix="lr_conv", proj_prefix="lr_proj", init_fn=init_sinusoidal,
                  activation_param=0.1)


# ---- Fuzz generators (seeded random configs) ----


def generate_fuzz():
    rng = random.Random(42)

    activations_mlp = [
        ("relu", nn.ReLU, None),
        ("tanh", nn.Tanh, None),
        ("sigmoid", nn.Sigmoid, None),
        ("gelu", lambda: nn.GELU(approximate='tanh'), None),
        ("identity", nn.Identity, None),
        ("swish", nn.SiLU, None),
        ("elu", nn.ELU, 1.0),
        ("leaky_relu", nn.LeakyReLU, 0.01),
    ]

    activations_conv = [
        ("relu", None),
        ("tanh", None),
        ("sigmoid", None),
        ("gelu", None),
        ("identity", None),
        ("swish", None),
        ("elu", 1.0),
        ("leaky_relu", 0.01),
    ]

    init_fns = [init_linspace, init_sinusoidal]

    # Fuzz LSTM
    for i in range(4):
        input_size = rng.randint(1, 8)
        hidden_size = rng.randint(2, 16)
        output_size = rng.randint(1, 4)
        n_steps = rng.randint(5, 20)
        generate_rnn(f"fuzz_lstm_{i}", nn.LSTM, 4,
                     input_size=input_size, hidden_size=hidden_size, output_size=output_size,
                     inputs=make_inputs(n_steps, input_size, seed=100+i),
                     init_fn=rng.choice(init_fns),
                     rnn_prefix=f"fuzz{i}.lstm", proj_prefix=f"fuzz{i}.fc")

    # Fuzz GRU
    for i in range(4):
        input_size = rng.randint(1, 8)
        hidden_size = rng.randint(2, 16)
        output_size = rng.randint(1, 4)
        n_steps = rng.randint(5, 20)
        generate_rnn(f"fuzz_gru_{i}", nn.GRU, 3,
                     input_size=input_size, hidden_size=hidden_size, output_size=output_size,
                     inputs=make_inputs(n_steps, input_size, seed=200+i),
                     init_fn=rng.choice(init_fns),
                     rnn_prefix=f"fuzz{i}.gru", proj_prefix=f"fuzz{i}.proj")

    # Fuzz MLP f32
    for i in range(4):
        n_hidden = rng.randint(0, 3)
        input_size = rng.randint(1, 8)
        sizes = [input_size]
        for _ in range(n_hidden):
            sizes.append(rng.randint(2, 10))
        sizes.append(rng.randint(1, 4))
        act_name, act_cls, act_param = rng.choice(activations_mlp)
        generate_mlp(f"fuzz_mlp_f32_{i}", sizes, act_cls, act_name, torch.float32,
                     inputs=make_inputs(rng.randint(3, 8), input_size, seed=300+i),
                     prefix=f"fuzz{i}", init_fn=rng.choice(init_fns), tolerance=1e-5,
                     activation_param=act_param)

    # Fuzz MLP f64
    for i in range(2):
        n_hidden = rng.randint(0, 2)
        input_size = rng.randint(1, 6)
        sizes = [input_size]
        for _ in range(n_hidden):
            sizes.append(rng.randint(2, 8))
        sizes.append(rng.randint(1, 3))
        act_name, act_cls, act_param = rng.choice(activations_mlp)
        generate_mlp(f"fuzz_mlp_f64_{i}", sizes, act_cls, act_name, torch.float64,
                     inputs=make_inputs(rng.randint(3, 6), input_size, seed=400+i),
                     prefix=f"fuzz{i}", init_fn=rng.choice(init_fns), tolerance=1e-10,
                     activation_param=act_param)

    # Fuzz Conv1d
    for i in range(4):
        input_ch = rng.randint(1, 6)
        kernel_size = rng.randint(2, 6)
        filters = rng.randint(1, 8)
        output_size = rng.randint(1, 4)
        n_steps = rng.randint(5, 15)
        act_name, act_param = rng.choice(activations_conv)
        generate_conv(f"fuzz_conv1d_{i}", input_ch, kernel_size, filters, output_size,
                      activation_cls=None, activation_name=act_name,
                      inputs=make_inputs(n_steps, input_ch, seed=500+i),
                      conv_prefix=f"fuzz{i}.conv", proj_prefix=f"fuzz{i}.proj",
                      init_fn=rng.choice(init_fns), activation_param=act_param)


if __name__ == "__main__":
    print("Generating fixtures...")
    # LSTM
    generate_lstm()
    generate_lstm_large()
    generate_lstm_single_output()
    # GRU
    generate_gru()
    generate_gru_large()
    generate_gru_multi_output()
    # MLP f32
    generate_mlp_f32()
    generate_mlp_f32_tanh()
    generate_mlp_f32_sigmoid()
    generate_mlp_f32_gelu()
    generate_mlp_f32_single_layer()
    generate_mlp_f32_deep()
    generate_mlp_f32_swish()
    generate_mlp_f32_elu()
    generate_mlp_f32_leaky_relu()
    # MLP f64
    generate_mlp_f64()
    generate_mlp_f64_no_prefix()
    generate_mlp_f64_tanh()
    # Conv1d
    generate_conv1d()
    generate_conv1d_tanh()
    generate_conv1d_identity()
    generate_conv1d_large()
    generate_conv1d_sigmoid()
    generate_conv1d_swish()
    generate_conv1d_elu()
    generate_conv1d_leaky_relu()
    # Fuzz (seeded random configs)
    generate_fuzz()
    print("Done.")
