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
    assert rnn.weight_ih_l0.shape[0] == gate_mult * hidden_size, \
        f"{name}: gate_mult={gate_mult} disagrees with {type(rnn).__name__} gate layout"
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


# ---- Stacked RNN generators ----


def generate_stacked_rnn(name, rnn_cls, gate_mult, input_size, hidden_size,
                         output_size, num_layers, inputs, init_fn,
                         rnn_prefix, proj_prefix, tolerance=1e-5):
    rnn = rnn_cls(input_size, hidden_size, num_layers=num_layers, batch_first=True)
    assert rnn.weight_ih_l0.shape[0] == gate_mult * hidden_size, \
        f"{name}: gate_mult={gate_mult} disagrees with {type(rnn).__name__} gate layout"
    fc = nn.Linear(hidden_size, output_size)

    with torch.no_grad():
        for k in range(num_layers):
            wih = getattr(rnn, f"weight_ih_l{k}")
            whh = getattr(rnn, f"weight_hh_l{k}")
            bih = getattr(rnn, f"bias_ih_l{k}")
            bhh = getattr(rnn, f"bias_hh_l{k}")
            init_fn(wih)
            init_fn(whh, -0.15, 0.15)
            bih.fill_(0.01)
            bhh.fill_(-0.01)
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
            h = torch.zeros(num_layers, 1, hidden_size)
            c = torch.zeros(num_layers, 1, hidden_size)
            for inp in inputs:
                x = torch.tensor([[inp]])
                out, (h, c) = rnn(x, (h, c))
                y = fc(out.squeeze(0)).squeeze(0)
                outputs.append(y.tolist())
        else:
            h = torch.zeros(num_layers, 1, hidden_size)
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
                "num_layers": num_layers,
                "inputs": inputs,
                "outputs": outputs,
                "tolerance": tolerance,
            },
            f,
            indent=2,
        )
        f.write("\n")

    print(f"  {name}: {len(inputs)} steps, L={num_layers} I={input_size} H={hidden_size} O={output_size}")


def generate_stacked_lstm_2layer():
    generate_stacked_rnn("stacked_lstm_2layer", nn.LSTM, 4,
                         input_size=3, hidden_size=4, output_size=2,
                         num_layers=2,
                         inputs=make_inputs(8, 3, seed=10),
                         init_fn=init_linspace,
                         rnn_prefix="lstm", proj_prefix="fc")


def generate_stacked_lstm_3layer():
    generate_stacked_rnn("stacked_lstm_3layer", nn.LSTM, 4,
                         input_size=4, hidden_size=8, output_size=1,
                         num_layers=3,
                         inputs=make_inputs(12, 4, seed=11),
                         init_fn=init_sinusoidal,
                         rnn_prefix="encoder.lstm", proj_prefix="encoder.fc")


def generate_stacked_lstm_large():
    generate_stacked_rnn("stacked_lstm_large", nn.LSTM, 4,
                         input_size=6, hidden_size=16, output_size=3,
                         num_layers=2,
                         inputs=make_inputs(15, 6, seed=12),
                         init_fn=init_sinusoidal,
                         rnn_prefix="rnn", proj_prefix="head")


def generate_stacked_gru_2layer():
    generate_stacked_rnn("stacked_gru_2layer", nn.GRU, 3,
                         input_size=3, hidden_size=4, output_size=1,
                         num_layers=2,
                         inputs=make_inputs(8, 3, seed=13),
                         init_fn=init_linspace,
                         rnn_prefix="gru", proj_prefix="fc")


def generate_stacked_gru_3layer():
    generate_stacked_rnn("stacked_gru_3layer", nn.GRU, 3,
                         input_size=4, hidden_size=6, output_size=2,
                         num_layers=3,
                         inputs=make_inputs(10, 4, seed=14),
                         init_fn=init_sinusoidal,
                         rnn_prefix="model.gru", proj_prefix="model.fc")


def generate_stacked_gru_large():
    generate_stacked_rnn("stacked_gru_large", nn.GRU, 3,
                         input_size=5, hidden_size=12, output_size=2,
                         num_layers=2,
                         inputs=make_inputs(12, 5, seed=15),
                         init_fn=init_sinusoidal,
                         rnn_prefix="gru", proj_prefix="proj")


# ---- MLP generators ----


def generate_mlp(name, layer_sizes, activation_cls, activation_name, dtype,
                 inputs, prefix, init_fn, tolerance, activation_param=None,
                 bias=True, batchnorm=False, layernorm=False):
    layers = []
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1], bias=bias))
        if i < len(layer_sizes) - 2:
            if batchnorm:
                layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))
            if layernorm:
                layers.append(nn.LayerNorm(layer_sizes[i + 1]))
            layers.append(activation_cls())
    mlp = nn.Sequential(*layers)
    if dtype == torch.float64:
        mlp = mlp.double()

    with torch.no_grad():
        for module in mlp:
            if isinstance(module, nn.Linear):
                init_fn(module.weight)
                if module.bias is not None:
                    module.bias.fill_(0.01)
            elif isinstance(module, nn.BatchNorm1d):
                n = module.num_features
                module.running_mean.copy_(
                    torch.linspace(-0.5, 0.5, n, dtype=module.running_mean.dtype))
                module.running_var.copy_(
                    torch.linspace(0.5, 2.0, n, dtype=module.running_var.dtype))
                if module.affine:
                    init_sinusoidal(module.weight, 0.8, 1.2)
                    module.bias.fill_(0.05)
            elif isinstance(module, nn.LayerNorm):
                init_sinusoidal(module.weight, 0.8, 1.2)
                module.bias.fill_(0.05)

    if batchnorm:
        mlp.eval()

    state = {}
    for k, v in mlp.state_dict().items():
        key = f"{prefix}.{k}" if prefix else k
        state[key] = v
    save_file(state, FIXTURES_DIR / f"{name}.safetensors")

    outputs = []
    with torch.no_grad():
        for inp in inputs:
            t = torch.tensor(inp, dtype=dtype)
            if batchnorm:
                t = t.unsqueeze(0)
            y = mlp(t)
            if batchnorm:
                y = y.squeeze(0)
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

    tags = []
    if not bias:
        tags.append("bias=False")
    if batchnorm:
        tags.append("BN")
    if layernorm:
        tags.append("LN")
    tag_str = f" [{', '.join(tags)}]" if tags else ""
    sizes_str = "->".join(str(s) for s in layer_sizes)
    print(f"  {name}: {sizes_str}, {activation_name}{tag_str}, {len(inputs)} inputs")


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


def generate_mlp_f32_no_bias():
    generate_mlp("mlp_f32_no_bias", [3, 6, 4, 2], nn.ReLU, "relu", torch.float32,
                 inputs=make_inputs(4, 3, seed=40),
                 prefix="nb", init_fn=init_linspace, tolerance=1e-5,
                 bias=False)


def generate_mlp_f32_batchnorm():
    generate_mlp("mlp_f32_batchnorm", [3, 8, 4, 2], nn.ReLU, "relu", torch.float32,
                 inputs=make_inputs(4, 3, seed=41),
                 prefix="bn", init_fn=init_linspace, tolerance=1e-5,
                 batchnorm=True)


def generate_mlp_f32_batchnorm_no_bias():
    generate_mlp("mlp_f32_batchnorm_no_bias", [4, 8, 4, 1], nn.ReLU, "relu", torch.float32,
                 inputs=make_inputs(5, 4, seed=42),
                 prefix="bnb", init_fn=init_sinusoidal, tolerance=1e-5,
                 bias=False, batchnorm=True)


def generate_mlp_f32_layernorm():
    generate_mlp("mlp_f32_layernorm", [3, 8, 4, 2], nn.ReLU, "relu", torch.float32,
                 inputs=make_inputs(4, 3, seed=43),
                 prefix="ln", init_fn=init_linspace, tolerance=1e-5,
                 layernorm=True)


def generate_mlp_f32_layernorm_no_bias():
    generate_mlp("mlp_f32_layernorm_no_bias", [4, 8, 4, 1], nn.ReLU, "relu", torch.float32,
                 inputs=make_inputs(5, 4, seed=44),
                 prefix="lnb", init_fn=init_sinusoidal, tolerance=1e-5,
                 bias=False, layernorm=True)


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


# ---- SSM generators ----


def generate_ssm(name, input_size, hidden_size, output_size, inputs,
                 prefix, init_fn, has_d=True):
    a_diag = torch.empty(hidden_size)
    b = torch.empty(hidden_size, input_size)
    c = torch.empty(output_size, hidden_size)

    init_fn(a_diag, 0.5, 0.99)
    init_fn(b)
    init_fn(c)

    state = {
        f"{prefix}.a_diag": a_diag,
        f"{prefix}.b": b,
        f"{prefix}.c": c,
    }

    if has_d:
        d = torch.empty(output_size, input_size)
        init_fn(d, -0.05, 0.05)
        state[f"{prefix}.d"] = d
    else:
        d = torch.zeros(output_size, input_size)

    save_file(state, FIXTURES_DIR / f"{name}.safetensors")

    outputs = []
    h = torch.zeros(hidden_size)
    with torch.no_grad():
        for inp in inputs:
            u = torch.tensor(inp, dtype=torch.float32)
            h = a_diag * h + b @ u
            y = c @ h + d @ u
            outputs.append(y.tolist())

    with open(FIXTURES_DIR / f"{name}_expected.json", "w") as f:
        json.dump({
            "prefix": prefix,
            "has_d": has_d,
            "inputs": inputs,
            "outputs": outputs,
            "tolerance": 1e-5,
        }, f, indent=2)
        f.write("\n")

    d_tag = "" if has_d else " [no D]"
    print(f"  {name}: I={input_size} H={hidden_size} O={output_size}{d_tag}, {len(inputs)} steps")


def generate_ssm_basic():
    generate_ssm("ssm", input_size=2, hidden_size=4, output_size=1,
                 inputs=make_inputs(5, 2, seed=50),
                 prefix="ssm", init_fn=init_linspace)


def generate_ssm_no_skip():
    generate_ssm("ssm_no_skip", input_size=3, hidden_size=4, output_size=1,
                 inputs=make_inputs(8, 3, seed=51),
                 prefix="model.ssm", init_fn=init_sinusoidal, has_d=False)


def generate_ssm_multi_output():
    generate_ssm("ssm_multi_output", input_size=2, hidden_size=6, output_size=3,
                 inputs=make_inputs(8, 2, seed=52),
                 prefix="enc.ssm", init_fn=init_linspace)


def generate_ssm_large():
    generate_ssm("ssm_large", input_size=4, hidden_size=16, output_size=2,
                 inputs=make_inputs(15, 4, seed=53),
                 prefix="ssm", init_fn=init_sinusoidal)


# ---- BNN generators ----


def make_binary_weights(hidden_size, init_fn, lo=-0.3, hi=0.3):
    """Generate deterministic ±1 weights by thresholding initialized floats.

    Uses >= 0 convention (matching Rust binarize): 0.0 → +1.
    """
    tmp = torch.empty(hidden_size, hidden_size)
    init_fn(tmp, lo, hi)
    return torch.where(tmp >= 0,
                       torch.tensor(1, dtype=torch.int8),
                       torch.tensor(-1, dtype=torch.int8))


def generate_bnn(name, input_size, hidden_size, output_size, num_binary,
                 inputs, prefix, init_fn, tolerance=1e-5):
    # fp32 input layer
    w_input = torch.empty(hidden_size, input_size)
    b_input = torch.empty(hidden_size)
    init_fn(w_input)
    init_fn(b_input, -0.1, 0.1)

    # binary layers (i8 ±1 weights, fp32 biases from BN folding)
    binary_weights = []
    binary_biases = []
    for k in range(num_binary):
        bw = make_binary_weights(hidden_size, init_fn,
                                 -0.3 + 0.1 * k, 0.3 + 0.1 * k)
        bb = torch.empty(hidden_size)
        init_fn(bb, -0.5, 0.5)
        binary_weights.append(bw)
        binary_biases.append(bb)

    # fp32 output layer
    w_output = torch.empty(output_size, hidden_size)
    b_output = torch.empty(output_size)
    init_fn(w_output)
    init_fn(b_output, -0.05, 0.05)

    # Save tensors
    state = {
        f"{prefix}.input_weight": w_input.float(),
        f"{prefix}.input_bias": b_input.float(),
        f"{prefix}.output_weight": w_output.float(),
        f"{prefix}.output_bias": b_output.float(),
    }
    for k, (bw, bb) in enumerate(zip(binary_weights, binary_biases)):
        state[f"{prefix}.binary_weight_{k}"] = bw
        state[f"{prefix}.binary_bias_{k}"] = bb.float()

    save_file(state, FIXTURES_DIR / f"{name}.safetensors")

    # Forward pass (manual BNN equations)
    outputs = []
    with torch.no_grad():
        for inp in inputs:
            x = torch.tensor(inp, dtype=torch.float32)

            # fp32 input layer
            h = w_input @ x + b_input

            # binarize: sign(h), with sign(0) = +1
            h = torch.where(h >= 0, torch.ones_like(h), -torch.ones_like(h))

            # binary hidden layers
            for bw, bb in zip(binary_weights, binary_biases):
                # i8 matmul (cast to float for torch, but result is exact integer)
                dot = bw.float() @ h + bb
                h = torch.where(dot >= 0, torch.ones_like(dot), -torch.ones_like(dot))

            # fp32 output layer
            y = w_output @ h + b_output
            outputs.append(y.tolist())

    with open(FIXTURES_DIR / f"{name}_expected.json", "w") as f:
        json.dump({
            "prefix": prefix,
            "num_binary": num_binary,
            "inputs": inputs,
            "outputs": outputs,
            "tolerance": tolerance,
        }, f, indent=2)
        f.write("\n")

    bl_tag = f", {num_binary} binary" if num_binary > 0 else ", no binary"
    print(f"  {name}: I={input_size} H={hidden_size} O={output_size}{bl_tag}, {len(inputs)} steps")


def generate_bnn_basic():
    generate_bnn("bnn", input_size=4, hidden_size=64, output_size=1,
                 num_binary=0,
                 inputs=make_inputs(5, 4, seed=60),
                 prefix="bnn", init_fn=init_linspace)


def generate_bnn_one_binary():
    generate_bnn("bnn_one_binary", input_size=4, hidden_size=64, output_size=1,
                 num_binary=1,
                 inputs=make_inputs(5, 4, seed=61),
                 prefix="model.bnn", init_fn=init_sinusoidal)


def generate_bnn_two_binary():
    generate_bnn("bnn_two_binary", input_size=4, hidden_size=64, output_size=2,
                 num_binary=2,
                 inputs=make_inputs(8, 4, seed=62),
                 prefix="enc.bnn", init_fn=init_linspace)


def generate_bnn_large():
    generate_bnn("bnn_large", input_size=8, hidden_size=128, output_size=2,
                 num_binary=2,
                 inputs=make_inputs(10, 8, seed=63),
                 prefix="bnn", init_fn=init_sinusoidal)


# ---- Fuzz generators (seeded random configs) ----


def generate_fuzz():
    # Each fuzz family below seeds its own independent RNG (distinct fixed
    # seed) so that adding or reordering a family never perturbs the configs
    # or fixtures of the others.

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
    rng = random.Random(42)
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
    rng = random.Random(142)
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
    rng = random.Random(242)
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

    # Fuzz Conv1d
    rng = random.Random(442)
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

    # Fuzz Stacked LSTM
    rng = random.Random(542)
    for i in range(4):
        input_size = rng.randint(1, 8)
        hidden_size = rng.randint(2, 16)
        output_size = rng.randint(1, 4)
        num_layers = rng.randint(2, 4)
        n_steps = rng.randint(5, 20)
        generate_stacked_rnn(f"fuzz_stacked_lstm_{i}", nn.LSTM, 4,
                             input_size=input_size, hidden_size=hidden_size,
                             output_size=output_size, num_layers=num_layers,
                             inputs=make_inputs(n_steps, input_size, seed=600+i),
                             init_fn=rng.choice(init_fns),
                             rnn_prefix=f"fuzz{i}.lstm", proj_prefix=f"fuzz{i}.fc",
                             tolerance=5e-5)

    # Fuzz Stacked GRU
    rng = random.Random(642)
    for i in range(4):
        input_size = rng.randint(1, 8)
        hidden_size = rng.randint(2, 16)
        output_size = rng.randint(1, 4)
        num_layers = rng.randint(2, 4)
        n_steps = rng.randint(5, 20)
        generate_stacked_rnn(f"fuzz_stacked_gru_{i}", nn.GRU, 3,
                             input_size=input_size, hidden_size=hidden_size,
                             output_size=output_size, num_layers=num_layers,
                             inputs=make_inputs(n_steps, input_size, seed=700+i),
                             init_fn=rng.choice(init_fns),
                             rnn_prefix=f"fuzz{i}.gru", proj_prefix=f"fuzz{i}.proj",
                             tolerance=5e-5)

    # Fuzz SSM
    rng = random.Random(742)
    for i in range(4):
        input_size = rng.randint(1, 6)
        hidden_size = rng.randint(2, 16)
        output_size = rng.randint(1, 4)
        n_steps = rng.randint(5, 20)
        has_d = rng.choice([True, False])
        generate_ssm(f"fuzz_ssm_{i}",
                      input_size=input_size, hidden_size=hidden_size,
                      output_size=output_size,
                      inputs=make_inputs(n_steps, input_size, seed=800+i),
                      prefix=f"fuzz{i}.ssm", init_fn=rng.choice(init_fns),
                      has_d=has_d)

    # Fuzz TCN
    rng = random.Random(842)
    for i in range(4):
        input_size = rng.randint(1, 4)
        filters = rng.choice([2, 4, 8])
        kernel_size = rng.choice([2, 3])
        num_layers = rng.randint(1, 4)
        output_size = rng.randint(1, 2)
        residual = rng.choice([True, False])
        act = rng.choice(["relu", "identity", "tanh"])
        rf = 1 + (kernel_size - 1) * (2**num_layers - 1)
        n_steps = max(rf + 5, 10)
        generate_tcn(f"fuzz_tcn_{i}",
                     input_size=input_size, filters=filters,
                     kernel_size=kernel_size, num_layers=num_layers,
                     output_size=output_size, activation_name=act,
                     residual=residual,
                     inputs=make_inputs(n_steps, input_size, seed=1000+i),
                     prefix=f"fuzz{i}.tcn", init_fn=rng.choice(init_fns))

    # Fuzz Quantized MLP
    rng = random.Random(1042)
    activations_qmlp = ["relu", "identity", "tanh", "sigmoid"]
    for i in range(4):
        n_hidden = rng.randint(0, 2)
        input_size = rng.randint(1, 8)
        sizes = [input_size]
        for _ in range(n_hidden):
            sizes.append(rng.randint(2, 10))
        sizes.append(rng.randint(1, 4))
        act_name = rng.choice(activations_qmlp)
        symmetric = rng.choice([True, False])
        generate_quantized_mlp(f"fuzz_quantized_mlp_{i}", sizes, act_name,
                               inputs=make_inputs(rng.randint(3, 8), input_size, seed=1200+i),
                               prefix=f"fuzz{i}.qmlp", init_fn=rng.choice(init_fns),
                               tolerance=1e-3, symmetric=symmetric)

    # Fuzz BNN
    rng = random.Random(942)
    for i in range(4):
        input_size = rng.randint(1, 8)
        hidden_size = rng.choice([64, 128])
        output_size = rng.randint(1, 4)
        num_binary = rng.randint(0, 3)
        n_steps = rng.randint(3, 8)
        generate_bnn(f"fuzz_bnn_{i}",
                     input_size=input_size, hidden_size=hidden_size,
                     output_size=output_size, num_binary=num_binary,
                     inputs=make_inputs(n_steps, input_size, seed=900+i),
                     prefix=f"fuzz{i}.bnn", init_fn=rng.choice(init_fns),
                     tolerance=2e-5)


# ---- TCN generators ----


def generate_tcn(name, input_size, filters, kernel_size, num_layers, output_size,
                 activation_name, residual, inputs, prefix, init_fn,
                 activation_param=None):
    convs = []
    for i in range(num_layers):
        in_ch = input_size if i == 0 else filters
        convs.append(nn.Conv1d(in_ch, filters, kernel_size, dilation=2**i))

    output_proj = nn.Linear(filters, output_size)

    with torch.no_grad():
        for conv in convs:
            init_fn(conv.weight)
            conv.bias.fill_(0.01)
        init_fn(output_proj.weight)
        output_proj.bias.fill_(0.0)

    state = {}
    for i, conv in enumerate(convs):
        for k, v in conv.state_dict().items():
            state[f"{prefix}.conv_{i}.{k}"] = v
    for k, v in output_proj.state_dict().items():
        state[f"{prefix}.output.{k}"] = v
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
        x = torch.tensor(inputs, dtype=torch.float32).T.unsqueeze(0)  # (1, C, T)
        for i in range(num_layers):
            d = 2 ** i
            pad = (kernel_size - 1) * d
            x_prev = x
            x = activation_fn(convs[i](F.pad(x, (pad, 0))))
            if residual and (i > 0 or input_size == filters):
                x = x + x_prev

        for t in range(len(inputs)):
            y = output_proj(x[0, :, t])
            outputs.append(y.tolist())

    with open(FIXTURES_DIR / f"{name}_expected.json", "w") as f:
        meta = {
            "prefix": prefix,
            "activation": activation_name,
            "residual": residual,
            "inputs": inputs,
            "outputs": outputs,
            "tolerance": 1e-5,
        }
        if activation_param is not None:
            meta["activation_param"] = activation_param
        json.dump(meta, f, indent=2)
        f.write("\n")

    rf = 1 + (kernel_size - 1) * (2**num_layers - 1)
    print(f"  {name}: I={input_size} F={filters} K={kernel_size} L={num_layers} "
          f"O={output_size}, {activation_name}, residual={residual}, RF={rf}, "
          f"{len(inputs)} steps")


def generate_tcn_basic():
    generate_tcn("tcn", input_size=2, filters=4, kernel_size=3, num_layers=2,
                 output_size=1, activation_name="relu", residual=False,
                 inputs=make_inputs(20, 2, seed=2000),
                 prefix="tcn", init_fn=init_linspace)


def generate_tcn_residual():
    generate_tcn("tcn_residual", input_size=4, filters=4, kernel_size=3,
                 num_layers=3, output_size=1, activation_name="relu",
                 residual=True,
                 inputs=make_inputs(20, 4, seed=2001),
                 prefix="tcn_r", init_fn=init_sinusoidal)


def generate_tcn_identity():
    generate_tcn("tcn_identity", input_size=2, filters=4, kernel_size=2,
                 num_layers=1, output_size=2, activation_name="identity",
                 residual=False,
                 inputs=make_inputs(10, 2, seed=2002),
                 prefix="tcn_id", init_fn=init_linspace)


def generate_tcn_large():
    generate_tcn("tcn_large", input_size=4, filters=8, kernel_size=3,
                 num_layers=4, output_size=2, activation_name="relu",
                 residual=True,
                 inputs=make_inputs(40, 4, seed=2003),
                 prefix="tcn_lg", init_fn=init_sinusoidal)


# ---- Quantized MLP generators ----


def quantize_to_i8(x_f32, scale, zero_point):
    """Quantize f32 → i8 matching Rust quantize_f32_to_i8."""
    return [max(-128, min(127, round(v / scale) + zero_point)) for v in x_f32]


def quantized_mlp_forward(layers, inputs, activation_name, activation_param=None):
    """Simulate quantized MLP inference matching Rust predict_into exactly.

    Each layer: quantize_input→i8 matmul→i32→zero-point corrections→dequant→activation.
    """
    import numpy as np

    outputs = []
    n_layers = len(layers)

    for inp in inputs:
        x = list(inp)
        for layer_idx in range(n_layers):
            layer = layers[layer_idx]
            w_i8 = layer["w_i8"]         # list of i8 (out * in, row-major)
            bias = layer["bias"]         # list of f32
            w_scale = layer["w_scale"]
            w_zp = layer["w_zp"]
            in_scale = layer["in_scale"]
            in_zp = layer["in_zp"]
            in_size = layer["in_size"]
            out_size = layer["out_size"]
            is_last = (layer_idx == n_layers - 1)

            inv_in_scale = 1.0 / in_scale
            combined_scale = w_scale * in_scale

            # Step 1: Quantize input
            x_i8 = quantize_to_i8(x, in_scale, in_zp)

            # Step 2: Integer matmul (matching Rust scalar path)
            acc = [0] * out_size
            for j in range(out_size):
                s = 0
                for i in range(in_size):
                    s += w_i8[j * in_size + i] * x_i8[i]
                acc[j] = s

            # Step 3: Zero-point corrections + dequant + bias + activation
            # Precompute row_sum for each output neuron
            row_sums = []
            for j in range(out_size):
                rs = sum(w_i8[j * in_size + i] for i in range(in_size))
                row_sums.append(rs)

            result = [0.0] * out_size
            for j in range(out_size):
                a = acc[j]

                # Correct for input zero point
                if in_zp != 0:
                    a -= in_zp * row_sums[j]

                # Correct for weight zero point
                if w_zp != 0:
                    input_sum = sum(x_i8)
                    a -= w_zp * input_sum
                    a += in_size * w_zp * in_zp

                # Dequantize + bias (use Python float mul_add equivalent)
                y = float(a) * combined_scale + bias[j]

                # Activation (not on last layer)
                if not is_last:
                    if activation_name == "relu":
                        y = max(0.0, y)
                    elif activation_name == "tanh":
                        y = math.tanh(y)
                    elif activation_name == "sigmoid":
                        y = 1.0 / (1.0 + math.exp(-y))
                    elif activation_name == "identity":
                        pass
                    elif activation_name == "swish":
                        y = y / (1.0 + math.exp(-y))
                    elif activation_name == "elu":
                        alpha = activation_param if activation_param else 1.0
                        if y < 0:
                            y = alpha * (math.exp(y) - 1.0)
                    elif activation_name == "leaky_relu":
                        alpha = activation_param if activation_param else 0.01
                        if y < 0:
                            y = alpha * y
                    elif activation_name == "gelu":
                        y = 0.5 * y * (1.0 + math.tanh(
                            math.sqrt(2.0 / math.pi) * (y + 0.044715 * y**3)))

                result[j] = y

            x = result

        outputs.append(x)

    return outputs


def generate_quantized_mlp(name, layer_sizes, activation_name, inputs, prefix,
                           init_fn, tolerance=1e-4, symmetric=True,
                           activation_param=None):
    """Generate a quantized MLP fixture.

    1. Create float weights using init_fn
    2. Quantize weights per-layer (affine: scale + zero_point)
    3. Simulate quantized inference in Python (matching Rust exactly)
    4. Save model parameters and expected outputs
    """
    layers = []
    state = {}

    for k in range(len(layer_sizes) - 1):
        in_size = layer_sizes[k]
        out_size = layer_sizes[k + 1]

        # Generate float weights using init_fn
        w_param = torch.empty(out_size, in_size)
        init_fn(w_param)
        w_float = w_param.flatten().tolist()

        bias = [0.01 * (j + 1) for j in range(out_size)]

        # Determine quantization parameters for weights
        w_min = min(w_float)
        w_max = max(w_float)
        w_range = max(abs(w_min), abs(w_max))

        if symmetric or w_range == 0:
            w_scale = w_range / 127.0 if w_range > 0 else 1.0
            w_zp = 0
        else:
            w_scale = (w_max - w_min) / 254.0 if w_max > w_min else 1.0
            w_zp = int(round(-128 - w_min / w_scale))
            w_zp = max(-128, min(127, w_zp))

        # Quantize weights to i8
        w_i8 = quantize_to_i8(w_float, w_scale, w_zp)

        # Input quantization params (calibrated for expected input range)
        in_scale = 1.0 / 127.0 if symmetric else 1.0 / 255.0
        in_zp = 0 if symmetric else 0

        layer_info = {
            "w_i8": w_i8,
            "bias": bias,
            "w_scale": w_scale,
            "w_zp": w_zp,
            "in_scale": in_scale,
            "in_zp": in_zp,
            "in_size": in_size,
            "out_size": out_size,
        }
        layers.append(layer_info)

        # Save to safetensors state dict
        w_key = f"{prefix}.layer_{k}.weight"
        b_key = f"{prefix}.layer_{k}.bias"
        ws_key = f"{prefix}.layer_{k}.weight_scale"
        wzp_key = f"{prefix}.layer_{k}.weight_zero_point"
        is_key = f"{prefix}.layer_{k}.input_scale"
        izp_key = f"{prefix}.layer_{k}.input_zero_point"

        state[w_key] = torch.tensor(w_i8, dtype=torch.int8).reshape(out_size, in_size)
        state[b_key] = torch.tensor(bias, dtype=torch.float32)
        state[ws_key] = torch.tensor([w_scale], dtype=torch.float32)
        state[wzp_key] = torch.tensor([w_zp], dtype=torch.int8)
        state[is_key] = torch.tensor([in_scale], dtype=torch.float32)
        state[izp_key] = torch.tensor([in_zp], dtype=torch.int8)

    save_file(state, FIXTURES_DIR / f"{name}.safetensors")

    # Run quantized inference in Python (matching Rust exactly)
    outputs = quantized_mlp_forward(layers, inputs, activation_name,
                                    activation_param)

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
    sym_str = "symmetric" if symmetric else "asymmetric"
    print(f"  {name}: {sizes_str}, {activation_name}, {sym_str}, {len(inputs)} inputs")


def generate_quantized_mlp_basic():
    generate_quantized_mlp("quantized_mlp_basic", [4, 8, 1], "relu",
                           inputs=make_inputs(5, 4, seed=1100),
                           prefix="qmlp", init_fn=init_linspace)


def generate_quantized_mlp_identity():
    generate_quantized_mlp("quantized_mlp_identity", [3, 6, 2], "identity",
                           inputs=make_inputs(5, 3, seed=1101),
                           prefix="qmlp", init_fn=init_sinusoidal)


def generate_quantized_mlp_deep():
    generate_quantized_mlp("quantized_mlp_deep", [4, 8, 8, 4, 1], "relu",
                           inputs=make_inputs(5, 4, seed=1102),
                           prefix="qmlp", init_fn=init_linspace,
                           tolerance=5e-4)


def generate_quantized_mlp_asymmetric():
    generate_quantized_mlp("quantized_mlp_asymmetric", [4, 8, 2], "relu",
                           inputs=make_inputs(5, 4, seed=1103),
                           prefix="qmlp", init_fn=init_linspace,
                           symmetric=False)


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
    generate_mlp_f32_no_bias()
    generate_mlp_f32_batchnorm()
    generate_mlp_f32_batchnorm_no_bias()
    generate_mlp_f32_layernorm()
    generate_mlp_f32_layernorm_no_bias()
    # MLP f64
    # Conv1d
    generate_conv1d()
    generate_conv1d_tanh()
    generate_conv1d_identity()
    generate_conv1d_large()
    generate_conv1d_sigmoid()
    generate_conv1d_swish()
    generate_conv1d_elu()
    generate_conv1d_leaky_relu()
    # Stacked LSTM
    generate_stacked_lstm_2layer()
    generate_stacked_lstm_3layer()
    generate_stacked_lstm_large()
    # Stacked GRU
    generate_stacked_gru_2layer()
    generate_stacked_gru_3layer()
    generate_stacked_gru_large()
    # SSM
    generate_ssm_basic()
    generate_ssm_no_skip()
    generate_ssm_multi_output()
    generate_ssm_large()
    # BNN
    generate_bnn_basic()
    generate_bnn_one_binary()
    generate_bnn_two_binary()
    generate_bnn_large()
    # TCN
    generate_tcn_basic()
    generate_tcn_residual()
    generate_tcn_identity()
    generate_tcn_large()
    # Quantized MLP
    generate_quantized_mlp_basic()
    generate_quantized_mlp_identity()
    generate_quantized_mlp_deep()
    generate_quantized_mlp_asymmetric()
    # Fuzz (seeded random configs)
    generate_fuzz()
    print("Done.")
