"""Generate regressions with ONNX Runtime references and a float64 DFT for frame-20 STFT."""

from itertools import product
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper

ROOT = Path(__file__).resolve().parent


def save(name, inputs, nodes, constants, dynamic=False, output_names=None, reference=None):
    output_names = output_names or [n.output[0] for n in nodes]
    graph = helper.make_graph(
        nodes, name,
        [helper.make_tensor_value_info(k, numpy_helper.from_array(v).data_type, v.shape) for k, v in inputs.items()],
        [helper.make_empty_tensor_value_info(name) for name in output_names],
        [numpy_helper.from_array(v, k) for k, v in constants.items()],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20), helper.make_opsetid("com.microsoft", 1)], ir_version=8)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.log_severity_level = 3
    reference_model = onnx.ModelProto()
    reference_model.CopyFrom(model)
    # ORT requires explicit zero-point inputs for these reference kernels.
    # Supply equivalent explicit zeros only to the reference graph.
    dtype = next(iter(inputs.values())).dtype
    reference_model.graph.initializer.append(numpy_helper.from_array(np.array(0, dtype), "default_zero"))
    for node in reference_model.graph.node:
        if node.op_type in ["QLinearAdd", "QLinearLeakyRelu", "QLinearSigmoid"]:
            arity = 8 if node.op_type == "QLinearAdd" else 5
            while len(node.input) < arity:
                node.input.append("default_zero")
            for i in ([2, 5, 7] if arity == 8 else [2, 4]):
                if not node.input[i]:
                    node.input[i] = "default_zero"
    if reference is None:
        session = ort.InferenceSession(reference_model.SerializeToString(), options, providers=["CPUExecutionProvider"])
        reference = session.run(None, inputs)
    del model.graph.output[:]
    for output_name, value in zip(output_names, reference):
        model.graph.output.append(helper.make_tensor_value_info(output_name, numpy_helper.from_array(value).data_type, [None] * value.ndim if dynamic else value.shape))
    onnx.checker.check_model(model)
    case = ROOT / name
    data = case / "test_data_set_0"
    data.mkdir(parents=True, exist_ok=True)
    onnx.save(model, case / "model.onnx")
    for kind, tensors in [("input", list(inputs.values())), ("output", reference)]:
        for i, tensor in enumerate(tensors):
            (data / f"{kind}_{i}.pb").write_bytes(numpy_helper.from_array(tensor).SerializeToString())


def quantized():
    for dtype in [np.uint8, np.int8]:
        suffix = np.dtype(dtype).name
        zero = 128 if dtype == np.uint8 else 0
        inputs = {"x": np.arange(np.iinfo(dtype).min, np.iinfo(dtype).max + 1, dtype=dtype).reshape(16, 16), "b": np.array([zero - 4, zero + 4] * 8, dtype=dtype)}
        c = {"xs": np.array(0.5, np.float32), "xz": np.array(zero, dtype), "bs": np.array(0.25, np.float32), "bz": np.array(zero, dtype), "ys": np.array(1., np.float32), "yz": np.array(zero, dtype)}
        nodes = []
        for present_x, present_b, present_y in product([False, True], repeat=3):
            for op in ["QLinearAdd", "QLinearLeakyRelu", "QLinearSigmoid"]:
                if op != "QLinearAdd" and present_b:
                    continue
                name = f"{op}_{int(present_x)}{int(present_b)}{int(present_y)}"
                args = ["x", "xs", "xz" if present_x else ""]
                if op == "QLinearAdd":
                    args += ["b", "bs", "bz" if present_b else ""]
                args += ["ys"]
                if present_y:
                    args += ["yz"]
                nodes.append(helper.make_node(op, args, [name], domain="com.microsoft", name=name, **({"alpha": 0.25} if op == "QLinearLeakyRelu" else {})))
        nodes.append(helper.make_node("QLinearConcat", ["ys", "yz", "x", "xs", "xz", "x", "bs", "bz"], ["concat"], domain="com.microsoft", axis=-1))
        c["sigmoid_scale"] = np.array(1 / 256, np.float32)
        c["sigmoid_zero"] = np.array(0 if dtype == np.uint8 else -128, dtype)
        c["clip_scale"] = np.array(0.125, np.float32)
        nodes.append(helper.make_node("QLinearSigmoid", ["x", "xs", "xz", "sigmoid_scale", "sigmoid_zero"], ["sigmoid_range"], domain="com.microsoft"))
        nodes.append(helper.make_node("QLinearLeakyRelu", ["x", "xs", "xz", "clip_scale", "yz"], ["leaky_default"], domain="com.microsoft"))
        inputs["xs"] = c.pop("xs")
        save(f"quantized_{suffix}", inputs, nodes, c)


def matmul():
    inputs, nodes = {}, []
    for ta, tb, ba, bb in product([False, True], repeat=4):
        name = f"mm_{int(ta)}{int(tb)}{int(ba)}{int(bb)}"
        a = (np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5) % 11 - 5) / 4
        b = (np.arange(2 * 3 * 5 * 6, dtype=np.float32).reshape(2, 3, 5, 6) % 13 - 6) / 4
        if ta: a = a.swapaxes(-1, -2)
        if tb: b = b.swapaxes(-1, -2)
        if ba: a = np.moveaxis(a, -2, 0)
        if bb: b = np.moveaxis(b, -2, 0)
        inputs[name + "a"], inputs[name + "b"] = a.copy(), b.copy()
        nodes.append(helper.make_node("FusedMatMul", [name + "a", name + "b"], [name], domain="com.microsoft", transA=int(ta), transB=int(tb), transBatchA=int(ba), transBatchB=int(bb), alpha=0.5))
    for name, a, b in [
        ("dot", np.arange(4, dtype=np.float32), np.arange(4, dtype=np.float32)),
        ("vector_matrix", np.arange(4, dtype=np.float32), np.arange(24, dtype=np.float32).reshape(2, 4, 3)),
        ("matrix_vector", np.arange(24, dtype=np.float32).reshape(2, 3, 4), np.arange(4, dtype=np.float32)),
    ]:
        inputs[name + "a"], inputs[name + "b"] = a, b
        nodes.append(helper.make_node("FusedMatMul", [name + "a", name + "b"], [name], domain="com.microsoft"))
    save("fused_matmul", inputs, nodes, {})


def stft():
    x = np.arange(32, dtype=np.float32).reshape(2, 16) / 16
    inputs = {"x": x, "real": x[..., None], "complex": np.stack([x, -x], axis=-1)}
    c = {"step": np.array(3, np.int64), "length": np.array(8, np.int64), "window": np.array([0., 0.25, 0.5, 0.75, 1., 0.75, 0.5, 0.25], np.float32)}
    nodes = []
    for key in inputs:
        for window in [False, True]:
            name = f"stft_{key}_{int(window)}"
            nodes.append(helper.make_node("STFT", [key, "step", "window" if window else "", "length"], [name], name=name, onesided=int(key != "complex")))
    save("stft_real_rank2", inputs, nodes, c)


def stft_frame20():
    rng = np.random.default_rng(20)
    x = rng.uniform(-0.125, 0.125, (2, 53)).astype(np.float32)
    inputs = {"x": x, "real": x[..., None], "complex": np.stack([x, -x / 2], axis=-1)}
    constants = {"step": np.array(5, np.int64), "length": np.array(20, np.int64),
                 "window": np.hanning(20).astype(np.float32)}
    nodes = []
    for key in inputs:
        for window in ["", "window"]:
            name = f"stft20_{key}_{window or 'unwindowed'}"
            nodes.append(helper.make_node("STFT", [key, "step", window, "length"], [name], name=name, onesided=int(key != "complex")))
    references = []
    for node in nodes:
        value = inputs[node.input[0]]
        if value.ndim == 2:
            value = value[..., None]
        frames = np.stack([value[:, i:i + 20] for i in range(0, 34, 5)], axis=1)
        window = constants[node.input[2]] if node.input[2] else np.ones(20, np.float32)
        frames = frames * window[None, None, :, None]
        signal = frames[..., 0].astype(np.complex128)
        if frames.shape[-1] == 2:
            signal += 1j * frames[..., 1]
        spectrum = np.fft.fft(signal, axis=-1)
        if value.shape[-1] == 1:
            spectrum = spectrum[..., :11]
        reference = np.stack([spectrum.real, spectrum.imag], axis=-1).astype(np.float32)
        references.append(reference)
    save("stft_frame20", inputs, nodes, constants, reference=references)


def quantized_matmul():
    inputs = {"x": np.array([[124, 130, 125, 136]], np.uint8)}
    zero = np.array([-3, 0, 2, -1, 1, 3], np.int8)
    c = {
        "xs": np.array(0.5, np.float32), "xz": np.array(128, np.uint8),
        "w": ((np.arange(24).reshape(4, 6) % 9 - 4) + zero).astype(np.int8),
        "ws": np.array([0.125, 0.25, 0.5, 1, 0.25, 0.5], np.float32), "wz": zero,
        "ys": np.array(0.25, np.float32), "yz": np.array(127, np.uint8),
        "axes": np.array([2], np.int64),
    }
    nodes = [
        helper.make_node("QLinearMatMul", ["x", "xs", "xz", "w", "ws", "wz", "ys", "yz"], ["mm"]),
        helper.make_node("Unsqueeze", ["mm", "axes"], ["expanded"]),
    ]
    save("quantized_matmul_axes", inputs, nodes, c, output_names=["expanded"])


def symbolic_slice():
    inputs = {"limit": np.array(31, np.int64)}
    c = {"start": np.array(0, np.int64), "step": np.array(1, np.int64), "zero": np.array([0], np.int64), "one": np.array([1], np.int64), "axis": np.array([0], np.int64), "data": np.arange(20, dtype=np.float32), "length": np.array([20], np.int64)}
    nodes = [
        helper.make_node("Range", ["start", "limit", "step"], ["range"]),
        helper.make_node("Shape", ["range"], ["end"]),
        helper.make_node("Add", ["end", "length"], ["larger"]),
        helper.make_node("Slice", ["data", "zero", "larger", "axis", "one"], ["clipped"]),
        helper.make_node("Slice", ["data", "larger", "end", "axis", "one"], ["empty"]),
    ]
    save("symbolic_slice_bounds", inputs, nodes, c, dynamic=True, output_names=["clipped", "empty"])


def symbolic_resize_precision_guard():
    inputs = {"limit": np.array(16777499, np.int64)}
    c = {"start": np.array(0, np.int64), "step": np.array(1, np.int64), "scale": np.array([1 / 300], np.float32)}
    nodes = [
        helper.make_node("Range", ["start", "limit", "step"], ["r"]),
        helper.make_node("Cast", ["r"], ["x"], to=onnx.TensorProto.FLOAT),
        helper.make_node("Resize", ["x", "", "scale"], ["resized"], mode="nearest", coordinate_transformation_mode="asymmetric", nearest_mode="floor"),
        helper.make_node("ReduceSum", ["resized"], ["sum"], keepdims=0),
    ]
    save("symbolic_resize_precision_guard", inputs, nodes, c, dynamic=True, output_names=["sum"])


def dynamic_range():
    inputs = {"limit": np.array(5, np.int64)}
    c = {
        "start": np.array(0, np.int64), "step": np.array(1, np.int64),
        "axes": np.array([0, 1], np.int64),
        "up": np.array([1, 1, 600], np.float32),
        "down": np.array([1, 1, 1 / 300], np.float32),
        "weight": np.array([[[0, 1, 0]]], np.float32),
    }
    nodes = [
        helper.make_node("Range", ["start", "limit", "step"], ["r"]),
        helper.make_node("Cast", ["r"], ["f"], to=onnx.TensorProto.FLOAT),
        helper.make_node("Unsqueeze", ["f", "axes"], ["x"]),
        helper.make_node("Resize", ["x", "", "up"], ["u"], mode="nearest", coordinate_transformation_mode="asymmetric", nearest_mode="floor"),
        helper.make_node("Resize", ["u", "", "down"], ["d"], mode="nearest", coordinate_transformation_mode="asymmetric", nearest_mode="floor"),
        helper.make_node("Conv", ["d", "weight"], ["half"], pads=[1, 1], strides=[2]),
        helper.make_node("Concat", ["x", "half"], ["result"], axis=1),
    ]
    save("dynamic_range_resize", inputs, nodes, c, dynamic=True)


def dynamic_lstm():
    rng = np.random.default_rng(1634)
    for dtype in [np.int8, np.uint8]:
        inputs, nodes, constants, outputs = {}, [], {}, []
        for direction, channels, bias, initial in product(["forward", "reverse", "bidirectional"], [False, True], [False, True], [False, True]):
            tag = f"lstm_{direction}_{int(channels)}{int(bias)}{int(initial)}"
            dirs = 2 if direction == "bidirectional" else 1
            h, k, batch, seq = 3, 5, 2, 4
            x = rng.normal(size=(seq, batch, k)).astype(np.float32)
            x[0] *= 0.02
            inputs[tag + "x"] = x
            params = {"w": rng.integers(-60, 61, (dirs, k, 4*h)), "r": rng.integers(-60, 61, (dirs, h, 4*h))}
            qshape = (dirs, 4*h) if channels else (dirs,)
            for key in ["w", "r"]:
                zp = 0 if dtype == np.int8 else 110
                params[key] = (params[key] + zp).astype(dtype)
                params[key+"s"] = rng.uniform(0.001, 0.008, qshape).astype(np.float32)
                params[key+"z"] = np.full(qshape, zp, dtype)
            args = [tag+"x", tag+"w", tag+"r", "", "", "", "", "", tag+"ws", tag+"wz", tag+"rs", tag+"rz"]
            if bias:
                params["b"] = rng.normal(0, 0.2, (dirs, 8*h)).astype(np.float32); args[3] = tag+"b"
            if initial:
                params["ih"] = rng.normal(0, 0.2, (dirs, batch, h)).astype(np.float32)
                params["ic"] = rng.normal(0, 0.2, (dirs, batch, h)).astype(np.float32)
                args[5:7] = [tag+"ih", tag+"ic"]
            constants.update({tag+key: value for key, value in params.items()})
            node_outputs = [tag+"y", tag+"yh", tag+"yc"]
            if initial and not bias:
                node_outputs[0] = ""
            nodes.append(helper.make_node("DynamicQuantizeLSTM", args, node_outputs, name=tag, domain="com.microsoft", direction=direction, hidden_size=h))
            outputs.extend(x for x in node_outputs if x)
        save("dynamic_lstm_"+np.dtype(dtype).name, inputs, nodes, constants, output_names=outputs)
    inputs = {"x": np.zeros((2, 1, 3), np.float32)}
    constants = {"w": np.zeros((1, 3, 8), np.int8), "r": np.zeros((1, 2, 8), np.int8), "ws": np.ones(1, np.float32), "rs": np.ones(1, np.float32), "wz": np.zeros(1, np.int8), "rz": np.zeros(1, np.int8)}
    save("dynamic_lstm_zero", inputs, [helper.make_node("DynamicQuantizeLSTM", ["x", "w", "r", "", "", "", "", "", "ws", "wz", "rs", "rz"], ["y", "yh", "yc"], domain="com.microsoft", hidden_size=2)], constants, output_names=["y", "yh", "yc"])


def nonfinite_quantization():
    inputs = {"x": np.array([[np.nan, np.inf, -np.inf, 1e30, -1e30, 0., 0.25, 0.75, -0.25, -0.75]], np.float32)}
    constants = {"scale": np.array(0.5, np.float32), "scales": np.array([0.5] * 10, np.float32)}
    nodes = []
    for dtype, zero in [(np.uint8, 63), (np.int8, 1), (np.uint8, 0), (np.int8, -1)]:
        tag = np.dtype(dtype).name + "_" + str(zero)
        constants[tag] = np.array(zero, dtype)
        constants[tag+"s"] = np.full(10, zero, dtype)
        nodes.append(helper.make_node("QuantizeLinear", ["x", "scale", tag], [tag+"y"]))
        nodes.append(helper.make_node("QuantizeLinear", ["x", "scales", tag+"s"], [tag+"axis"], axis=1))
    save("nonfinite_quantization", inputs, nodes, constants)


def phase_zero():
    inputs = {"imag": np.array([0., -0., 0., 1., -1., 1., -1., 0., 0.], np.float32), "real": np.array([0., 0., 1., 1., 1., -1., -1., -1., -0.], np.float32)}
    constants = {"zero": np.array(0., np.float32), "pi": np.array(np.pi, np.float32)}
    nodes = [
        helper.make_node("Div", ["imag", "real"], ["ratio"]),
        helper.make_node("Atan", ["ratio"], ["angle"]),
        helper.make_node("Greater", ["imag", "zero"], ["positive_imaginary"]),
        helper.make_node("Less", ["real", "zero"], ["negative_real"]),
        helper.make_node("Add", ["angle", "pi"], ["upper"]),
        helper.make_node("Sub", ["angle", "pi"], ["lower"]),
        helper.make_node("Where", ["positive_imaginary", "upper", "lower"], ["negative_phase"]),
        helper.make_node("Where", ["negative_real", "negative_phase", "angle"], ["phase"], name="/decoder/decoder/generator/Where_1"),
    ]
    save("kokoro_phase_zero", inputs, nodes, constants, output_names=["phase"])
    output = ROOT / "kokoro_phase_zero/test_data_set_0/output_0.pb"
    tensor = onnx.TensorProto.FromString(output.read_bytes())
    values = numpy_helper.to_array(tensor).copy()
    values[(inputs["imag"] == 0) & (inputs["real"] == 0)] = 0
    output.write_bytes(numpy_helper.from_array(values).SerializeToString())


def dynamic_quantize_zero():
    save("dynamic_quantize_zero", {"x": np.zeros((2, 3), np.float32)},
         [helper.make_node("DynamicQuantizeLinear", ["x"], ["y", "scale", "zero"])], {},
         output_names=["y", "scale", "zero"])


def gelu_tails():
    inputs = {"x": np.array([-12., -8., -6., -0.75251085, 0., 0.60003972, 6., 6.993848], np.float32)}
    nodes = [helper.make_node("FastGelu", ["x"], ["gelu"], domain="com.microsoft"),
             helper.make_node("DynamicQuantizeLinear", ["gelu"], ["q", "scale", "zero"])]
    save("fast_gelu_tails", inputs, nodes, {}, output_names=["gelu", "q", "scale", "zero"])
    save("fast_gelu_nonfinite", {"x": np.array([np.nan, -np.inf, np.inf, -1e30, 1e30], np.float32)},
         [helper.make_node("FastGelu", ["x"], ["gelu"], domain="com.microsoft")], {}, output_names=["gelu"])


def folded_gather_shape():
    for empty in [False, True]:
        inputs = {"data": np.array([[3., 5., 7., 11.]], np.float32)}
        constants = {"mask": np.array([False, not empty, not empty, False]), "axis": np.array([0], np.int64), "axis1": np.array([1], np.int64), "axes": np.array([1, 2], np.int64), "ones": np.array([1, 1], np.int64), "zero": np.array([[[0]]], np.int64)}
        nodes = [
            helper.make_node("NonZero", ["mask"], ["nz"]),
            helper.make_node("Squeeze", ["nz", "axis"], ["indices"]),
            helper.make_node("Gather", ["data", "indices"], ["values"], axis=1),
            helper.make_node("Shape", ["indices"], ["count"]),
            helper.make_node("Concat", ["count", "ones"], ["shape"], axis=0),
            helper.make_node("Expand", ["zero", "shape"], ["zeros"]),
            helper.make_node("Unsqueeze", ["indices", "axes"], ["column"]),
            helper.make_node("Expand", ["column", "shape"], ["expanded"]),
            helper.make_node("Concat", ["zeros", "expanded"], ["grid"], axis=2),
        ]
        save("folded_gather_shape" + ("_empty" if empty else ""), inputs, nodes, constants, dynamic=True, output_names=["values", "grid"])


def dynamic_quantize_nan():
    save("dynamic_quantize_nan", {"x": np.array([-3., 2., np.nan], np.float32)},
         [helper.make_node("DynamicQuantizeLinear", ["x"], ["y", "scale", "zero"])], {}, output_names=["y", "scale", "zero"])


def dynamic_quantize_batch():
    rng = np.random.default_rng(255)
    x = rng.uniform(-3, 5, (2053, 3)).astype(np.float32)
    nodes = [helper.make_node("DynamicQuantizeLinear", ["x"], ["y", "scale", "zero"])]
    save("dynamic_quantize_batch", {"x": x}, nodes, {}, output_names=["y", "scale", "zero"])
    x = np.empty((0, 3), np.float32)
    save("dynamic_quantize_empty", {"x": x}, nodes, {}, output_names=["y", "scale", "zero"],
         reference=[np.empty((0, 3), np.uint8), np.array(1, np.float32), np.array(0, np.uint8)])


def integer_matmul_batches():
    rng = np.random.default_rng(32)
    inputs, nodes = {}, []
    for adt, bdt in product([np.int8, np.uint8], repeat=2):
        name = np.dtype(adt).name + "_" + np.dtype(bdt).name
        inputs[name + "a"] = rng.integers(np.iinfo(adt).min, np.iinfo(adt).max + 1, (17, 33), dtype=adt)
        inputs[name + "b"] = rng.integers(np.iinfo(bdt).min, np.iinfo(bdt).max + 1, (33, 19), dtype=bdt)
        inputs[name + "az"] = np.arange(17, dtype=adt)
        inputs[name + "bz"] = np.arange(19, dtype=bdt)
        for az, bz in product([False, True], repeat=2):
            output = name + str(int(az)) + str(int(bz))
            args = [name + "a", name + "b", name + "az" if az else "", name + "bz" if bz else ""]
            nodes.append(helper.make_node("MatMulInteger", args, [output]))
    inputs["batcha"] = rng.integers(0, 256, (2, 3, 17, 33), dtype=np.uint8)
    inputs["batchb"] = rng.integers(-128, 128, (1, 3, 33, 19), dtype=np.int8)
    nodes.append(helper.make_node("MatMulInteger", ["batcha", "batchb"], ["batch_output"]))
    inputs["largea"] = np.full((1, 1025), 255, np.uint8)
    inputs["largeb"] = np.full((1025, 1), 255, np.uint8)
    nodes.append(helper.make_node("MatMulInteger", ["largea", "largeb"], ["large_output"]))
    references = []
    for node in nodes:
        a, b = [inputs[name].astype(np.int64) for name in node.input[:2]]
        if len(node.input) > 2 and node.input[2]:
            a = a - inputs[node.input[2]].astype(np.int64)[:, None]
        if len(node.input) > 3 and node.input[3]:
            b = b - inputs[node.input[3]].astype(np.int64)
        references.append(np.matmul(a, b).astype(np.int32))
    save("integer_matmul_batches", inputs, nodes, {}, reference=references)
    save("integer_matmul_constant", {"offset": np.array(0, np.int32)}, [
        helper.make_node("MatMulInteger", ["a", "b"], ["product"]),
        helper.make_node("Add", ["product", "offset"], ["output"]),
    ], {"a": inputs["largea"], "b": inputs["largeb"]}, output_names=["output"],
         reference=[np.array([[65025 * 1025]], np.int32)])


def symbolic_deconv():
    inputs = {"end": np.array(5, np.int64)}
    constants = {"start": np.array(0, np.int64), "step": np.array(1, np.int64), "axes": np.array([0, 1], np.int64), "kernel": np.array([[[1., 2., 3.]]], np.float32), "bias": np.array([0.5], np.float32)}
    nodes = [helper.make_node("Range", ["start", "end", "step"], ["range"]),
             helper.make_node("Cast", ["range"], ["floats"], to=onnx.TensorProto.FLOAT),
             helper.make_node("Unsqueeze", ["floats", "axes"], ["data"]),
             helper.make_node("ConvTranspose", ["data", "kernel", "bias"], ["output"], strides=[2], pads=[1, 1], output_padding=[1])]
    save("symbolic_deconv", inputs, nodes, constants, dynamic=True, output_names=["output"])


def symbolic_cumsum():
    inputs = {"end": np.array(5, np.int64), "data": np.array([2.], np.float32)}
    constants = {"start": np.array(0, np.int64), "step": np.array(1, np.int64), "axis": np.array(0, np.int64)}
    nodes = [helper.make_node("Range", ["start", "end", "step"], ["positions"]),
             helper.make_node("Shape", ["positions"], ["shape"]),
             helper.make_node("CumSum", ["shape", "axis"], ["sums"]),
             helper.make_node("Cast", ["sums"], ["floats"], to=onnx.TensorProto.FLOAT),
             helper.make_node("Mul", ["data", "floats"], ["scaled"]),
             helper.make_node("ReduceSum", ["positions"], ["total"], keepdims=0),
             helper.make_node("Cast", ["total"], ["float_total"], to=onnx.TensorProto.FLOAT),
             helper.make_node("Add", ["scaled", "float_total"], ["output"])]
    save("symbolic_cumsum", inputs, nodes, constants, dynamic=True, output_names=["output"])


def quantized_conv_nhwc():
    inputs = {"data": np.arange(14, dtype=np.uint8).reshape(1, 7, 2) + 60}
    constants = {"kernel": (np.arange(12, dtype=np.int8).reshape(2, 2, 3) - 5), "xs": np.array(0.5, np.float32), "xz": np.array(63, np.uint8), "ws": np.array(0.25, np.float32), "wz": np.array(0, np.int8), "ys": np.array(0.125, np.float32), "yz": np.array(63, np.uint8), "bias": np.array([3, -2], np.int32)}
    nodes = [helper.make_node("Transpose", ["data"], ["nchw"], perm=[0, 2, 1]),
             helper.make_node("QLinearConv", ["nchw", "xs", "xz", "kernel", "ws", "wz", "ys", "yz", "bias"], ["conv"], pads=[1, 1]),
             helper.make_node("Transpose", ["conv"], ["output"], perm=[0, 2, 1])]
    save("quantized_conv_nhwc", inputs, nodes, constants, output_names=["output"])


if __name__ == "__main__":
    quantized()
    matmul()
    stft()
    stft_frame20()
    dynamic_range()
    quantized_matmul()
    symbolic_slice()
    symbolic_resize_precision_guard()

    dynamic_lstm()

    nonfinite_quantization()

    phase_zero()

    dynamic_quantize_zero()

    gelu_tails()
    folded_gather_shape()
    dynamic_quantize_nan()
    dynamic_quantize_batch()
    integer_matmul_batches()
    symbolic_deconv()
    symbolic_cumsum()
    quantized_conv_nhwc()
