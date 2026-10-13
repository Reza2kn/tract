"""Generate checked ONNX fixtures with outputs from ONNX's reference evaluator."""

from itertools import product
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator


ROOT = Path(__file__).resolve().parent
TYPES = {"i8": (np.int8, TensorProto.INT8, -3), "u8": (np.uint8, TensorProto.UINT8, 127)}


def save(name, inputs, nodes, initializers, outputs):
    graph = helper.make_graph(
        nodes, name,
        [helper.make_tensor_value_info(k, numpy_helper.from_array(v).data_type, v.shape) for k, v in inputs.items()],
        outputs, [numpy_helper.from_array(v, k) for k, v in initializers.items()],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    onnx.checker.check_model(model)
    reference = ReferenceEvaluator(model).run(None, inputs)
    case = ROOT / name
    data = case / "test_data_set_0"
    data.mkdir(parents=True, exist_ok=True)
    onnx.save(model, case / "model.onnx")
    for kind, tensors in [("input", list(inputs.values())), ("output", reference)]:
        for i, tensor in enumerate(tensors):
            (data / f"{kind}_{i}.pb").write_bytes(numpy_helper.from_array(tensor).SerializeToString())


def operands(channels=2, filters=3, group=1):
    inputs, initializers = {}, {}
    delta = (np.arange(channels * 5) % 9 - 4).reshape(1, channels, 5) * 8
    weights = (np.arange(filters * (channels // group) * 2) % 7 - 3).reshape(filters, channels // group, 2)
    for name, (dtype, _, zero) in TYPES.items():
        inputs[f"x_{name}"] = (delta + zero).astype(dtype)
        initializers[f"xz_{name}"] = np.array(zero, dtype=dtype)
        initializers[f"w_{name}"] = (weights + zero).astype(dtype)
        initializers[f"wz_{name}"] = np.array(zero, dtype=dtype)
        initializers[f"yz_{name}"] = np.array(zero, dtype=dtype)
    initializers.update(xs=np.array(0.25, np.float32), ws=np.array(0.5, np.float32), ys=np.array(0.125, np.float32))
    return inputs, initializers


def qlinear_types():
    inputs, initializers = operands()
    initializers["bias"] = np.array([-7, 13, 3], np.int32)
    nodes, outputs = [], []
    for x, w, y in product(TYPES, repeat=3):
        name = f"y_{x}_{w}_{y}"
        args = [f"x_{x}", "xs", f"xz_{x}", f"w_{w}", "ws", f"wz_{w}", "ys", f"yz_{y}", "bias"]
        nodes.append(helper.make_node("QLinearConv", args, [name], name=name))
        outputs.append(helper.make_tensor_value_info(name, TYPES[y][1], [None, None, None]))
    save("qlinear_types", inputs, nodes, initializers, outputs)


def qlinear_grouped():
    inputs, initializers = operands(channels=4, filters=4, group=2)
    inputs = {"x_u8": np.expand_dims(inputs["x_u8"], 2)}
    initializers["w_i8"] = np.expand_dims(initializers["w_i8"], 2)
    initializers["ws"] = np.array([0.125, 0.25, 0.5, 1.0], np.float32)
    initializers["wz_i8"] = np.array([-3, -2, -1, 0], np.int8)
    nodes, outputs = [], []
    for y in TYPES:
        name = f"y_{y}"
        args = ["x_u8", "xs", "xz_u8", "w_i8", "ws", "wz_i8", "ys", f"yz_{y}"]
        nodes.append(helper.make_node("QLinearConv", args, [name], name=name, group=2, pads=[0, 1, 0, 1]))
        outputs.append(helper.make_tensor_value_info(name, TYPES[y][1], [None, None, None, None]))
    save("qlinear_grouped", inputs, nodes, initializers, outputs)


def integer_defaults():
    inputs, initializers = operands()
    nodes, outputs = [], []
    for x, w in product(TYPES, repeat=2):
        for xz, wz in product([False, True], repeat=2):
            name = f"y_{x}_{w}_{int(xz)}_{int(wz)}"
            args = [f"x_{x}", f"w_{w}"]
            if xz or wz:
                args.append(f"xz_{x}" if xz else "")
            if wz:
                args.append(f"wz_{w}")
            nodes.append(helper.make_node("ConvInteger", args, [name], name=name))
            outputs.append(helper.make_tensor_value_info(name, TensorProto.INT32, [None, None, None]))
    save("integer_defaults", inputs, nodes, initializers, outputs)


if __name__ == "__main__":
    qlinear_types()
    qlinear_grouped()
    integer_defaults()
