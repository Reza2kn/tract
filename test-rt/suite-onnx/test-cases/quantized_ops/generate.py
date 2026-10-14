from functools import partial
from itertools import product
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference import finalize, save as write_case, np, onnx, ort, helper, numpy_helper
ROOT = Path(os.environ.get("TRACT_FIXTURE_ROOT", Path(__file__).resolve().parent))
save = partial(write_case, ROOT)

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


def integer_matmul_constant():
    save("integer_matmul_constant", {"offset": np.array(0, np.int32)}, [
        helper.make_node("MatMulInteger", ["a", "b"], ["product"]),
        helper.make_node("Add", ["product", "offset"], ["output"]),
    ], {"a": np.full((1, 1025), 255, np.uint8), "b": np.full((1025, 1), 255, np.uint8)},
         output_names=["output"], reference=[np.array([[65025 * 1025]], np.int32)])


if __name__ == "__main__":
    quantized_matmul()
    nonfinite_quantization()
    integer_matmul_constant()
    finalize(ROOT)
