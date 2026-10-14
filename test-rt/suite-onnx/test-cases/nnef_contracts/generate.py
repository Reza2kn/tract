from functools import partial
from itertools import product
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference import finalize, save as write_case, np, onnx, ort, helper, numpy_helper
ROOT = Path(os.environ.get("TRACT_FIXTURE_ROOT", Path(__file__).resolve().parent))
save = partial(write_case, ROOT)

def quantized_conv_nhwc():
    inputs = {"data": np.arange(14, dtype=np.uint8).reshape(1, 7, 2) + 60}
    constants = {"kernel": (np.arange(12, dtype=np.int8).reshape(2, 2, 3) - 5), "xs": np.array(0.5, np.float32), "xz": np.array(63, np.uint8), "ws": np.array(0.25, np.float32), "wz": np.array(0, np.int8), "ys": np.array(0.125, np.float32), "yz": np.array(63, np.uint8), "bias": np.array([3, -2], np.int32)}
    nodes = [helper.make_node("Transpose", ["data"], ["nchw"], perm=[0, 2, 1]),
             helper.make_node("QLinearConv", ["nchw", "xs", "xz", "kernel", "ws", "wz", "ys", "yz", "bias"], ["conv"], pads=[1, 1]),
             helper.make_node("Transpose", ["conv"], ["output"], perm=[0, 2, 1])]
    save("quantized_conv_nhwc", inputs, nodes, constants, output_names=["output"])


if __name__ == "__main__":
    quantized_conv_nhwc()
    finalize(ROOT)
