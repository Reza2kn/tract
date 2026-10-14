from functools import partial
from itertools import product
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference import finalize, save as write_case, np, onnx, ort, helper, numpy_helper
ROOT = Path(os.environ.get("TRACT_FIXTURE_ROOT", Path(__file__).resolve().parent))
save = partial(write_case, ROOT)

def dynamic_lstm():
    inputs = {"x": np.zeros((2, 1, 3), np.float32)}
    constants = {"w": np.zeros((1, 3, 8), np.int8), "r": np.zeros((1, 2, 8), np.int8), "ws": np.ones(1, np.float32), "rs": np.ones(1, np.float32), "wz": np.zeros(1, np.int8), "rz": np.zeros(1, np.int8)}
    save("dynamic_lstm_zero", inputs, [helper.make_node("DynamicQuantizeLSTM", ["x", "w", "r", "", "", "", "", "", "ws", "wz", "rs", "rz"], ["y", "yh", "yc"], domain="com.microsoft", hidden_size=2)], constants, output_names=["y", "yh", "yc"])


def dynamic_quantize_zero():
    save("dynamic_quantize_zero", {"x": np.zeros((2, 3), np.float32)},
         [helper.make_node("DynamicQuantizeLinear", ["x"], ["y", "scale", "zero"])], {},
         output_names=["y", "scale", "zero"])


def dynamic_quantize_nan():
    save("dynamic_quantize_nan", {"x": np.array([-3., 2., np.nan], np.float32)},
         [helper.make_node("DynamicQuantizeLinear", ["x"], ["y", "scale", "zero"])], {}, output_names=["y", "scale", "zero"])


def dynamic_quantize_batch():
    nodes = [helper.make_node("DynamicQuantizeLinear", ["x"], ["y", "scale", "zero"])]
    x = np.empty((0, 3), np.float32)
    save("dynamic_quantize_empty", {"x": x}, nodes, {}, output_names=["y", "scale", "zero"],
         reference=[np.empty((0, 3), np.uint8), np.array(1, np.float32), np.array(0, np.uint8)])


if __name__ == "__main__":
    dynamic_lstm()
    dynamic_quantize_zero()
    dynamic_quantize_nan()
    dynamic_quantize_batch()
    finalize(ROOT)
