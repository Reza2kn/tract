from functools import partial
from itertools import product
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference import finalize, save as write_case, np, onnx, ort, helper, numpy_helper
ROOT = Path(os.environ.get("TRACT_FIXTURE_ROOT", Path(__file__).resolve().parent))
save = partial(write_case, ROOT)

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


def v11_half_resize():
    constants={"start":np.array(0,np.int64),"step":np.array(1,np.int64),"axes":np.array([0,1],np.int64),"up":np.array([1,1,300],np.float32),"down":np.array([1,1,float(np.float16(1/300))],np.float32)}
    nodes=[helper.make_node("Range",["start","end","step"],["positions"]),
           helper.make_node("Cast",["positions"],["floats"],to=onnx.TensorProto.FLOAT),
           helper.make_node("Unsqueeze",["floats","axes"],["x"]),
           helper.make_node("Resize",["x","","up"],["upsampled"],mode="nearest",coordinate_transformation_mode="asymmetric",nearest_mode="floor"),
           helper.make_node("Resize",["upsampled","","down"],["output"],mode="nearest",coordinate_transformation_mode="asymmetric",nearest_mode="floor")]
    for name,end in [("symbolic_resize_half_reciprocal",3),("symbolic_resize_half_precision_guard",5000)]:
        guarded = name.endswith("precision_guard")
        final_nodes = nodes + ([helper.make_node("ReduceSum", ["output"], ["sum"], keepdims=0)] if guarded else [])
        save(name,{"end":np.array(end,np.int64)},final_nodes,constants,dynamic=True,output_names=["sum" if guarded else "output"])


def topk_symbolic():
    nodes=[helper.make_node('Range',['start','end','step'],['range']),
           helper.make_node('Unsqueeze',['range','zero_axis'],['matrix']),
           helper.make_node('Shape',['matrix'],['shape']),
           helper.make_node('Gather',['shape','one'],['lengths'],axis=0),
           helper.make_node('TopK',['lengths','one'],['top','indices']),
           helper.make_node('Cast',['top'],['output'],to=onnx.TensorProto.INT32)]
    save('topk_symbolic',{'end':np.array(7,np.int64)},nodes,{'start':np.array(0,np.int64),'step':np.array(1,np.int64),'one':np.array([1],np.int64),'zero_axis':np.array([0],np.int64)},output_names=['range','output','indices'])


def v11_lstm_and_branch():
    rng=np.random.default_rng(1101634)
    constants={"start":np.array(0,np.int64),"step":np.array(1,np.int64),"zero_axis":np.array([0],np.int64),"x_axes":np.array([1,2],np.int64),"one_axis":np.array([1],np.int64),"one":np.array([1],np.int64)}
    nodes=[helper.make_node("Range",["start","end","step"],["positions"]),
           helper.make_node("Unsqueeze",["positions","zero_axis"],["matrix"]),
           helper.make_node("Shape",["matrix"],["shape"]),
           helper.make_node("Gather",["shape","one"],["lengths"],axis=0),
           helper.make_node("TopK",["lengths","one"],["sorted_lengths","indices"]),
           helper.make_node("Cast",["sorted_lengths"],["seq_lens"],to=onnx.TensorProto.INT32),
           helper.make_node("ReduceMax",["sorted_lengths"],["limit"],keepdims=0),
           helper.make_node("Range",["start","limit","step"],["same_positions"]),
           helper.make_node("Add",["positions","same_positions"],["summed"]),
           helper.make_node("Cast",["summed"],["float_positions"],to=onnx.TensorProto.FLOAT),
           helper.make_node("Unsqueeze",["float_positions","x_axes"],["x"])]
    constants.update(w=rng.normal(0,0.1,(1,8,1)).astype(np.float32),r=rng.normal(0,0.1,(1,8,2)).astype(np.float32),qw=rng.integers(-5,6,(1,1,8),dtype=np.int8),qr=rng.integers(-5,6,(1,2,8),dtype=np.int8),ws=np.array([0.02],np.float32),rs=np.array([0.02],np.float32),wz=np.array([0],np.int8),rz=np.array([0],np.int8))
    nodes += [helper.make_node("LSTM",["x","w","r","","seq_lens"],["y","yh","yc"],hidden_size=2),
              helper.make_node("DynamicQuantizeLSTM",["x","qw","qr","","seq_lens","","","","ws","wz","rs","rz"],["qy","qyh","qyc"],domain="com.microsoft",hidden_size=2)]
    save("dynamic_lstm_symbolic_lengths",{"end":np.array(5,np.int64)},nodes,constants,dynamic=True,output_names=["y","yh","yc","qy","qyh","qyc"])


if __name__ == "__main__":
    dynamic_range()
    symbolic_slice()
    symbolic_resize_precision_guard()
    folded_gather_shape()
    symbolic_deconv()
    symbolic_cumsum()
    v11_half_resize()
    topk_symbolic()
    v11_lstm_and_branch()
    finalize(ROOT)
