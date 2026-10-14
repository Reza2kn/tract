from functools import partial
from itertools import product
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference import finalize, save as write_case, np, onnx, ort, helper, numpy_helper
ROOT = Path(os.environ.get("TRACT_FIXTURE_ROOT", Path(__file__).resolve().parent))
save = partial(write_case, ROOT)

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


def v11_half_precision():
    source=ROOT/"kokoro_phase_zero"
    model=onnx.load(source/"model.onnx")
    for node in model.graph.node:
        if node.name=="/decoder/decoder/generator/Where_1":node.name="/decoder/generator/Where_1"
    for info in [*model.graph.input,*model.graph.output]:info.type.tensor_type.elem_type=onnx.TensorProto.FLOAT16
    for i,tensor in enumerate(model.graph.initializer):
        replacement=numpy_helper.from_array(numpy_helper.to_array(tensor).astype(np.float16),tensor.name)
        model.graph.initializer[i].CopyFrom(replacement)
    case=ROOT/"kokoro_phase_zero_f16";data=case/"test_data_set_0";data.mkdir(parents=True,exist_ok=True);onnx.save(model,case/"model.onnx")
    for f in (source/"test_data_set_0").glob("*.pb"):
        tensor=onnx.TensorProto.FromString(f.read_bytes());values=numpy_helper.to_array(tensor).astype(np.float16)
        (data/f.name).write_bytes(numpy_helper.from_array(values).SerializeToString())


def v11_half_sine_phase():
    prefix="/decoder/generator/m_source/l_sin_gen"
    source=ROOT/"kokoro_phase_zero_f16"
    model=onnx.load(source/"model.onnx")
    inputs={}
    for i,info in enumerate(model.graph.input):inputs[info.name]=numpy_helper.to_array(onnx.TensorProto.FromString((source/f"test_data_set_0/input_{i}.pb").read_bytes())).copy()
    inputs["phase_steps"]=np.full((1,1,100),0.75,np.float32)
    constants={t.name:numpy_helper.to_array(t).copy() for t in model.graph.initializer}
    constants.update(phase_axis=np.array(2,np.int64),phase_scale=np.array(1884,np.float16),phase_up=np.array([1,1,300],np.float32),mean_limit=np.array(0.1,np.float32))
    nodes=list(model.graph.node)+[
        helper.make_node("CumSum",["phase_steps","phase_axis"],["phase_sum"],name=prefix+"/CumSum"),
        helper.make_node("Cast",["phase_sum"],["half_phase"],name=prefix+"/CumSum_output_0_cast_to_fp16_node",to=onnx.TensorProto.FLOAT16),
        helper.make_node("Mul",["half_phase","phase_scale"],["large_phase"],name=prefix+"/Mul_9"),
        helper.make_node("Resize",["large_phase","","phase_up"],["upsampled_phase"],name=prefix+"/Resize_1",mode="linear",coordinate_transformation_mode="half_pixel"),
        helper.make_node("Sin",["upsampled_phase"],["sine"],name=prefix+"/Sin"),
        helper.make_node("IsNaN",["sine"],["nan_mask"]),helper.make_node("Not",["nan_mask"],["finite"]),
        helper.make_node("Cast",["sine"],["sine_f32"],to=onnx.TensorProto.FLOAT),helper.make_node("Abs",["sine_f32"],["abs_sine"]),helper.make_node("ReduceMean",["abs_sine"],["mean"],keepdims=0),helper.make_node("Greater",["mean","mean_limit"],["signal"])]
    phase=numpy_helper.to_array(onnx.TensorProto.FromString((source/"test_data_set_0/output_0.pb").read_bytes())).copy()
    save("kokoro_phase_zero_sine_f16",inputs,nodes,constants,output_names=["phase","finite","signal"],reference=[phase,np.ones((1,1,30000),bool),np.array(True)])


if __name__ == "__main__":
    phase_zero()
    v11_half_precision()
    v11_half_sine_phase()
    finalize(ROOT)
