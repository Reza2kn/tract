from functools import partial
from itertools import product
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference import finalize, save as write_case, np, onnx, ort, helper, numpy_helper
ROOT = Path(os.environ.get("TRACT_FIXTURE_ROOT", Path(__file__).resolve().parent))
save = partial(write_case, ROOT)

def gelu_tails():
    inputs = {"x": np.array([-12., -8., -6., -0.75251085, 0., 0.60003972, 6., 6.993848], np.float32)}
    nodes = [helper.make_node("FastGelu", ["x"], ["gelu"], domain="com.microsoft"),
             helper.make_node("DynamicQuantizeLinear", ["gelu"], ["q", "scale", "zero"])]
    save("fast_gelu_tails", inputs, nodes, {}, output_names=["gelu", "q", "scale", "zero"])
    save("fast_gelu_nonfinite", {"x": np.array([np.nan, -np.inf, np.inf, -1e30, 1e30], np.float32)},
         [helper.make_node("FastGelu", ["x"], ["gelu"], domain="com.microsoft")], {}, output_names=["gelu"])


def gemm_f16_scalars():
    save("gemm_f16_scalars",{"a":np.array([[1,2,3],[4,5,6]],np.float16)},[helper.make_node("Gemm",["a","b","c"],["output"],alpha=0.5,beta=0.75)],{"b":np.array([[0.5,1],[1.5,2],[2.5,3]],np.float16),"c":np.array([0.25,-0.25],np.float16)},output_names=["output"])


def random_like_name():
    nodes=[helper.make_node('Range',['start','end','step'],['range']),
           helper.make_node('RandomNormalLike',['range'],['normal'],name='arbitrary_exported_name',dtype=onnx.TensorProto.FLOAT),
           helper.make_node('RandomUniformLike',['range'],['uniform'],name='other_exported_name',dtype=onnx.TensorProto.FLOAT),
           helper.make_node('Shape',['normal'],['normal_shape']),
           helper.make_node('Shape',['uniform'],['uniform_shape']),
           helper.make_node('Add',['normal_shape','uniform_shape'],['sum_shape']),
           helper.make_node('Cast',['sum_shape'],['output'],to=onnx.TensorProto.INT32)]
    save('random_like_name',{'end':np.array(7,np.int64)},nodes,{'start':np.array(0,np.int64),'step':np.array(1,np.int64)},dynamic=True,output_names=['range','output'])


def v11_half_random():
    constants={"minus_one":np.array(-1,np.float32),"one":np.array(1,np.float32),"minus_two":np.array(-2,np.float32),"three":np.array(3,np.float32),"mean_tolerance":np.array(0.1,np.float32),"normal_var_lo":np.array(0.9,np.float32),"normal_var_hi":np.array(1.1,np.float32),"uniform_var_lo":np.array(1.98,np.float32),"uniform_var_hi":np.array(2.18,np.float32),"uniform_target_mean":np.array(0.5,np.float32)}
    nodes=[helper.make_node("RandomNormalLike",["x"],["normal_half"],name="normal_export",dtype=onnx.TensorProto.FLOAT16,seed=123.0),helper.make_node("RandomUniformLike",["x"],["uniform_half"],name="uniform_export",dtype=onnx.TensorProto.FLOAT16,seed=123.0,low=-2.0,high=3.0)]
    outputs=[]
    for tag in ["normal","uniform"]:
        nodes += [helper.make_node("Cast",[tag+"_half"],[tag],to=onnx.TensorProto.FLOAT),helper.make_node("ReduceMin",[tag],[tag+"_min"],keepdims=0),helper.make_node("ReduceMax",[tag],[tag+"_max"],keepdims=0),helper.make_node("ReduceMean",[tag],[tag+"_mean"],keepdims=0)]
        if tag=="normal":checks=[("Less",tag+"_min","minus_one"),("Greater",tag+"_max","one")];mean=tag+"_mean"
        else:
            checks=[("GreaterOrEqual",tag+"_min","minus_two"),("LessOrEqual",tag+"_max","three")];mean=tag+"_centered"
            nodes.append(helper.make_node("Sub",[tag+"_mean","uniform_target_mean"],[mean]))
        nodes.append(helper.make_node("Abs",[mean],[tag+"_absmean"]))
        checks.append(("Less",tag+"_absmean","mean_tolerance"))
        nodes += [helper.make_node("Sub",[tag,tag+"_mean"],[tag+"_center"]),helper.make_node("Mul",[tag+"_center",tag+"_center"],[tag+"_squared"]),helper.make_node("ReduceMean",[tag+"_squared"],[tag+"_variance"],keepdims=0)]
        checks += [("Greater",tag+"_variance",tag+"_var_lo"),("Less",tag+"_variance",tag+"_var_hi")]
        for i,(op,a,b) in enumerate(checks):
            output=tag+"_check_"+str(i);nodes.append(helper.make_node(op,[a,b],[output]));outputs.append(output)
    save("random_f16_distributions",{"x":np.zeros(16384,np.float16)},nodes,constants,output_names=outputs)


def v11_half_division():
    save("division_f16_subnormal",{"a":np.array([2**-23,2**-24,-2**-23,0,1],np.float16),"b":np.array([2**-24,2**-24,2**-24,2**-24,0.25],np.float16)},[helper.make_node("Div",["a","b"],["output"])],{},output_names=["output"])


if __name__ == "__main__":
    gelu_tails()
    gemm_f16_scalars()
    random_like_name()
    v11_half_random()
    v11_half_division()
    finalize(ROOT)
