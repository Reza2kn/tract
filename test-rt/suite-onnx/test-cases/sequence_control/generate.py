from functools import partial
from itertools import product
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference import finalize, save as write_case, np, onnx, ort, helper, numpy_helper
ROOT = Path(os.environ.get("TRACT_FIXTURE_ROOT", Path(__file__).resolve().parent))
save = partial(write_case, ROOT)

def v11_sequence_and_shapes():
    body_nodes = [
        helper.make_node('SequenceAt', ['count_parts', 'iteration'], ['count']),
        helper.make_node('SequenceAt', ['value_parts', 'iteration'], ['value']),
        helper.make_node('Constant', [], ['one_axis'], value=numpy_helper.from_array(np.array([1], np.int64))),
        helper.make_node('Unsqueeze', ['value', 'one_axis'], ['value_matrix']),
        helper.make_node('Constant', [], ['one'], value=numpy_helper.from_array(np.array([1], np.int64))),
        helper.make_node('Constant', [], ['empty'], value=numpy_helper.from_array(np.array([], np.int64))),
        helper.make_node('Concat', ['one', 'count', 'empty'], ['target'], axis=0),
        helper.make_node('Shape', ['target'], ['target_shape']),
        helper.make_node('ConstantOfShape', ['target_shape'], ['ones'], value=numpy_helper.from_array(np.array([1], np.int64))),
        helper.make_node('Constant', [], ['minus_one'], value=numpy_helper.from_array(np.array(-1, np.int64))),
        helper.make_node('Mul', ['ones', 'minus_one'], ['minus_ones']),
        helper.make_node('Equal', ['target', 'minus_ones'], ['infer_dimension']),
        helper.make_node('Where', ['infer_dimension', 'ones', 'target'], ['resolved_target']),
        helper.make_node('Expand', ['value_matrix', 'resolved_target'], ['repeated_matrix']),
        helper.make_node('Constant', [], ['flatten'], value=numpy_helper.from_array(np.array([-1], np.int64))),
        helper.make_node('Reshape', ['repeated_matrix', 'flatten'], ['repeated']),
        helper.make_node('SequenceInsert', ['carried', 'repeated'], ['inserted']),
        helper.make_node('Cast', ['condition'], ['next_condition'], to=onnx.TensorProto.BOOL),
    ]
    for name, dtype, values, counts, trip_count in [
        ('sequence_repeat', np.int64, [11, 17, 23, 31], [2, 0, 3, 1], 4),
        ('sequence_repeat_prefix', np.int64, [11, 17, 23, 31], [2, 0, 3, 1], 2),
        ('sequence_repeat_infer_dim', np.float32, [1.5, 2.25, -3.25, 4.5], [-1, 0, 2, 1], 4),
    ]:
        dt=numpy_helper.from_array(np.array(values,dtype)).data_type
        seq=helper.make_tensor_sequence_value_info('carried',dt,[None])
        body=helper.make_graph(body_nodes,'repeat_body',[
            helper.make_tensor_value_info('iteration',onnx.TensorProto.INT64,[]),
            helper.make_tensor_value_info('body_condition',onnx.TensorProto.BOOL,[]),seq],
            [helper.make_tensor_value_info('next_condition',onnx.TensorProto.BOOL,[]),
             helper.make_tensor_sequence_value_info('inserted',dt,[None])])
        nodes=[helper.make_node('SequenceEmpty',[],['initial'],dtype=dt),
               helper.make_node('Shape',['counts'],['count_shape']),
               helper.make_node('ConstantOfShape',['count_shape'],['unit_splits'],value=numpy_helper.from_array(np.array([1],np.int64))),
               helper.make_node('SplitToSequence',['counts','unit_splits'],['count_parts'],axis=0),
               helper.make_node('SplitToSequence',['values','unit_splits'],['value_parts'],axis=0),
               helper.make_node('Loop',['trip_count','condition','initial'],['result_sequence'],body=body),
               helper.make_node('ConcatFromSequence',['result_sequence'],['output'],axis=0)]
        save(name,{'values':np.array(values,dtype),'counts':np.array(counts,np.int64),'trip_count':np.array(trip_count,np.int64)},nodes,{'condition':np.array(True)},dynamic=True,output_names=['output'])


def if_constant_rank():
    then=helper.make_graph([helper.make_node("Squeeze",["data","axis"],["then_output"])],"then",[],[helper.make_tensor_value_info("then_output",onnx.TensorProto.FLOAT,[None])])
    otherwise=helper.make_graph([helper.make_node("Identity",["data"],["else_output"])],"else",[],[helper.make_tensor_value_info("else_output",onnx.TensorProto.FLOAT,[1,None])])
    save("if_constant_rank",{"data":np.arange(7,dtype=np.float32).reshape(1,7)},[helper.make_node("If",["condition"],["selected"],then_branch=then,else_branch=otherwise),helper.make_node("Mul",["selected","two"],["output"])],{"condition":np.array(True),"axis":np.array([0],np.int64),"two":np.array(2,np.float32)},dynamic=True,output_names=["output"])


if __name__ == "__main__":
    v11_sequence_and_shapes()
    if_constant_rank()
    finalize(ROOT)
