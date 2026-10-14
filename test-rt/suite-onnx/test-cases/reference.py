"""Reference fixture generation; Python is needed only to regenerate checked corner cases."""
from pathlib import Path
import json
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper

def save(ROOT, name, inputs, nodes, constants, dynamic=False, output_names=None, reference=None):
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


def finalize(ROOT):
    for case in ROOT.iterdir():
        if not case.is_dir():
            continue
        for data in case.glob("test_data_set_*"):
            payload = {}
            for prefix in ["input", "output"]:
                tensors = []
                for pb in sorted(data.glob(prefix + "_*.pb"), key=lambda p: int(p.stem.split("_")[1])):
                    tensor = onnx.TensorProto.FromString(pb.read_bytes())
                    a = numpy_helper.to_array(tensor)
                    raw = a.ravel()
                    bits = raw.view(np.dtype((np.void, raw.dtype.itemsize)))
                    period = len(raw)
                    for size in range(1, min(len(raw), 64) + 1):
                        if len(raw) % size == 0 and np.array_equal(bits, np.tile(bits[:size], len(raw) // size)):
                            period = size
                            break
                    values = raw[:period].tolist()
                    values = [str(v) if isinstance(v, float) and not np.isfinite(v) else v for v in values]
                    descriptor = dict(dtype=tensor.data_type, shape=list(a.shape), values=values)
                    runs = []
                    for value in values:
                        if runs and repr(runs[-1][0]) == repr(value):
                            runs[-1][1] += 1
                        else:
                            runs.append([value, 1])
                    if len(values) > 32 and len(runs) * 2 < len(values):
                        descriptor.pop("values")
                        descriptor["runs"] = runs
                    tensors.append(descriptor)
                payload[prefix] = tensors
            text = "{\n" + ",\n".join(
                "  " + json.dumps(k) + ": [\n" + ",\n".join(
                    "    " + json.dumps(t, allow_nan=False) for t in v
                ) + "\n  ]" for k, v in payload.items()
            ) + "\n}\n"
            (data / "data.json").write_text(text)
            for pb in data.glob("*.pb"):
                pb.unlink()
