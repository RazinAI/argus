"""Append batched NMS to a YOLO ONNX model exported with nms=False.

Ultralytics' own nms=True export unrolls NMS for the export-time batch size,
so a batch=1 export only returns detections for image 0 of every batch. This
script handles the batch dimension itself, in one of two modes:

  --mode onnx   ONNX NonMaxSuppression. Its output length depends on the data,
                so TensorRT reads the box count back to the CPU mid-model and
                enqueueV3 blocks until the GPU gets there.
  --mode fixed  Cluster-NMS from fixed-shape ops: the top --candidates scores,
                their pairwise IoU matrix, and --iterations rounds of "suppressed
                by a kept, higher-scoring box". Every shape is known at build
                time, so nothing is read back. It matches greedy NMS once the
                rounds converge, which takes as many rounds as the longest chain
                of boxes suppressing each other. Scores past --candidates are
                dropped.

Input:  output0 [batch, 4 + classes, anchors], boxes as cx, cy, w, h
Output: output0 [batch, max_det, 6], rows of x1, y1, x2, y2, conf, class,
        sorted by conf. Unused rows have conf 0.

--uint8-input replaces the float input with images [batch, 640, 640, 3]
uint8, interleaved RGB as swscale writes it, and converts it inside the
model: Cast to float, divide by 255, transpose to planar. Frames then cross
PCIe as 1.2 MB instead of 4.9 MB.

--fp16 runs the YOLO layers in fp16, which Ampere and newer tensor cores do
much faster. The engine builder uses strongly typed networks, so precision
comes from the types in this file. The NMS block stays fp32: class-offset
boxes go past fp16's 65504 maximum.

Usage: python tools/add_nms.py in.onnx out.onnx [--conf 0.25] [--iou 0.7]
       [--max-det 300] [--mode onnx|fixed] [--candidates 300]
       [--iterations 4] [--uint8-input] [--fp16]
"""

import argparse

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

# Shifts each class's boxes this far apart so one class-agnostic NMS pass
# behaves like per-class NMS (same trick as Ultralytics).
CLASS_OFFSET = 7680.0


def fixed_nms(args, const, node, reduce_max):
    """nms_boxes_shifted [batch, anchors, 4] and nms_best [batch, 1, anchors]
    -> nms_conf_top [batch, max_det] and nms_idx (anchor index) [batch, max_det]."""
    k = args.candidates
    const("nms_k", np.array([k], dtype=np.int64))
    const("nms_conf_s", np.array(args.conf, dtype=np.float32))
    const("nms_iou_s", np.array(args.iou, dtype=np.float32))
    const("nms_one", np.array(1.0, dtype=np.float32))
    const("nms_tiny", np.array(1e-9, dtype=np.float32))
    const("nms_xyxy_split", np.array([1, 1, 1, 1], dtype=np.int64))
    const("nms_rows", np.arange(k, dtype=np.int64).reshape(k, 1))
    const("nms_cols", np.arange(k, dtype=np.int64).reshape(1, k))

    # Zero scores under the conf threshold, then keep the k best anchors,
    # sorted, with their (class-shifted) boxes.
    node("Squeeze", ["nms_best", "nms_split_axis"], ["nms_best_2d"])
    node("Greater", ["nms_best_2d", "nms_conf_s"], ["nms_pass"])
    node("Cast", ["nms_pass"], ["nms_pass_f"], to=TensorProto.FLOAT)
    node("Mul", ["nms_best_2d", "nms_pass_f"], ["nms_best_pass"])
    node("TopK", ["nms_best_pass", "nms_k"], ["nms_cand_conf", "nms_cand_idx"],
         axis=1, largest=1, sorted=1)
    node("Unsqueeze", ["nms_cand_idx", "nms_last_axis"], ["nms_cand_idx_3d"])
    node("Expand", ["nms_cand_idx_3d", "nms_box_cols"], ["nms_cand_idx_boxes"])
    node("GatherElements", ["nms_boxes_shifted", "nms_cand_idx_boxes"],
         ["nms_cand_boxes"], axis=1)

    # IoU of every candidate pair [batch, k, k]: rows i against columns j.
    node("Split", ["nms_cand_boxes", "nms_xyxy_split"],
         ["nms_ax1", "nms_ay1", "nms_ax2", "nms_ay2"], axis=2)
    for c in ("x1", "y1", "x2", "y2"):
        node("Transpose", [f"nms_a{c}"], [f"nms_b{c}"], perm=[0, 2, 1])
    for axis in ("x", "y"):
        node("Min", [f"nms_a{axis}2", f"nms_b{axis}2"], [f"nms_{axis}_hi"])
        node("Max", [f"nms_a{axis}1", f"nms_b{axis}1"], [f"nms_{axis}_lo"])
        node("Sub", [f"nms_{axis}_hi", f"nms_{axis}_lo"], [f"nms_{axis}_span"])
        node("Relu", [f"nms_{axis}_span"], [f"nms_{axis}_overlap"])
    node("Mul", ["nms_x_overlap", "nms_y_overlap"], ["nms_inter"])
    for side in ("a", "b"):
        node("Sub", [f"nms_{side}x2", f"nms_{side}x1"], [f"nms_{side}w"])
        node("Sub", [f"nms_{side}y2", f"nms_{side}y1"], [f"nms_{side}h"])
        node("Mul", [f"nms_{side}w", f"nms_{side}h"], [f"nms_{side}area"])
    node("Add", ["nms_aarea", "nms_barea"], ["nms_area_sum"])
    node("Sub", ["nms_area_sum", "nms_inter"], ["nms_union"])
    node("Max", ["nms_union", "nms_tiny"], ["nms_union_safe"])
    node("Div", ["nms_inter", "nms_union_safe"], ["nms_pair_iou"])

    # suppress[b, i, j] = 1 when higher-scoring box i overlaps box j too much.
    # Candidates are sorted, so "higher-scoring" is just i < j.
    node("Greater", ["nms_pair_iou", "nms_iou_s"], ["nms_overlaps"])
    node("Less", ["nms_rows", "nms_cols"], ["nms_earlier"])
    node("And", ["nms_overlaps", "nms_earlier"], ["nms_suppress_b"])
    node("Cast", ["nms_suppress_b"], ["nms_suppress"], to=TensorProto.FLOAT)

    # Only kept boxes suppress. Start from every candidate above the threshold
    # and repeat; each round fixes boxes that were wrongly suppressed by a box
    # that is itself suppressed.
    node("Greater", ["nms_cand_conf", "nms_zero_f"], ["nms_valid_b"])
    const("nms_zero_f", np.array(0.0, dtype=np.float32))
    node("Cast", ["nms_valid_b"], ["nms_keep_0"], to=TensorProto.FLOAT)
    for t in range(args.iterations):
        node("Unsqueeze", [f"nms_keep_{t}", "nms_last_axis"], [f"nms_keep_col_{t}"])
        node("Mul", ["nms_suppress", f"nms_keep_col_{t}"], [f"nms_active_{t}"])
        reduce_max(f"nms_active_{t}", f"nms_hit_{t}", 1, 0)
        node("Sub", ["nms_one", f"nms_hit_{t}"], [f"nms_free_{t}"])
        node("Mul", ["nms_keep_0", f"nms_free_{t}"], [f"nms_keep_{t + 1}"])
    node("Mul", ["nms_cand_conf", f"nms_keep_{args.iterations}"], ["nms_cand_kept"])

    # Highest max_det kept scores, mapped back to anchor indices.
    node("TopK", ["nms_cand_kept", "nms_max_det"], ["nms_conf_top", "nms_top_pos"],
         axis=1, largest=1, sorted=1)
    node("GatherElements", ["nms_cand_idx", "nms_top_pos"], ["nms_idx"], axis=1)


def fp16_model(model, head):
    """Run every op that computes from the image in fp16. Ops fed only by
    shapes (the anchor grid) stay fp32 and get cast where they meet image
    data. The input and head keep fp32 at the graph edges."""
    graph = model.graph
    inferred = onnx.shape_inference.infer_shapes(model).graph
    dtype = {v.name: v.type.tensor_type.elem_type
             for v in [*inferred.input, *inferred.value_info, *inferred.output]}
    inits = {i.name: i for i in graph.initializer}
    dtype.update((name, i.data_type) for name, i in inits.items())

    name = graph.input[0].name
    nodes = [helper.make_node("Cast", [name], ["fp16_input"], name="/fp16/input",
                              to=TensorProto.FLOAT16)]
    half = {"fp16_input"}
    casts = {}

    def to_half(t):
        if t not in casts:
            casts[t] = f"{t}_fp16"
            if t in inits:
                value = numpy_helper.to_array(inits[t]).astype(np.float16)
                graph.initializer.append(numpy_helper.from_array(value, casts[t]))
            else:
                nodes.append(helper.make_node("Cast", [t], [casts[t]],
                                              name=f"/fp16/{t}", to=TensorProto.FLOAT16))
        return casts[t]

    for n in graph.node:
        n.input[:] = ["fp16_input" if i == name else i for i in n.input]
        if not any(i in half for i in n.input) or n.op_type == "Shape":
            nodes.append(n)
            continue
        if n.op_type == "Cast":
            raise SystemExit(f"fp16: unexpected Cast of image data in {n.name}")
        # Resize's roi and scales must stay fp32.
        data_inputs = n.input[:1] if n.op_type == "Resize" else n.input
        for k, i in enumerate(data_inputs):
            if i and i not in half and dtype.get(i) == TensorProto.FLOAT:
                n.input[k] = to_half(i)
        nodes.append(n)
        half.update(o for o in n.output if o)

    if head not in half:
        raise SystemExit("fp16: head is not computed from the input")
    for n in nodes:
        n.output[:] = [f"{head}_fp16" if o == head else o for o in n.output]
    nodes.append(helper.make_node("Cast", [f"{head}_fp16"], [head],
                                  name="/fp16/head", to=TensorProto.FLOAT))
    del graph.node[:]
    graph.node.extend(nodes)
    used = {i for n in graph.node for i in n.input}
    for init in [i for i in graph.initializer if i.name not in used]:
        graph.initializer.remove(init)


def uint8_input(graph, size):
    """Put a uint8 [batch, size, size, 3] input in front of the float
    [batch, 3, H, W] one, keeping the input name."""
    if len(graph.input) != 1:
        raise SystemExit("expected a model with one input")
    old = graph.input[0]
    if old.type.tensor_type.elem_type != TensorProto.FLOAT:
        raise SystemExit("model input is not float")
    name = old.name
    planar = "u8_planar"
    for n in graph.node:
        n.input[:] = [planar if i == name else i for i in n.input]

    graph.initializer.append(
        numpy_helper.from_array(np.array(255.0, dtype=np.float32), "u8_scale"))
    nodes = [
        helper.make_node("Cast", [name], ["u8_float"], name="/u8/cast",
                         to=TensorProto.FLOAT),
        helper.make_node("Div", ["u8_float", "u8_scale"], ["u8_scaled"],
                         name="/u8/div"),
        helper.make_node("Transpose", ["u8_scaled"], [planar],
                         name="/u8/transpose", perm=[0, 3, 1, 2]),
    ]
    for i, n in enumerate(nodes):
        graph.node.insert(i, n)

    new = helper.make_tensor_value_info(name, TensorProto.UINT8,
                                        [None, size, size, 3])
    new.type.tensor_type.shape.dim[0].CopyFrom(old.type.tensor_type.shape.dim[0])
    del graph.input[:]
    graph.input.append(new)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--mode", choices=["onnx", "fixed"], default="onnx")
    parser.add_argument("--candidates", type=int, default=300)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--uint8-input", action="store_true")
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    if args.mode == "fixed" and args.candidates < args.max_det:
        raise SystemExit("--candidates must be at least --max-det")

    model = onnx.load(args.input)
    graph = model.graph
    if len(graph.output) != 1:
        raise SystemExit("expected a model with one output")

    head = graph.output[0]
    dims = head.type.tensor_type.shape.dim
    num_classes = dims[1].dim_value - 4
    if num_classes <= 0:
        raise SystemExit("output must be [batch, 4 + classes, anchors]")

    # Rename the head so the new final tensor can take its name.
    raw = "nms_raw"
    for node in graph.node:
        node.output[:] = [raw if o == head.name else o for o in node.output]
    if args.fp16:
        fp16_model(model, raw)

    def const(name, value):
        graph.initializer.append(numpy_helper.from_array(np.array(value), name))
        return name

    const("nms_split", np.array([2, 2, num_classes], dtype=np.int64))
    const("nms_half", np.array(0.5, dtype=np.float32))
    const("nms_offset", np.array(CLASS_OFFSET, dtype=np.float32))
    const("nms_max_det", np.array([args.max_det], dtype=np.int64))
    const("nms_iou", np.array([args.iou], dtype=np.float32))
    const("nms_conf", np.array([args.conf], dtype=np.float32))
    const("nms_zero", np.array([0], dtype=np.int64))
    const("nms_box_cols", np.array([1, 1, 4], dtype=np.int64))
    const("nms_split_axis", np.array([1], dtype=np.int64))

    def node(op, inputs, outputs, **attrs):
        graph.node.append(
            helper.make_node(op, inputs, outputs, name=f"/nms/{outputs[0]}", **attrs)
        )

    # Boxes: cx, cy, w, h -> x1, y1, x2, y2, laid out [batch, anchors, 4].
    node("Split", [raw, "nms_split"], ["nms_cxcy", "nms_wh", "nms_scores"], axis=1)
    node("Mul", ["nms_wh", "nms_half"], ["nms_half_wh"])
    node("Sub", ["nms_cxcy", "nms_half_wh"], ["nms_x1y1"])
    node("Add", ["nms_cxcy", "nms_half_wh"], ["nms_x2y2"])
    node("Concat", ["nms_x1y1", "nms_x2y2"], ["nms_xyxy"], axis=1)
    node("Transpose", ["nms_xyxy"], ["nms_boxes"], perm=[0, 2, 1])

    opset = next(o.version for o in model.opset_import if o.domain in ("", "ai.onnx"))

    def reduce_max(inp, out, axis, keepdims):
        if opset >= 18:  # ReduceMax takes axes as an input from opset 18
            axes = const(f"{out}_axes", np.array([axis], dtype=np.int64))
            node("ReduceMax", [inp, axes], [out], keepdims=keepdims)
        else:
            node("ReduceMax", [inp], [out], axes=[axis], keepdims=keepdims)

    # Best class per anchor: [batch, 1, anchors].
    reduce_max("nms_scores", "nms_best", 1, 1)
    node("ArgMax", ["nms_scores"], ["nms_class_i"], axis=1, keepdims=1)
    node("Cast", ["nms_class_i"], ["nms_class"], to=TensorProto.FLOAT)

    # Offset boxes by class, then one NMS pass per image.
    node("Transpose", ["nms_class"], ["nms_class_t"], perm=[0, 2, 1])
    node("Mul", ["nms_class_t", "nms_offset"], ["nms_shift"])
    node("Add", ["nms_boxes", "nms_shift"], ["nms_boxes_shifted"])
    if args.mode == "onnx":
        node(
            "NonMaxSuppression",
            ["nms_boxes_shifted", "nms_best", "nms_max_det", "nms_iou", "nms_conf"],
            ["nms_selected"],
        )

        # selected is [kept, 3] of (batch, 0, anchor). Turn it into a keep mask
        # the same shape as nms_best and zero every score NMS dropped.
        node("Shape", ["nms_best"], ["nms_best_shape"])
        node("ConstantOfShape", ["nms_best_shape"], ["nms_zeros"],
             value=helper.make_tensor("v", TensorProto.FLOAT, [1], [0.0]))
        node("Shape", ["nms_selected"], ["nms_selected_shape"])
        node("Gather", ["nms_selected_shape", "nms_zero"], ["nms_kept_count"], axis=0)
        node("ConstantOfShape", ["nms_kept_count"], ["nms_ones"],
             value=helper.make_tensor("v", TensorProto.FLOAT, [1], [1.0]))
        node("ScatterND", ["nms_zeros", "nms_selected", "nms_ones"], ["nms_mask"])
        node("Mul", ["nms_best", "nms_mask"], ["nms_kept"])

        # Highest max_det kept scores per image, then gather their boxes/classes.
        node("Squeeze", ["nms_kept", "nms_split_axis"], ["nms_kept_2d"])
        node("TopK", ["nms_kept_2d", "nms_max_det"], ["nms_conf_top", "nms_idx"],
             axis=1, largest=1, sorted=1)
    else:
        fixed_nms(args, const, node, reduce_max)

    node("Unsqueeze", ["nms_idx", "nms_last_axis"], ["nms_idx_3d"])
    const("nms_last_axis", np.array([2], dtype=np.int64))
    node("Expand", ["nms_idx_3d", "nms_box_cols"], ["nms_idx_boxes"])
    node("GatherElements", ["nms_boxes", "nms_idx_boxes"], ["nms_boxes_top"], axis=1)
    node("Squeeze", ["nms_class", "nms_split_axis"], ["nms_class_2d"])
    node("GatherElements", ["nms_class_2d", "nms_idx"], ["nms_class_top"], axis=1)

    node("Unsqueeze", ["nms_conf_top", "nms_last_axis"], ["nms_conf_col"])
    node("Unsqueeze", ["nms_class_top", "nms_last_axis"], ["nms_class_col"])
    node("Concat", ["nms_boxes_top", "nms_conf_col", "nms_class_col"],
         [head.name], axis=2)

    batch = dims[0]
    del graph.output[:]
    out = helper.make_tensor_value_info(head.name, TensorProto.FLOAT,
                                        [None, args.max_det, 6])
    out.type.tensor_type.shape.dim[0].CopyFrom(batch)
    graph.output.append(out)
    del graph.value_info[:]
    if args.uint8_input:
        uint8_input(graph, args.input_size)

    onnx.checker.check_model(model)
    onnx.save(model, args.output)
    print(f"wrote {args.output}: mode={args.mode} classes={num_classes} "
          f"conf={args.conf} iou={args.iou} max_det={args.max_det} "
          f"input={'uint8' if args.uint8_input else 'float'} "
          f"precision={'fp16' if args.fp16 else 'fp32'}")


if __name__ == "__main__":
    main()
