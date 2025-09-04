#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import onnx
from onnx import helper, TensorProto, numpy_helper

SCOPE_DEFAULT = "prompt_encoder"  # 只修这个作用域，安全保守；可用 --all 放开

def _collect_used_names(graph: onnx.GraphProto):
    used = set()
    for n in graph.node:
        used.update(n.output)
    for init in graph.initializer:
        used.add(init.name)
    for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
        used.add(vi.name)
    return used

def _unique_name(base: str, used: set[str]) -> str:
    name = base
    idx = 0
    while name in used or name == "":
        idx += 1
        name = f"{base}_{idx}"
    used.add(name)
    return name

def _make_const_vec(output_name: str, vals, dtype=TensorProto.INT64):
    if isinstance(vals, int):
        vals = [vals]
    tensor = helper.make_tensor(
        name=output_name + "_value",
        data_type=dtype,
        dims=[len(vals)],
        vals=vals,
    )
    node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=[output_name],
        name=output_name,  # node 名也唯一
        value=tensor,
    )
    return node

def _insert_before_nodes(graph: onnx.GraphProto, plan: list[tuple[int, list[onnx.NodeProto]]]):
    # plan: [(node_index_in_graph, [new_node1, new_node2, ...]), ...] 按原索引插入
    insert_map = {}
    for idx, new_list in plan:
        insert_map.setdefault(idx, []).extend(new_list)
    new_nodes = []
    for i, node in enumerate(graph.node):
        if i in insert_map:
            new_nodes.extend(insert_map[i])  # 先插入，再接原节点
        new_nodes.append(node)
    del graph.node[:]
    graph.node.extend(new_nodes)

def fix_where_and_expand(model: onnx.ModelProto, scope_filter: str):
    g = model.graph
    used_names = _collect_used_names(g)
    insert_plan = []

    # --- 1) 修 Where：cond 添加 Unsqueeze(-1) ---
    for i, node in enumerate(list(g.node)):
        if node.op_type != "Where":
            continue
        if scope_filter and (scope_filter not in (node.name or "")):
            continue

        cond_in = node.input[0]
        axes_out = _unique_name(node.name + "_fix_axes_lastdim", used_names)
        unsq_out = _unique_name(node.name + "_fix_cond_unsq", used_names)

        const_axes = _make_const_vec(axes_out, [-1])
        unsq = helper.make_node(
            "Unsqueeze",
            inputs=[cond_in, axes_out],  # v13：data, axes
            outputs=[unsq_out],
            name=_unique_name(node.name + "_fix_unsqueeze_node", used_names),
        )

        # 替换 Where 的 cond
        node.input[0] = unsq_out

        # 记录插入计划（在该 Where 之前插）
        insert_plan.append((i, [const_axes, unsq]))

    # --- 2) 修 Expand：shape 规范为 1D int64 ---
    init_map = {init.name: init for init in g.initializer}

    for i, node in enumerate(list(g.node)):
        if node.op_type != "Expand":
            continue
        if scope_filter and (scope_filter not in (node.name or "")):
            continue
        if len(node.input) != 2:
            continue

        shape_in = node.input[1]

        if shape_in in init_map:
            # A) shape 是 initializer：改为 1D int64
            init = init_map[shape_in]
            np_val = numpy_helper.to_array(init)
            if np_val.ndim != 1 or init.data_type != TensorProto.INT64:
                np_val = np_val.reshape(-1).astype("int64")
                new_init = numpy_helper.from_array(np_val, name=shape_in)
                # 替换 initializer
                for k, old in enumerate(list(g.initializer)):
                    if old.name == shape_in:
                        g.initializer.remove(old)
                        break
                g.initializer.append(new_init)
        else:
            # B) 动态 shape：Cast -> Reshape([-1])
            cast_out = _unique_name(node.name + "_fix_shape_cast_i64", used_names)
            cast_node = helper.make_node(
                "Cast",
                inputs=[shape_in],
                outputs=[cast_out],
                name=_unique_name(node.name + "_fix_cast_node", used_names),
                to=TensorProto.INT64,
            )

            vec_shape_const = _unique_name(node.name + "_fix_shape_vec", used_names)
            const_vec = _make_const_vec(vec_shape_const, [-1])

            reshape_out = _unique_name(node.name + "_fix_shape_1d", used_names)
            reshape_node = helper.make_node(
                "Reshape",
                inputs=[cast_out, vec_shape_const],
                outputs=[reshape_out],
                name=_unique_name(node.name + "_fix_reshape_node", used_names),
            )

            node.input[1] = reshape_out
            insert_plan.append((i, [const_vec, cast_node, reshape_node]))

    _insert_before_nodes(g, insert_plan)
    return model

def main(inp: str, outp: str, patch_all: bool):
    m = onnx.load(inp)
    scope = "" if patch_all else SCOPE_DEFAULT
    m = fix_where_and_expand(m, scope_filter=scope)
    onnx.checker.check_model(m)
    onnx.save(m, outp)
    print(f"[OK] saved fixed model to: {outp}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser("Fix PromptEncoder broadcasting (Where + Expand) with SSA-safe names")
    ap.add_argument("--in", dest="inp", required=True, help="path to interactive decoder onnx")
    ap.add_argument("--out", dest="out", required=True, help="output path")
    ap.add_argument("--all", action="store_true", help="patch ALL Where/Expand nodes (default: only those under 'prompt_encoder')")
    args = ap.parse_args()
    main(args.inp, args.out, args.all)
