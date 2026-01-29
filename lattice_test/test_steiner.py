from lattice_double_patches import LayoutEngine
from attached_preset_layout import build_7x9_magic_ring_layout, nxm_ring_layout_single_qubits

#eng = build_7x9_magic_ring_layout()
eng = nxm_ring_layout_single_qubits(4, 4)
graph, ports_by_patch, pos, patch_used_by_port = eng.build_routing_graph()
eng.visualize_graph(graph, pos, "Routing graph overlay")

# pick terminals: one magic port + one data port
#t_magic = ports_by_patch["mT1"]["M"][0]
#t_data  = ports_by_patch["q0"]["X"][0]
terminals = [
    ports_by_patch["mB3"]["M"][0],
    ports_by_patch["q_0_0"]["X"][1],
    ports_by_patch["q_2_0"]["Z"][1],
]

sol_nodes, sol_edges = eng.steiner_tree(graph, terminals)
eng.visualize_solution(graph, pos, sol_edges, terminals=terminals,
                       title="Steiner (2 terminals = shortest path)")
