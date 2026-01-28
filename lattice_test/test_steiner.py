from lattice_double_patches import LayoutEngine
from attached_preset_layout import build_7x9_magic_ring_layout

eng = build_7x9_magic_ring_layout()
graph, ports_by_patch, pos, patch_used_by_port = eng.build_routing_graph()
eng.visualize_graph(graph, pos, "Routing graph overlay")

# pick terminals: one magic port + one data port
t_magic = ports_by_patch["mT1"]["M"][0]
t_data  = ports_by_patch["q0"]["X"][0]

sol_nodes, sol_edges = eng.steiner_tree(graph, [t_magic, t_data])
eng.visualize_solution(graph, pos, sol_edges, terminals=[t_magic, t_data],
                       title="Steiner (2 terminals = shortest path)")
