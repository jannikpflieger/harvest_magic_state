"""
Test Steiner tree routing on the lattice layout.
"""

from harvest.layout import LayoutEngine, nxm_ring_layout_single_qubits

eng = nxm_ring_layout_single_qubits(4, 4)
graph, ports_by_patch, pos, patch_used_by_port = eng.build_routing_graph()
eng.visualize_graph(graph, pos, "Routing graph overlay")

terminals = [
    ports_by_patch["mB3"]["M"][0],
    ports_by_patch["q_0_0"]["X"][1],
    ports_by_patch["q_2_0"]["Z"][1],
]

sol_nodes, sol_edges = eng.steiner_tree(graph, terminals)
eng.visualize_solution(graph, pos, sol_edges, terminals=terminals,
                       title="Steiner (2 terminals = shortest path)")
