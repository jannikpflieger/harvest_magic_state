"""
Layout module for lattice-surgery surface code patches and routing topology.
"""

from .engine import (
    LayoutEngine,
    EdgePort,
    Patch,
    data_patch_1cell,
    magic_patch_1cell,
    wide_patch_1q_2cells,
    paired_patches_2q_alternating,
)

from .presets import (
    build_7x9_magic_ring_layout,
    nxm_ring_layout_single_qubits,
    nxm_ring_layout_single_qubits_large_spacing,
)

__all__ = [
    'LayoutEngine',
    'EdgePort',
    'Patch',
    'data_patch_1cell',
    'magic_patch_1cell',
    'wide_patch_1q_2cells',
    'paired_patches_2q_alternating',
    'build_7x9_magic_ring_layout',
    'nxm_ring_layout_single_qubits',
    'nxm_ring_layout_single_qubits_large_spacing',
]
