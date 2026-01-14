"""
Grid lattice visualization module for quantum computing layouts.

This module provides visual representations of the lattice layout showing
data qubits and ancilla qubits in different states.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap

from .lattice_layout import Layout, AncillaState, DataPatch

# Simple color scheme
COLORS = {
    'data': '#1f77b4',          # Blue for data qubits
    'ancilla': '#FFFFFF',        # White for ancilla qubits
    'grid_lines': '#CCCCCC',     # Light gray for grid lines
    'text': '#333333'            # Dark gray for text
}

def visualize_layout(layout: Layout, title: str = "Quantum Lattice Layout", 
                    figsize: Tuple[int, int] = (10, 8), 
                    show_qubit_ids: bool = True,
                    save_path: Optional[str] = None) -> None:
    """
    Create a simple visual representation of the lattice layout.
    
    Args:
        layout: The Layout object to visualize
        title: Title for the plot
        figsize: Figure size as (width, height)
        show_qubit_ids: Whether to show data qubit IDs
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create a simple binary color matrix (0 = ancilla, 1 = data)
    color_matrix = np.full((layout.H, layout.W), 0, dtype=int)
    
    # Fill in the matrix: 1 for data qubits, 0 for ancilla
    for y in range(layout.H):
        for x in range(layout.W):
            coord = (x, y)
            if coord in layout.data_at:
                color_matrix[y, x] = 1  # Data qubit
    
    # Define simple colormap
    colors = [COLORS['ancilla'], COLORS['data']]
    cmap = ListedColormap(colors)
    
    # Create the main plot
    im = ax.imshow(color_matrix, cmap=cmap, aspect='equal', origin='upper')
    
    # Add grid lines
    for x in range(layout.W + 1):
        ax.axvline(x - 0.5, color=COLORS['grid_lines'], linewidth=0.5)
    for y in range(layout.H + 1):
        ax.axhline(y - 0.5, color=COLORS['grid_lines'], linewidth=0.5)
    
    # Add qubit ID labels for data qubits
    if show_qubit_ids:
        for y in range(layout.H):
            for x in range(layout.W):
                coord = (x, y)
                if coord in layout.data_at:
                    data_patch = layout.data_at[coord]
                    ax.text(x, y, f'Q{data_patch.qid}', ha='center', va='center', 
                           fontsize=10, fontweight='bold', color='white')
    
    # Customize the plot
    ax.set_xlim(-0.5, layout.W - 0.5)
    ax.set_ylim(-0.5, layout.H - 0.5)
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Invert y-axis to match typical coordinate system
    ax.invert_yaxis()
    
    # Create simple legend
    legend_elements = [
        patches.Patch(color=COLORS['data'], label='Data Qubits'),
        patches.Patch(color=COLORS['ancilla'], label='Ancilla Qubits')
    ]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Add basic statistics
    stats_text = f"""Layout Statistics:
Dimensions: {layout.W} × {layout.H}
Data Qubits: {len(layout.data_at)}
Ancilla Qubits: {layout.W * layout.H - len(layout.data_at)}"""
    
    ax.text(1.02, 0.02, stats_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()



if __name__ == "__main__":
    # Simple example usage
    from .lattice_layout import make_layout
    
    # Create a test layout
    layout = make_layout(n_qubits=6, W=12, H=10)
    
    # Visualize
    visualize_layout(layout, "Simple Lattice Layout")