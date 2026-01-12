from circuit.pre_processing import convert_to_PCB, convert_to_CliffordT
from lattice.lattice_layout import create_default_lattice_layout

def start_pipline(circuit, lattice_layout = None, is_PCB = False, is_CLiffordT = False):
    if not is_CLiffordT:
        circuit = convert_to_CliffordT(circuit)
    if not is_PCB:
        circuit = convert_to_PCB(circuit)
    
    if lattice_layout is None:
        create_default_lattice_layout(circuit)

    return circuit
    

if __name__ == "__main__":
    start_pipline("lol")
    pass
