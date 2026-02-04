from circuit.circuit_to_pbc_dag import convert_to_PCB, convert_to_CliffordT, create_dag, create_random_circuit


def start_pipline(circuit, lattice_layout = None, is_PCB = False, is_CLiffordT = False):
    #if not is_CLiffordT:
    #    circuit = convert_to_CliffordT(circuit)
    if not is_PCB:
        circuit = convert_to_PCB(circuit)

    dag = create_dag(circuit)


    return dag
    

if __name__ == "__main__":
    qc = create_random_circuit(5, 10, seed=42)
    dag = start_pipline(qc)
    print(dag)
    pass
