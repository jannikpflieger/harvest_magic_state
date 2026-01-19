import pennylane as qml
import jax.numpy as jnp

dev = qml.device("null.qubit", wires=3)

def qfunc():
    qml.Hadamard(wires=0)
    qml.S(wires=1)
    qml.PauliX(wires=2)
    qml.CNOT(wires=[0, 1])
    qml.CZ(wires=[1, 2])
    qml.T(wires=0)
    qml.RZ(jnp.pi / 4, wires=2)
    qml.SX(wires=0)
    qml.PauliY(wires=1)

qml.capture.enable()
qml.decomposition.enable_graph()

gate_set = {qml.T, qml.S, qml.CNOT, qml.H, qml.GlobalPhase, qml.PauliRot, qml.ops.PauliMeasure}

@qml.qjit(target="mlir")
@qml.transforms.ppm_compilation
@qml.transforms.gridsynth
@qml.transforms.decompose(gate_set=gate_set)
@qml.qnode(dev)
def circuit():
    qfunc()
    return qml.expval(qml.Z(0))

print(qml.specs(circuit, level=5)())