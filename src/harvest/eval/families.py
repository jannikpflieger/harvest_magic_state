"""Circuit families: group benchmarks by what they compute, not by folder.

``circuit_family("ising_n420") == "ising"``, ``"gf2^16_mult" -> "gf2_mult"``,
``"qaoa_barabasi_albert_N40_3reps" -> "qaoa"``, ``"tof_4" -> "tof"``.  The
size suffix (``_n<k>``, ``_N<k>``, ``_<k>``, ``^<k>``, ``-low`` ...) is
stripped and a few aliases merge spellings of the same construction.
"""

from __future__ import annotations

import re

_ALIASES = {
    "qft_n": "qft", "qaoa_barabasi_albert": "qaoa", "square_heisenberg": "heisenberg_square",
    "transpiled_trotterataisingtr": "trotter_ising_ata", "transpiled_trotterchain": "trotter_chain",
    "transpiled_trotterheisenbergxyz": "trotter_heisenberg_xyz",
    "transpiled_trotter2dguagerotatedsurface": "trotter_surface", "csla_mux": "mux",
    "csum_mux": "mux", "mod_adder": "modular", "mod_mult": "modular", "mod_red": "modular",
    "mod5": "modular", "mod": "modular", "ham": "ham15", "qcla_adder": "qcla", "qcla_com": "qcla", "qcla_mod": "qcla",
    "ham15": "ham15", "random_q_depth_seed": "random", "qv_n": "qv", "dtc_cx": "dtc",
    "dtc_cz": "dtc", "bigadder": "adder", "multiply": "multiplier",
}
# Approximately synthesized families (arbitrary-angle rotations); the rest are exact.
APPROXIMATE = {"ising", "qaoa", "qft", "heisenberg_square", "knn", "qv", "random", "dtc",
               "trotter_ising_ata", "trotter_chain", "trotter_heisenberg_xyz", "trotter_surface",
               "qugan", "vqe_uccsd", "vqe", "dnn", "swap_test", "wstate", "square_root", "qpe",
               "hhl", "pea", "ipea", "variational", "basis_trotter", "quantumwalks", "qrng"}


def circuit_family(name: str) -> str:
    stem = name.lower().removesuffix(".qasm")
    stem = re.sub(r"_transpiled$", "", stem)
    stem = re.sub(r"\^\d+", "", stem)                      # gf2^16_mult -> gf2_mult
    stem = re.sub(r"[-_](low|med|high)$", "", stem)        # ham15-high
    stem = re.sub(r"_\d+reps$", "", stem)                  # qaoa ..._3reps
    stem = re.sub(r"_qb\d+$|_distance\d+$", "", stem)      # chemistry trotter sizes
    stem = re.sub(r"_(n|N)?\d+(_\d+)?$", "", stem, flags=re.I)  # _n420, _N004_12345, _4
    stem = re.sub(r"(\d+)q_depth\d+_seed\d+$", "q_depth_seed", stem)
    stem = re.sub(r"\d+$", "", stem).rstrip("_-")          # hwb10, mod5_4 -> mod5
    if stem.startswith("random_"):
        stem = "random_q_depth_seed"
    if stem.startswith("dtc"):
        return "dtc"
    return _ALIASES.get(stem, stem)


def synthesis_kind(family: str) -> str:
    return "approximate" if family in APPROXIMATE else "exact"
