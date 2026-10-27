"""Evaluation infrastructure for the HARVEST paper (see EVAL_PLAN.md).

Modules:

``validate``     schedule validator (rules V1-V4)
``faults``       ``HARVEST_FAULT`` mutation hooks (inactive unless the env var is set)
``fingerprint``  canonical schedule hashes for the bit-identity regression gate
``manifest``     ``manifest.json`` / ``runs.csv`` run metadata
``collect``      raw trial JSON -> tidy ``data.csv`` rows
``stats``        geomeans over shared subsets, confidence intervals
``features``     per-circuit features (parallelism, T fraction, Pauli weight, ...)
``placement``    index / random / circuit-aware placement on any preset layout
``plotting``     shared PDF style (ACM widths, fonts, one color per compiler)
"""
