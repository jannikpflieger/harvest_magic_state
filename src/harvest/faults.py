"""Fault injection for the validator mutation study (EVAL_PLAN B.7 / E7b).

A fault is active only while the environment variable ``HARVEST_FAULT`` names
it; unset (or empty) means every hook is inert and the scheduler and pruner
behave exactly as without this module.  Each fault fires at its first
opportunity in a run (``fire_once``) so it models one subtle bug rather than
a wholesale breakage.
"""

from __future__ import annotations

import os

ENV_VAR = "HARVEST_FAULT"
FAULTS = (
    "no_defer",        # skip the congestion drop loop (conflicting routes are committed)
    "reuse_terminal",  # give two products of one step the same magic terminal
    "skip_dep",        # mark one not-yet-ready product as ready
    "early_consume",   # consume magic states before routing succeeds
    "prune_used",      # the pruner drops one routing tile a route uses
    "drop_y_port",     # omit the Z port of Y factors
)


def current():
    """The active fault name, or None.  Unknown names are an error, not a no-op."""
    value = os.environ.get(ENV_VAR)
    if not value:
        return None
    if value not in FAULTS:
        raise ValueError(f"{ENV_VAR}={value!r} is not one of {FAULTS}")
    return value


def active(name: str) -> bool:
    return current() == name


# How often each fault changed behavior in the current run.  A fault that
# never activates (e.g. drop_y_port on a circuit without Y factors) is an
# equivalent mutant, so kill rates are reported over activated runs only.
_ACTIVATIONS: dict = {}


def note(name: str) -> None:
    _ACTIVATIONS[name] = _ACTIVATIONS.get(name, 0) + 1


def reset_activations() -> None:
    _ACTIVATIONS.clear()


def activations() -> dict:
    return dict(_ACTIVATIONS)


def fire_once(holder, name: str) -> bool:
    """True the first time ``name`` is active for ``holder`` (e.g. a processor)."""
    if not active(name):
        return False
    fired = holder.__dict__.setdefault("_faults_fired", set())
    if name in fired:
        return False
    fired.add(name)
    note(name)
    return True
