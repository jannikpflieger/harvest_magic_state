//! ``harvest._router``: HARVEST's per-step router in Rust.
//!
//! Nodes are the integer indices the Python wrapper
//! (``harvest.routing.rust_backend``) assigns to the routing graph's nodes.

mod assign;
mod pack;
mod search;
mod steiner;

use std::collections::{HashMap, HashSet};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use pack::{Core, Params, SetResult, SetSpec};
use search::Graph;

/// ``(kind, fixed terminals, candidates, [(patch, [port type])], initial)``;
/// kind 0 = fixed terminal set, 1 = magic-terminal choice.
type SetIn = (u8, Vec<u32>, Vec<u32>, Vec<(u32, Vec<u8>)>, Option<u32>);
/// ``(success, terminals or None, nodes, edges, error)``.
type SetOut = (bool, Option<Vec<u32>>, Vec<u32>, Vec<(u32, u32)>, Option<String>);
/// ``(iterations, initial_routes, reroutes, conflicted_reroutes, dropped,
/// filled, final_conflict_score, stopped_reason, no_defer_fired)``.
type StatsOut = (usize, usize, usize, usize, usize, usize, i64, String, bool);

fn to_specs(sets: Vec<SetIn>) -> Vec<SetSpec> {
    sets.into_iter()
        .map(|(kind, fixed, candidates, specs, initial)| {
            if kind == 0 {
                SetSpec::Fixed(fixed)
            } else {
                SetSpec::Choice { candidates, specs, initial }
            }
        })
        .collect()
}

fn to_out(results: Vec<SetResult>) -> Vec<SetOut> {
    results.into_iter().map(|r| (r.success, r.terminals, r.nodes, r.edges, r.error)).collect()
}

#[pyclass(module = "harvest._router")]
struct Router {
    core: Core,
}

#[pymethods]
impl Router {
    /// ``adj[u] = [(v, w)]``; ``cells[u]`` = (x, y) for routing cells;
    /// ``pos[u]`` = layout position (ports); ``dirs[u]`` = N/S/E/W bit mask
    /// of a port id; ``patch_ports[p] = [(type, [port])]`` in layout order;
    /// ``patch_pos[p]`` = position used to orient the patch's ports;
    /// ``bound``: Manhattan distance is a lower bound (A* allowed).
    #[new]
    fn new(
        adj: Vec<Vec<(u32, f64)>>,
        cells: Vec<Option<(i32, i32)>>,
        pos: Vec<Option<(f64, f64)>>,
        dirs: Vec<u8>,
        patch_ports: Vec<Vec<(u8, Vec<u32>)>>,
        patch_pos: Vec<Option<(f64, f64)>>,
        bound: bool,
    ) -> PyResult<Self> {
        let n = adj.len();
        if cells.len() != n || pos.len() != n || dirs.len() != n {
            return Err(PyValueError::new_err("adj, cells, pos and dirs must have one entry per node"));
        }
        if adj.iter().flatten().any(|&(v, _)| v as usize >= n) {
            return Err(PyValueError::new_err("edge to an unknown node"));
        }
        let g = Graph { adj, cell: cells };
        Ok(Router { core: Core::new(g, pos, dirs, patch_ports, patch_pos, bound) })
    }

    /// Uncongested distance from ``source`` to every node (inf = unreachable).
    fn distances(&mut self, source: u32) -> Vec<f64> {
        self.core.distances(source).iter().map(|&d| if d >= search::INF { f64::INFINITY } else { d }).collect()
    }

    /// One Steiner tree; raises ``ValueError`` if the terminals are not connected.
    #[pyo3(signature = (terminals, node_cost=None, blocked=None))]
    fn steiner_tree(
        &mut self,
        terminals: Vec<u32>,
        node_cost: Option<HashMap<u32, f64>>,
        blocked: Option<Vec<u32>>,
    ) -> PyResult<(Vec<u32>, Vec<(u32, u32)>)> {
        let blocked: Option<HashSet<u32>> = blocked.map(|b| b.into_iter().collect());
        let cost_fn = |v: u32| node_cost.as_ref().and_then(|c| c.get(&v).copied()).unwrap_or(0.0);
        let block_fn = |v: u32| blocked.as_ref().map_or(false, |b| b.contains(&v));
        let c: Option<&dyn Fn(u32) -> f64> = if node_cost.is_some() { Some(&cost_fn) } else { None };
        let b: Option<&dyn Fn(u32) -> bool> = if blocked.is_some() { Some(&block_fn) } else { None };
        self.core
            .steiner(&terminals, c, b)
            .map(|t| (t.nodes, t.edges))
            .map_err(PyValueError::new_err)
    }

    /// Negotiated packing of one step (``LayoutEngine.steiner_packing_harvest``).
    #[pyo3(signature = (sets, max_iters, alpha, beta, capacity, stagnation_limit, greedy_fill,
                        priorities=None, no_defer=false, match_initial=false, greedy_order="min_size"))]
    #[allow(clippy::too_many_arguments)]
    fn pack_harvest(
        &mut self,
        sets: Vec<SetIn>,
        max_iters: usize,
        alpha: f64,
        beta: f64,
        capacity: i32,
        stagnation_limit: usize,
        greedy_fill: bool,
        priorities: Option<Vec<f64>>,
        no_defer: bool,
        match_initial: bool,
        greedy_order: &str,
    ) -> PyResult<(Vec<SetOut>, StatsOut)> {
        let sets = to_specs(sets);
        if let Some(p) = &priorities {
            if p.len() != sets.len() {
                return Err(PyValueError::new_err("one priority per terminal set"));
            }
        }
        let params = Params { max_iters, alpha, beta, capacity, stagnation_limit, greedy_fill, priorities,
                              no_defer, match_initial, greedy_order: greedy_order.to_string() };
        let (results, s) = self.core.pack_harvest(&sets, &params);
        Ok((to_out(results), (s.iterations, s.initial_routes, s.reroutes, s.conflicted_reroutes,
                              s.dropped, s.filled, s.final_conflict_score, s.stopped_reason, s.no_defer_fired)))
    }

    /// Greedy packing of one step (``LayoutEngine.steiner_packing``).
    #[pyo3(signature = (sets, greedy_order="min_size", match_initial=false))]
    fn pack_greedy(&mut self, sets: Vec<SetIn>, greedy_order: &str, match_initial: bool) -> Vec<SetOut> {
        to_out(self.core.pack_greedy(&to_specs(sets), greedy_order, match_initial))
    }
}

#[pymodule]
fn _router(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Router>()?;
    Ok(())
}
