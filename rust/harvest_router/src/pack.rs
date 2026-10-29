//! One scheduler step: route a set of products node-disjointly.
//!
//! Ports of ``LayoutEngine.steiner_packing_harvest`` (negotiated congestion:
//! initial routes, selective rip-up and reroute with present + history costs,
//! dropping until capacity holds, greedy fill), ``LayoutEngine.steiner_packing``
//! (greedy), ``LayoutEngine._resolve_terminals`` (the router's magic-terminal
//! choice), ``DAGProcessor._compute_best_oriented_port`` (data-port
//! orientation) and the min-cost matching of ``scheduler._assign_terminals_matching``.

use std::collections::{HashMap, HashSet};

use crate::assign::assign;
use crate::search::{all_distances, nearest_source, Graph, Workspace, INF};
use crate::steiner::{steiner_tree, Tree};

const DIR_N: u8 = 1;
const DIR_S: u8 = 2;
const DIR_E: u8 = 4;
const DIR_W: u8 = 8;

/// One product's terminals: fixed ports, or a magic port the router picks
/// among ``candidates`` plus data ports oriented towards it (``specs`` =
/// ``[(patch, [port types])]``).
pub enum SetSpec {
    Fixed(Vec<u32>),
    Choice { candidates: Vec<u32>, specs: Vec<(u32, Vec<u8>)>, initial: Option<u32> },
}

impl SetSpec {
    /// ``len(terminal_set)`` in Python (a choice set is ``[TerminalChoice]``).
    fn len(&self) -> usize {
        match self {
            SetSpec::Fixed(t) => t.len(),
            SetSpec::Choice { .. } => 1,
        }
    }
}

pub struct Params {
    pub max_iters: usize,
    pub alpha: f64,
    pub beta: f64,
    pub capacity: i32,
    pub stagnation_limit: usize,
    pub greedy_fill: bool,
    pub priorities: Option<Vec<f64>>,
    pub no_defer: bool,
    pub match_initial: bool,
    pub greedy_order: String,
}

pub struct SetResult {
    pub success: bool,
    /// The concrete terminal set (``None``: a choice that never got a port).
    pub terminals: Option<Vec<u32>>,
    pub nodes: Vec<u32>,
    pub edges: Vec<(u32, u32)>,
    pub error: Option<String>,
}

#[derive(Default)]
pub struct Stats {
    pub iterations: usize,
    pub initial_routes: usize,
    pub reroutes: usize,
    pub conflicted_reroutes: usize,
    pub dropped: usize,
    pub filled: usize,
    pub final_conflict_score: i64,
    pub stopped_reason: String,
    pub no_defer_fired: bool,
}

struct Route {
    success: bool,
    concrete: Option<Vec<u32>>,
    magic: Option<u32>,
    nodes: Vec<u32>,
    edges: Vec<(u32, u32)>,
    rnodes: Vec<u32>, // routing cells of the route that are not terminals
    error: Option<String>,
}

impl Route {
    fn failed(error: String) -> Route {
        Route { success: false, concrete: None, magic: None, nodes: vec![], edges: vec![],
                rnodes: vec![], error: Some(error) }
    }
}

/// Python tuple comparison ``a > b`` (lexicographic; a proper prefix is smaller).
fn tuple_gt(a: &[f64], b: &[f64]) -> bool {
    for (x, y) in a.iter().zip(b.iter()) {
        if x != y {
            return x > y;
        }
    }
    a.len() > b.len()
}

pub struct Core {
    pub g: Graph,
    pub bound: bool,
    pos: Vec<Option<(f64, f64)>>,
    dirs: Vec<u8>,
    patch_ports: Vec<Vec<(u8, Vec<u32>)>>,
    patch_pos: Vec<Option<(f64, f64)>>,
    orient: HashMap<(u32, u8, u32), Option<u32>>,
    static_dist: HashMap<u32, Vec<f64>>,
    pub ws: Workspace,
}

impl Core {
    pub fn new(
        g: Graph,
        pos: Vec<Option<(f64, f64)>>,
        dirs: Vec<u8>,
        patch_ports: Vec<Vec<(u8, Vec<u32>)>>,
        patch_pos: Vec<Option<(f64, f64)>>,
        bound: bool,
    ) -> Core {
        let n = g.len();
        Core { g, bound, pos, dirs, patch_ports, patch_pos, orient: HashMap::new(),
               static_dist: HashMap::new(), ws: Workspace::new(n) }
    }

    pub fn distances(&mut self, source: u32) -> &Vec<f64> {
        if !self.static_dist.contains_key(&source) {
            let d = all_distances(&self.g, &mut self.ws, source);
            self.static_dist.insert(source, d);
        }
        &self.static_dist[&source]
    }

    fn ports(&self, patch: u32, ptype: u8) -> &[u32] {
        self.patch_ports[patch as usize]
            .iter()
            .find(|(t, _)| *t == ptype)
            .map(|(_, p)| p.as_slice())
            .unwrap_or(&[])
    }

    /// The data port of ``patch`` of type ``ptype`` facing ``magic``.
    fn oriented(&mut self, patch: u32, ptype: u8, magic: u32) -> Option<u32> {
        if let Some(&p) = self.orient.get(&(patch, ptype, magic)) {
            return p;
        }
        let ports = self.ports(patch, ptype).to_vec();
        let choice = if ports.is_empty() {
            None
        } else if ports.len() == 1 {
            Some(ports[0])
        } else {
            match (self.pos[magic as usize], self.patch_pos[patch as usize]) {
                (Some(m), Some(d)) => {
                    let (dx, dy) = (m.0 - d.0, m.1 - d.1);
                    let prefs: &[u8] = if dx < 0.0 && dy > 0.0 {
                        &[DIR_N, DIR_W]
                    } else if dx > 0.0 && dy > 0.0 {
                        &[DIR_N, DIR_E]
                    } else if dx < 0.0 && dy < 0.0 {
                        &[DIR_S, DIR_W]
                    } else if dx > 0.0 && dy < 0.0 {
                        &[DIR_S, DIR_E]
                    } else if dx == 0.0 && dy > 0.0 {
                        &[DIR_N]
                    } else if dx == 0.0 && dy < 0.0 {
                        &[DIR_S]
                    } else if dx < 0.0 && dy == 0.0 {
                        &[DIR_W]
                    } else if dx > 0.0 && dy == 0.0 {
                        &[DIR_E]
                    } else {
                        &[DIR_N, DIR_S, DIR_E, DIR_W]
                    };
                    prefs
                        .iter()
                        .find_map(|&d| ports.iter().copied().find(|&p| self.dirs[p as usize] & d != 0))
                        .or(Some(ports[0]))
                }
                _ => Some(ports[0]),
            }
        };
        self.orient.insert((patch, ptype, magic), choice);
        choice
    }

    /// The data ports of ``specs`` oriented towards ``magic``.
    fn complete(&mut self, specs: &[(u32, Vec<u8>)], magic: u32) -> Vec<u32> {
        let mut out = Vec::new();
        for (patch, types) in specs {
            for &t in types {
                if let Some(p) = self.oriented(*patch, t, magic) {
                    out.push(p);
                }
            }
        }
        out
    }

    /// Every port the data part of a choice may use (the nearest-terminal targets).
    fn choice_targets(&self, specs: &[(u32, Vec<u8>)]) -> Vec<u32> {
        let mut out = Vec::new();
        for (patch, types) in specs {
            for &t in types {
                out.extend_from_slice(self.ports(*patch, t));
            }
        }
        out
    }

    fn static_reach(&mut self, m: u32, targets: &[u32]) -> f64 {
        let d = self.distances(m);
        targets.iter().map(|&t| d[t as usize]).fold(INF, f64::min)
    }

    /// ``(terminals, magic)`` of a set: a choice is replaced by its magic port
    /// (``LayoutEngine._resolve_terminals``).  ``cost`` / ``blocked`` are
    /// ``None`` when there is no congestion cost / blocked node.
    fn resolve(
        &mut self,
        set: &SetSpec,
        initial: Option<u32>,
        taken: &HashSet<u32>,
        cost: Option<&dyn Fn(u32) -> f64>,
        blocked: Option<&dyn Fn(u32) -> bool>,
    ) -> Result<(Vec<u32>, Option<u32>), String> {
        let (candidates, specs) = match set {
            SetSpec::Fixed(t) => return Ok((t.clone(), None)),
            SetSpec::Choice { candidates, specs, .. } => (candidates, specs),
        };
        let sources: Vec<u32> = candidates
            .iter()
            .copied()
            .filter(|c| !taken.contains(c) && blocked.map_or(true, |b| !b(*c)))
            .collect();
        if sources.is_empty() {
            return Err("no free magic terminal left this step".into());
        }
        let targets = self.choice_targets(specs);
        let magic = if cost.is_none() && blocked.is_none() {
            match initial.filter(|i| sources.contains(i)) {
                Some(i) => i,
                None => {
                    let mut best = (INF, sources[0]);
                    for &m in &sources {
                        let d = self.static_reach(m, &targets);
                        if d < best.0 {
                            best = (d, m);
                        }
                    }
                    if best.0 >= INF {
                        return Err("no free magic terminal reaches the product".into());
                    }
                    best.1
                }
            }
        } else {
            let source_set: HashSet<u32> = sources.iter().copied().collect();
            let zero = |_: u32| 0.0;
            let open = |_: u32| false;
            let c: &dyn Fn(u32) -> f64 = cost.unwrap_or(&zero);
            let b: &dyn Fn(u32) -> bool = blocked.unwrap_or(&open);
            match nearest_source(&self.g, &mut self.ws, &|v| source_set.contains(&v), &targets, &c, &b) {
                Some(m) => m,
                None => return Err("no free magic terminal reaches the product".into()),
            }
        };
        let mut concrete = vec![magic];
        concrete.extend(self.complete(specs, magic));
        Ok((concrete, Some(magic)))
    }

    fn tree(
        &mut self,
        terminals: &[u32],
        cost: Option<&dyn Fn(u32) -> f64>,
        blocked: Option<&dyn Fn(u32) -> bool>,
    ) -> Result<Tree, String> {
        let zero = |_: u32| 0.0;
        let open = |_: u32| false;
        let c: &dyn Fn(u32) -> f64 = cost.unwrap_or(&zero);
        let b: &dyn Fn(u32) -> bool = blocked.unwrap_or(&open);
        steiner_tree(&self.g, &mut self.ws, terminals, self.bound, &c, &b)
    }

    /// Public single Steiner tree (tests, retry pass).
    pub fn steiner(
        &mut self,
        terminals: &[u32],
        cost: Option<&dyn Fn(u32) -> f64>,
        blocked: Option<&dyn Fn(u32) -> bool>,
    ) -> Result<Tree, String> {
        self.tree(terminals, cost, blocked)
    }

    fn routing_cells(&self, nodes: &[u32], set: &SetSpec) -> Vec<u32> {
        let fixed: &[u32] = match set {
            SetSpec::Fixed(t) => t,
            SetSpec::Choice { .. } => &[],
        };
        nodes.iter().copied().filter(|&n| self.g.is_cell(n) && !fixed.contains(&n)).collect()
    }

    fn order(&self, sets: &[SetSpec], priorities: &Option<Vec<f64>>, greedy_order: &str) -> Vec<usize> {
        let mut order: Vec<usize> = (0..sets.len()).collect();
        if let Some(p) = priorities {
            order.sort_by(|&a, &b| p[b].total_cmp(&p[a]).then(sets[a].len().cmp(&sets[b].len())));
        } else if greedy_order == "min_size" {
            order.sort_by_key(|&i| sets[i].len());
        } else if greedy_order == "max_size" {
            order.sort_by(|&a, &b| sets[b].len().cmp(&sets[a].len()));
        }
        order
    }

    /// Starting magic port per choice set: the min-cost matching of the first
    /// ``len(candidates)`` choice sets with options to the candidates, by
    /// uncongested distance (``scheduler._assign_terminals_matching``).
    fn match_initial(&mut self, sets: &[SetSpec]) -> Vec<Option<u32>> {
        let mut initial: Vec<Option<u32>> = sets
            .iter()
            .map(|s| match s {
                SetSpec::Choice { initial, .. } => *initial,
                _ => None,
            })
            .collect();
        let cands: Vec<u32> = match sets.iter().find_map(|s| match s {
            SetSpec::Choice { candidates, .. } => Some(candidates.clone()),
            _ => None,
        }) {
            Some(c) if !c.is_empty() => c,
            _ => return initial,
        };
        let unreachable = 1e6;
        let mut rows: Vec<(usize, Vec<bool>, Vec<f64>)> = Vec::new();
        for (si, set) in sets.iter().enumerate() {
            if rows.len() == cands.len() {
                break;
            }
            let specs = match set {
                SetSpec::Choice { specs, .. } => specs,
                _ => continue,
            };
            let mut has = Vec::with_capacity(cands.len());
            let mut costs = Vec::with_capacity(cands.len());
            for &t in &cands {
                let qts = self.complete(specs, t);
                if qts.is_empty() {
                    has.push(false);
                    costs.push(10.0 * unreachable);
                } else {
                    let d = self.distances(t);
                    has.push(true);
                    costs.push(qts.iter().map(|&p| {
                        let x = d[p as usize];
                        if x >= INF { unreachable } else { x }
                    }).sum());
                }
            }
            if has.iter().any(|&h| h) {
                rows.push((si, has, costs));
            }
        }
        if rows.is_empty() {
            return initial;
        }
        let matrix: Vec<Vec<f64>> = rows.iter().map(|r| r.2.clone()).collect();
        for (r, c) in assign(&matrix).into_iter().enumerate() {
            if c != usize::MAX && rows[r].1[c] {
                initial[rows[r].0] = Some(cands[c]);
            }
        }
        initial
    }

    /// ``LayoutEngine.steiner_packing_harvest``.
    pub fn pack_harvest(&mut self, sets: &[SetSpec], p: &Params) -> (Vec<SetResult>, Stats) {
        let n = self.g.len();
        let mut stats = Stats { stopped_reason: "max_iters".into(), ..Default::default() };
        let order = self.order(sets, &p.priorities, &p.greedy_order);
        let initial: Vec<Option<u32>> = if p.match_initial {
            self.match_initial(sets)
        } else {
            sets.iter().map(|s| match s {
                SetSpec::Choice { initial, .. } => *initial,
                _ => None,
            }).collect()
        };
        let cap = p.capacity;
        let mut routes: Vec<Option<Route>> = (0..sets.len()).map(|_| None).collect();
        let mut usage = vec![0i32; n];
        let mut hist = vec![0f64; n];
        let mut touched: Vec<u32> = Vec::new(); // nodes whose usage/history ever changed
        let mut is_touched = vec![false; n];

        let held = |routes: &Vec<Option<Route>>, skip: Option<usize>, only: Option<&HashSet<usize>>| {
            routes.iter().enumerate()
                .filter(|(j, r)| Some(*j) != skip && only.map_or(true, |o| o.contains(j))
                        && r.as_ref().map_or(false, |r| r.success))
                .filter_map(|(_, r)| r.as_ref().unwrap().magic)
                .collect::<HashSet<u32>>()
        };

        // --- initial routes ---
        for &i in &order {
            let taken = held(&routes, None, None);
            let route = match self.resolve(&sets[i], initial[i], &taken, None, None) {
                Ok((concrete, magic)) => match self.tree(&concrete, None, None) {
                    Ok(t) => {
                        let rnodes = self.routing_cells(&t.nodes, &sets[i]);
                        for &v in &rnodes {
                            usage[v as usize] += 1;
                            if !is_touched[v as usize] {
                                is_touched[v as usize] = true;
                                touched.push(v);
                            }
                        }
                        stats.initial_routes += 1;
                        Route { success: true, concrete: Some(concrete), magic, nodes: t.nodes,
                                edges: t.edges, rnodes, error: None }
                    }
                    Err(e) => Route::failed(e),
                },
                Err(e) => Route::failed(e),
            };
            routes[i] = Some(route);
        }

        // --- negotiation ---
        let mut best_score: Option<i64> = None;
        let mut stagnation = 0usize;
        for _ in 0..p.max_iters {
            let conflict: i64 = touched.iter().map(|&v| (usage[v as usize] - cap).max(0) as i64).sum();
            if conflict == 0 {
                stats.stopped_reason = "no_conflicts".into();
                break;
            }
            match best_score {
                None => best_score = Some(conflict),
                Some(b) if conflict < b => {
                    best_score = Some(conflict);
                    stagnation = 0;
                }
                Some(_) => {
                    stagnation += 1;
                    if stagnation >= p.stagnation_limit {
                        stats.stopped_reason = "stagnation".into();
                        break;
                    }
                }
            }
            stats.iterations += 1;
            let over: HashSet<u32> = touched.iter().copied().filter(|&v| usage[v as usize] > cap).collect();
            for &i in &order {
                let conflicted = match &routes[i] {
                    Some(r) if r.success => r.rnodes.iter().any(|v| over.contains(v)),
                    _ => false,
                };
                if !conflicted {
                    continue;
                }
                stats.conflicted_reroutes += 1;
                stats.reroutes += 1;
                for &v in &routes[i].as_ref().unwrap().rnodes {
                    usage[v as usize] -= 1;
                }
                let taken = held(&routes, Some(i), None);
                let has_cost = touched.iter().any(|&v| usage[v as usize] > 0 || hist[v as usize] > 0.0);
                let result = {
                    let (u, h, alpha, beta, capf) = (&usage, &hist, p.alpha, p.beta, cap as f64);
                    let cost = move |v: u32| {
                        let (c, hv) = (u[v as usize], h[v as usize]);
                        let mut x = 0.0;
                        if c > 0 {
                            x += alpha * (c as f64 / capf);
                        }
                        if hv > 0.0 {
                            x += beta * hv;
                        }
                        x
                    };
                    let cost_ref: Option<&dyn Fn(u32) -> f64> = if has_cost { Some(&cost) } else { None };
                    match self.resolve(&sets[i], initial[i], &taken, cost_ref, None) {
                        Ok((concrete, magic)) => self.tree(&concrete, cost_ref, None).map(|t| (concrete, magic, t)),
                        Err(e) => Err(e),
                    }
                };
                routes[i] = Some(match result {
                    Ok((concrete, magic, t)) => {
                        let rnodes = self.routing_cells(&t.nodes, &sets[i]);
                        for &v in &rnodes {
                            usage[v as usize] += 1;
                            if !is_touched[v as usize] {
                                is_touched[v as usize] = true;
                                touched.push(v);
                            }
                        }
                        Route { success: true, concrete: Some(concrete), magic, nodes: t.nodes,
                                edges: t.edges, rnodes, error: None }
                    }
                    Err(e) => Route::failed(e),
                });
            }
            for &v in &touched {
                let c = usage[v as usize];
                if c > cap {
                    hist[v as usize] += (c - cap) as f64;
                }
            }
        }

        // --- drop routes until every node is within capacity ---
        let overfull = |usage: &Vec<i32>| -> HashSet<u32> {
            touched.iter().copied().filter(|&v| usage[v as usize] > cap).collect()
        };
        let mut over = overfull(&usage);
        let mut dropped: HashSet<usize> = HashSet::new();
        if !over.is_empty() && p.no_defer {
            stats.no_defer_fired = true;
            over.clear();
        }
        while !over.is_empty() {
            let mut best_i: Option<usize> = None;
            let mut best: Vec<f64> = vec![-1.0, -1.0];
            for &i in &order {
                if dropped.contains(&i) {
                    continue;
                }
                let r = match &routes[i] {
                    Some(r) if r.success => r,
                    _ => continue,
                };
                let conflicts = r.rnodes.iter().filter(|v| over.contains(v)).count() as f64;
                let size = r.rnodes.len() as f64;
                let score = match &p.priorities {
                    Some(pr) if conflicts > 0.0 => vec![1.0, -pr[i], conflicts, size],
                    _ => vec![conflicts, size],
                };
                if tuple_gt(&score, &best) {
                    best = score;
                    best_i = Some(i);
                }
            }
            let bi = match best_i {
                Some(bi) if best[0] > 0.0 => bi,
                _ => break,
            };
            for &v in &routes[bi].as_ref().unwrap().rnodes {
                usage[v as usize] -= 1;
            }
            dropped.insert(bi);
            over = overfull(&usage);
        }

        // --- greedy fill of dropped / unrouted sets in the leftover space ---
        if p.greedy_fill {
            let mut committed = vec![false; n];
            let mut n_committed = 0usize;
            let mut kept: HashSet<usize> = HashSet::new();
            for &i in &order {
                if let Some(r) = &routes[i] {
                    if r.success && !dropped.contains(&i) {
                        for &v in &r.rnodes {
                            if !committed[v as usize] {
                                committed[v as usize] = true;
                                n_committed += 1;
                            }
                        }
                        kept.insert(i);
                    }
                }
            }
            for &i in &order {
                if kept.contains(&i) {
                    continue;
                }
                let taken = held(&routes, None, Some(&kept));
                let result = {
                    let c = &committed;
                    let blocked = move |v: u32| c[v as usize];
                    let blocked_ref: Option<&dyn Fn(u32) -> bool> =
                        if n_committed > 0 { Some(&blocked) } else { None };
                    match self.resolve(&sets[i], initial[i], &taken, None, blocked_ref) {
                        Ok((concrete, magic)) => self.tree(&concrete, None, blocked_ref).map(|t| (concrete, magic, t)),
                        Err(e) => Err(e),
                    }
                };
                if let Ok((concrete, magic, t)) = result {
                    let rnodes = self.routing_cells(&t.nodes, &sets[i]);
                    for &v in &rnodes {
                        if !committed[v as usize] {
                            committed[v as usize] = true;
                            n_committed += 1;
                        }
                    }
                    routes[i] = Some(Route { success: true, concrete: Some(concrete), magic,
                                             nodes: t.nodes, edges: t.edges, rnodes, error: None });
                    kept.insert(i);
                    dropped.remove(&i);
                    stats.filled += 1;
                }
            }
        }

        stats.dropped = dropped.len();
        stats.final_conflict_score = touched.iter().map(|&v| (usage[v as usize] - cap).max(0) as i64).sum();
        let results = routes
            .into_iter()
            .enumerate()
            .map(|(i, r)| {
                let r = r.unwrap_or_else(|| Route::failed("unrouted".into()));
                if r.success && !dropped.contains(&i) {
                    SetResult { success: true, terminals: r.concrete, nodes: r.nodes, edges: r.edges, error: None }
                } else {
                    let error = r.error.clone().unwrap_or_else(|| {
                        if dropped.contains(&i) { "dropped due to congestion".into() } else { "unrouted".into() }
                    });
                    SetResult { success: false, terminals: r.concrete, nodes: vec![], edges: vec![], error: Some(error) }
                }
            })
            .collect();
        (results, stats)
    }

    /// ``LayoutEngine.steiner_packing`` (greedy, no negotiation).
    pub fn pack_greedy(&mut self, sets: &[SetSpec], greedy_order: &str, match_initial: bool) -> Vec<SetResult> {
        let n = self.g.len();
        let order = self.order(sets, &None, greedy_order);
        let initial: Vec<Option<u32>> = if match_initial {
            self.match_initial(sets)
        } else {
            sets.iter().map(|s| match s {
                SetSpec::Choice { initial, .. } => *initial,
                _ => None,
            }).collect()
        };
        let mut used = vec![false; n];
        let mut n_used = 0usize;
        let mut taken: HashSet<u32> = HashSet::new();
        let mut results: Vec<Option<SetResult>> = (0..sets.len()).map(|_| None).collect();
        for &i in &order {
            let (resolved, tree) = {
                let u = &used;
                let blocked = move |v: u32| u[v as usize];
                let blocked_ref: Option<&dyn Fn(u32) -> bool> = if n_used > 0 { Some(&blocked) } else { None };
                match self.resolve(&sets[i], initial[i], &taken, None, None) {
                    Ok((concrete, magic)) => {
                        let t = self.tree(&concrete, None, blocked_ref);
                        (Some((concrete, magic)), t)
                    }
                    Err(e) => (None, Err(e)),
                }
            };
            results[i] = Some(match tree {
                Ok(t) => {
                    let (concrete, magic) = resolved.unwrap();
                    if let Some(m) = magic {
                        taken.insert(m);
                    }
                    for &v in &t.nodes {
                        if self.g.is_cell(v) && !used[v as usize] {
                            used[v as usize] = true;
                            n_used += 1;
                        }
                    }
                    SetResult { success: true, terminals: Some(concrete), nodes: t.nodes, edges: t.edges, error: None }
                }
                Err(e) => SetResult { success: false, terminals: resolved.map(|r| r.0), nodes: vec![],
                                      edges: vec![], error: Some(e) },
            });
        }
        results.into_iter().map(|r| r.unwrap()).collect()
    }
}
