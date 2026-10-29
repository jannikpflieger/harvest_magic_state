//! Shortest-path searches on the routing graph (``LayoutEngine._dijkstra``,
//! ``_astar`` and ``_nearest_source`` in ``harvest/layout/engine.py``).
//!
//! Node costs (congestion) are added to every edge entering a node, blocked
//! nodes are treated as absent, and the heap breaks ties first-in first-out,
//! as in the Python versions.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

pub const INF: f64 = 1e18;
pub const NONE: u32 = u32::MAX;

/// Routing graph: ``adj[u] = [(v, weight)]``; ``cell[u]`` is ``Some((x, y))``
/// for a routing cell and ``None`` for a port node.
pub struct Graph {
    pub adj: Vec<Vec<(u32, f64)>>,
    pub cell: Vec<Option<(i32, i32)>>,
}

impl Graph {
    pub fn len(&self) -> usize {
        self.adj.len()
    }

    pub fn is_cell(&self, v: u32) -> bool {
        self.cell[v as usize].is_some()
    }

    /// Routing cells through which ``v`` is entered (itself, or a port's cells).
    pub fn entry_cells(&self, v: u32) -> Vec<(i32, i32)> {
        match self.cell[v as usize] {
            Some(c) => vec![c],
            None => self.adj[v as usize]
                .iter()
                .filter_map(|&(w, _)| self.cell[w as usize])
                .collect(),
        }
    }

    /// Weight of edge ``u -> v`` (``None`` if absent).
    pub fn weight(&self, u: u32, v: u32) -> Option<f64> {
        self.adj[u as usize].iter().find(|&&(w, _)| w == v).map(|&(_, x)| x)
    }
}

#[derive(Clone, Copy)]
struct Entry {
    key: f64,
    tie: u64,
    g: f64,
    node: u32,
}

impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Entry {}
impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Entry {
    // BinaryHeap is a max-heap: reverse to pop the smallest (key, tie) first.
    fn cmp(&self, other: &Self) -> Ordering {
        other.key.total_cmp(&self.key).then_with(|| other.tie.cmp(&self.tie))
    }
}

/// Reusable search buffers (distance/predecessor arrays reset by a stamp).
pub struct Workspace {
    dist: Vec<f64>,
    prev: Vec<u32>,
    stamp: Vec<u32>,
    cur: u32,
    target: Vec<u32>, // node -> target index, NONE otherwise (reset after a search)
    heap: BinaryHeap<Entry>,
}

impl Workspace {
    pub fn new(n: usize) -> Self {
        Workspace {
            dist: vec![INF; n],
            prev: vec![NONE; n],
            stamp: vec![0; n],
            cur: 0,
            target: vec![NONE; n],
            heap: BinaryHeap::new(),
        }
    }

    fn reset(&mut self) {
        self.cur = self.cur.wrapping_add(1);
        if self.cur == 0 {
            self.stamp.iter_mut().for_each(|s| *s = 0);
            self.cur = 1;
        }
        self.heap.clear();
    }

    #[inline]
    pub fn dist(&self, v: u32) -> f64 {
        if self.stamp[v as usize] == self.cur {
            self.dist[v as usize]
        } else {
            INF
        }
    }

    #[inline]
    fn set(&mut self, v: u32, d: f64, p: u32) {
        let i = v as usize;
        self.stamp[i] = self.cur;
        self.dist[i] = d;
        self.prev[i] = p;
    }

    /// Path ``source -> v`` of the last search (``None`` if ``v`` was not reached).
    pub fn path_to(&self, source: u32, v: u32) -> Option<Vec<u32>> {
        if source == v {
            return Some(vec![v]);
        }
        if self.dist(v) >= INF {
            return None;
        }
        let mut path = vec![v];
        let mut cur = v;
        while cur != source {
            let p = self.prev[cur as usize];
            if p == NONE || self.stamp[cur as usize] != self.cur {
                return None;
            }
            path.push(p);
            cur = p;
        }
        path.reverse();
        Some(path)
    }
}

/// Distance lower bound to the bounding box of the open goal cells.
struct GoalBox {
    x0: i32,
    x1: i32,
    y0: i32,
    y1: i32,
}

impl GoalBox {
    fn of(cells: &[(i32, i32)]) -> GoalBox {
        let mut b = GoalBox { x0: i32::MAX, x1: i32::MIN, y0: i32::MAX, y1: i32::MIN };
        for &(x, y) in cells {
            b.x0 = b.x0.min(x);
            b.x1 = b.x1.max(x);
            b.y0 = b.y0.min(y);
            b.y1 = b.y1.max(y);
        }
        b
    }

    fn same(&self, o: &GoalBox) -> bool {
        self.x0 == o.x0 && self.x1 == o.x1 && self.y0 == o.y0 && self.y1 == o.y1
    }

    #[inline]
    fn h(&self, cell: Option<(i32, i32)>) -> f64 {
        match cell {
            None => 0.0,
            Some((x, y)) => {
                let dx = if x < self.x0 { self.x0 - x } else if x > self.x1 { x - self.x1 } else { 0 };
                let dy = if y < self.y0 { self.y0 - y } else if y > self.y1 { y - self.y1 } else { 0 };
                (dx + dy) as f64
            }
        }
    }
}

/// Search from ``source`` until every target is settled (or the graph is
/// exhausted); afterwards ``ws.dist(t)`` is exact for every reached target and
/// ``ws.path_to(source, t)`` gives a shortest path.
///
/// With ``bound`` the search is A* towards the bounding box of the open
/// targets' entry cells (``LayoutEngine._astar``); otherwise plain Dijkstra.
pub fn search<C, B>(
    g: &Graph,
    ws: &mut Workspace,
    source: u32,
    targets: &[u32],
    bound: bool,
    cost: &C,
    blocked: &B,
) where
    C: Fn(u32) -> f64,
    B: Fn(u32) -> bool,
{
    ws.reset();
    let mut remaining = 0usize;
    for (i, &t) in targets.iter().enumerate() {
        if ws.target[t as usize] == NONE {
            ws.target[t as usize] = i as u32;
            remaining += 1;
        }
    }
    let mut open = vec![true; targets.len()];
    // Goal cells per target (the source needs none: it is settled first).
    let goals: Vec<Vec<(i32, i32)>> = if bound {
        targets.iter().map(|&t| if t == source { Vec::new() } else { g.entry_cells(t) }).collect()
    } else {
        Vec::new()
    };
    let mut cells: Vec<(i32, i32)> = goals.iter().flatten().copied().collect();
    let use_bound = bound && !cells.is_empty();
    let mut gbox = GoalBox::of(&cells);

    ws.set(source, 0.0, NONE);
    let mut tie: u64 = 0;
    let h0 = if use_bound { gbox.h(g.cell[source as usize]) } else { 0.0 };
    ws.heap.push(Entry { key: h0, tie: 0, g: 0.0, node: source });

    while let Some(e) = ws.heap.pop() {
        let u = e.node;
        if e.g != ws.dist(u) {
            continue;
        }
        let ti = ws.target[u as usize];
        if ti != NONE && open[ti as usize] {
            open[ti as usize] = false;
            remaining -= 1;
            if remaining == 0 {
                break;
            }
            if use_bound {
                let left: Vec<(i32, i32)> = targets
                    .iter()
                    .enumerate()
                    .filter(|&(i, _)| open[i])
                    .flat_map(|(i, _)| goals[i].iter().copied())
                    .collect();
                if left.is_empty() {
                    break; // open targets without an entry cell are unreachable
                }
                if left.len() < cells.len() {
                    cells = left;
                    let nb = GoalBox::of(&cells);
                    if !nb.same(&gbox) {
                        gbox = nb;
                        let old: Vec<Entry> = ws.heap.drain().collect();
                        for mut x in old {
                            x.key = x.g + gbox.h(g.cell[x.node as usize]);
                            ws.heap.push(x);
                        }
                    }
                }
            }
        }
        let d = e.g;
        for &(v, w) in &g.adj[u as usize] {
            if blocked(v) {
                continue;
            }
            let nd = d + w + cost(v);
            if nd < ws.dist(v) {
                ws.set(v, nd, u);
                tie += 1;
                let key = if use_bound { nd + gbox.h(g.cell[v as usize]) } else { nd };
                ws.heap.push(Entry { key, tie, g: nd, node: v });
            }
        }
    }
    for &t in targets {
        ws.target[t as usize] = NONE;
    }
}

/// Uncongested distance from ``source`` to every node (``None`` = unreachable).
pub fn all_distances(g: &Graph, ws: &mut Workspace, source: u32) -> Vec<f64> {
    search(g, ws, source, &[], false, &|_| 0.0, &|_| false);
    (0..g.len() as u32).map(|v| ws.dist(v)).collect()
}

/// The source closest to any target, searched backwards from the targets
/// (``LayoutEngine._nearest_source``); ``None`` if no source is reachable.
pub fn nearest_source<C, B>(
    g: &Graph,
    ws: &mut Workspace,
    is_source: &dyn Fn(u32) -> bool,
    targets: &[u32],
    cost: &C,
    blocked: &B,
) -> Option<u32>
where
    C: Fn(u32) -> f64,
    B: Fn(u32) -> bool,
{
    ws.reset();
    let mut tie: u64 = 0;
    for &t in targets {
        let d0 = cost(t);
        if d0 < ws.dist(t) {
            ws.set(t, d0, NONE);
            ws.heap.push(Entry { key: d0, tie, g: d0, node: t });
            tie += 1;
        }
    }
    while let Some(e) = ws.heap.pop() {
        let u = e.node;
        if e.g != ws.dist(u) {
            continue;
        }
        if is_source(u) {
            return Some(u);
        }
        for &(v, w) in &g.adj[u as usize] {
            if blocked(v) {
                continue;
            }
            let nd = e.g + w + cost(v);
            if nd < ws.dist(v) {
                ws.set(v, nd, u);
                tie += 1;
                ws.heap.push(Entry { key: nd, tie, g: nd, node: v });
            }
        }
    }
    None
}
