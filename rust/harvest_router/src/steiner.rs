//! Approximate Steiner tree (``LayoutEngine.steiner_tree``): shortest paths
//! between the terminals, a Prim MST on their metric closure, the MST edges
//! expanded into graph paths, and cycles pruned Kruskal-style until every
//! terminal is connected.

use std::collections::HashMap;

use crate::search::{search, Graph, Workspace, INF};

pub struct Tree {
    pub nodes: Vec<u32>,
    pub edges: Vec<(u32, u32)>,
}

struct UnionFind {
    parent: HashMap<u32, u32>,
    rank: HashMap<u32, u32>,
}

impl UnionFind {
    fn new(nodes: &[u32]) -> Self {
        UnionFind {
            parent: nodes.iter().map(|&n| (n, n)).collect(),
            rank: nodes.iter().map(|&n| (n, 0)).collect(),
        }
    }

    fn find(&mut self, x: u32) -> u32 {
        let mut x = x;
        loop {
            let p = self.parent[&x];
            if p == x {
                return x;
            }
            let gp = self.parent[&p];
            self.parent.insert(x, gp);
            x = gp;
        }
    }

    fn union(&mut self, a: u32, b: u32) -> bool {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra == rb {
            return false;
        }
        let (ka, kb) = (self.rank[&ra], self.rank[&rb]);
        if ka < kb {
            self.parent.insert(ra, rb);
        } else if ka > kb {
            self.parent.insert(rb, ra);
        } else {
            self.parent.insert(rb, ra);
            self.rank.insert(ra, ka + 1);
        }
        true
    }
}

/// Steiner tree connecting ``terminals`` (``Err`` if they are not mutually
/// reachable); ``cost`` and ``blocked`` as in :func:`search`.
pub fn steiner_tree<C, B>(
    g: &Graph,
    ws: &mut Workspace,
    terminals: &[u32],
    bound: bool,
    cost: &C,
    blocked: &B,
) -> Result<Tree, String>
where
    C: Fn(u32) -> f64,
    B: Fn(u32) -> bool,
{
    let mut terms: Vec<u32> = Vec::with_capacity(terminals.len());
    for &t in terminals {
        if !terms.contains(&t) {
            terms.push(t);
        }
    }
    let k = terms.len();
    if k == 0 {
        return Ok(Tree { nodes: vec![], edges: vec![] });
    }
    if k == 1 {
        return Ok(Tree { nodes: vec![terms[0]], edges: vec![] });
    }

    // Metric closure: distances and paths between every terminal pair.
    let mut dist = vec![vec![INF; k]; k];
    let mut paths: Vec<Vec<Option<Vec<u32>>>> = vec![vec![None; k]; k];
    for i in 0..k {
        search(g, ws, terms[i], &terms, bound, cost, blocked);
        for j in 0..k {
            let d = ws.dist(terms[j]);
            if d < INF {
                dist[i][j] = d;
                if i != j {
                    paths[i][j] = ws.path_to(terms[i], terms[j]);
                }
            }
        }
    }

    // Prim MST over the terminals (members in insertion order).
    let mut in_mst = vec![false; k];
    let mut members = vec![0usize];
    in_mst[0] = true;
    let mut mst: Vec<(usize, usize)> = Vec::with_capacity(k - 1);
    while members.len() < k {
        let mut best = (INF, usize::MAX, usize::MAX);
        for &u in &members {
            for v in 0..k {
                if in_mst[v] {
                    continue;
                }
                let d = dist[u][v];
                if d < INF && d < best.0 {
                    best = (d, u, v);
                }
            }
        }
        if best.1 == usize::MAX {
            return Err("Terminals are not mutually reachable in the routing graph.".into());
        }
        mst.push((best.1, best.2));
        in_mst[best.2] = true;
        members.push(best.2);
    }

    // Expand the MST edges into graph paths (undirected edges, min weight,
    // first-insertion order).
    let mut sol_nodes: Vec<u32> = Vec::new();
    let mut in_sol: HashMap<u32, ()> = HashMap::new();
    let mut edge_pos: HashMap<(u32, u32), usize> = HashMap::new();
    let mut edges: Vec<((u32, u32), f64)> = Vec::new();
    for &(u, v) in &mst {
        let path = match (&paths[u][v], &paths[v][u]) {
            (Some(p), _) => p.clone(),
            (None, Some(p)) => p.clone(),
            _ => return Err("No path between terminals in routing graph.".into()),
        };
        for &n in &path {
            if in_sol.insert(n, ()).is_none() {
                sol_nodes.push(n);
            }
        }
        for w in path.windows(2) {
            let (a, b) = (w[0], w[1]);
            let weight = match g.weight(a, b) {
                Some(x) => x + cost(b),
                None => g.weight(b, a).map(|x| x + cost(a)).unwrap_or(1.0),
            };
            let key = if a <= b { (a, b) } else { (b, a) };
            match edge_pos.get(&key) {
                Some(&i) => {
                    if weight < edges[i].1 {
                        edges[i].1 = weight;
                    }
                }
                None => {
                    edge_pos.insert(key, edges.len());
                    edges.push((key, weight));
                }
            }
        }
    }

    // Prune cycles: Kruskal over the union edges until the terminals connect.
    let mut uf = UnionFind::new(&sol_nodes);
    let mut order: Vec<usize> = (0..edges.len()).collect();
    order.sort_by(|&i, &j| edges[i].1.total_cmp(&edges[j].1)); // stable
    let present: Vec<u32> = terms.iter().copied().filter(|t| in_sol.contains_key(t)).collect();
    let mut chosen: Vec<(u32, u32)> = Vec::new();
    for i in order {
        let (a, b) = edges[i].0;
        if uf.union(a, b) {
            chosen.push((a, b));
            let r0 = uf.find(terms[0]);
            if present.iter().all(|&t| uf.find(t) == r0) {
                break;
            }
        }
    }
    let mut nodes: Vec<u32> = Vec::new();
    let mut seen: HashMap<u32, ()> = HashMap::new();
    for &(a, b) in &chosen {
        for n in [a, b] {
            if seen.insert(n, ()).is_none() {
                nodes.push(n);
            }
        }
    }
    for &t in &present {
        if seen.insert(t, ()).is_none() {
            nodes.push(t);
        }
    }
    Ok(Tree { nodes, edges: chosen })
}
