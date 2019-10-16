pub mod walker;

use bumpalo::Bump;
use derivative::Derivative;
use smallvec::SmallVec;
use std::fmt::Debug;
pub use walker::Walker;

#[derive(Debug, Clone, PartialEq)]
struct Node<T> {
    inner: T,
    incoming: SmallVec<[EdgeIndex; 8]>,
    outgoing: SmallVec<[EdgeIndex; 8]>,
}
impl<T> Node<T> {
    pub fn new(inner: T) -> Self {
        Self {
            inner,
            incoming: SmallVec::default(),
            outgoing: SmallVec::default(),
        }
    }

    #[inline]
    pub fn edges_mut(&mut self, direction: Direction) -> &mut SmallVec<[EdgeIndex; 8]> {
        match direction {
            Direction::Incoming => &mut self.incoming,
            Direction::Outgoing => &mut self.outgoing,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct EdgeNodes {
    from: NodeIndex,
    to: NodeIndex,
}

impl EdgeNodes {
    #[inline]
    fn get_dir(&self, direction: Direction) -> NodeIndex {
        match direction {
            Direction::Incoming => self.from,
            Direction::Outgoing => self.to,
        }
    }

    #[inline]
    fn set_dir(&mut self, direction: Direction, index: NodeIndex) {
        match direction {
            Direction::Incoming => self.from = index,
            Direction::Outgoing => self.to = index,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct Edge<T> {
    inner: T,
    nodes: EdgeNodes,
}

impl<T> Edge<T> {
    fn new(inner: T, from: NodeIndex, to: NodeIndex) -> Self {
        Self {
            inner,
            nodes: EdgeNodes { from, to },
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum GraphError {
    Cyclical,
    Unknown,
    InvalidIndex,
}

#[derive(Debug, Default, Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct NodeIndex(u32);
impl std::ops::Add for NodeIndex {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }
}

impl NodeIndex {
    pub fn new(value: u32) -> Self {
        Self(value as _)
    }
    pub fn index(&self) -> usize {
        self.0 as _
    }
    pub fn parents(&self) -> ParentsWalker {
        ParentsWalker {
            node: *self,
            next: 0,
        }
    }
    pub fn children(&self) -> ChildrenWalker {
        ChildrenWalker {
            node: *self,
            next: 0,
        }
    }
}

#[derive(Debug, Default, Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct EdgeIndex(u32);
impl std::ops::Add for EdgeIndex {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }
}

impl EdgeIndex {
    pub fn new(value: u32) -> Self {
        Self(value as _)
    }
    pub fn index(&self) -> usize {
        self.0 as _
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum Direction {
    Incoming,
    Outgoing,
}

impl Direction {
    #[inline]
    fn reverse(&self) -> Self {
        match self {
            Direction::Incoming => Direction::Outgoing,
            Direction::Outgoing => Direction::Incoming,
        }
    }
}

#[derive(Derivative, Debug)]
#[derivative(Default(bound = ""))]
struct Nodes<'a, N> {
    inner: SmallVec<[Option<&'a mut Node<N>>; 128]>,
}

impl<'a, N> Nodes<'a, N> {
    fn get(&self, index: NodeIndex) -> Result<&Node<N>, GraphError> {
        self.inner
            .get(index.index())
            .and_then(|n| n.as_ref().map(|n| &**n))
            .ok_or(GraphError::InvalidIndex)
    }

    fn get_mut(&mut self, index: NodeIndex) -> Result<&mut Node<N>, GraphError> {
        self.inner
            .get_mut(index.index())
            .and_then(|n| n.as_mut().map(|n| &mut **n))
            .ok_or(GraphError::InvalidIndex)
    }

    fn get_mut_unchecked(&mut self, index: NodeIndex) -> &mut Node<N> {
        self.inner[index.index()].as_mut().unwrap()
    }

    fn get_unchecked(&self, index: NodeIndex) -> &Node<N> {
        self.inner[index.index()].as_ref().unwrap()
    }

    fn exists(&self, index: NodeIndex) -> bool {
        self.inner.get(index.index()).map_or(false, |n| n.is_some())
    }

    #[inline]
    fn pair_mut(&mut self, first: NodeIndex, second: NodeIndex) -> (&mut Node<N>, &mut Node<N>) {
        if first.index() < second.index() {
            let (left, right) = self.inner.split_at_mut(second.index());
            (
                left[first.index()].as_mut().unwrap(),
                right[0].as_mut().unwrap(),
            )
        } else if first.index() > second.index() {
            let (left, right) = self.inner.split_at_mut(first.index());
            (
                right[0].as_mut().unwrap(),
                left[second.index()].as_mut().unwrap(),
            )
        } else {
            panic!("Asked for pair of mutable references to the same node")
        }
    }
}

#[derive(Derivative, Debug)]
#[derivative(Default(bound = ""))]
struct Edges<'a, E> {
    inner: SmallVec<[Option<&'a mut Edge<E>>; 256]>,
}

impl<'a, E> Edges<'a, E> {
    fn get(&self, index: EdgeIndex) -> Result<&Edge<E>, GraphError> {
        self.inner
            .get(index.index())
            .and_then(|e| e.as_ref().map(|e| &**e))
            .ok_or(GraphError::InvalidIndex)
    }

    fn get_mut(&mut self, index: EdgeIndex) -> Result<&mut Edge<E>, GraphError> {
        self.inner
            .get_mut(index.index())
            .and_then(|e| e.as_mut().map(|e| &mut **e))
            .ok_or(GraphError::InvalidIndex)
    }

    fn get_unchecked(&self, index: EdgeIndex) -> &Edge<E> {
        self.inner[index.index()].as_ref().unwrap()
    }

    fn get_mut_unchecked(&mut self, index: EdgeIndex) -> &mut Edge<E> {
        self.inner[index.index()].as_mut().unwrap()
    }

    fn exists(&self, index: EdgeIndex) -> bool {
        self.inner.get(index.index()).map_or(false, |n| n.is_some())
    }
}

#[derive(Derivative, Debug)]
#[derivative(Default(bound = ""))]
pub struct Graph<'a, N, E> {
    nodes: Nodes<'a, N>,
    edges: Edges<'a, E>,
}
impl<'a, N, E> Graph<'a, N, E> {
    pub fn remove_node(&mut self, index: NodeIndex) -> Result<(), GraphError> {
        if !self.nodes.exists(index) {
            return Err(GraphError::InvalidIndex);
        }
        if let Some(node) = &self.nodes.inner[index.index()] {
            for edge_index in node.incoming.iter().chain(&node.outgoing) {
                self.edges.inner[edge_index.index()] = None;
            }
        }
        self.nodes.inner[index.index()] = None;
        Ok(())
    }

    pub fn node_count(&self) -> u32 {
        self.nodes.inner.len() as u32
    }

    pub fn edge_count(&self) -> u32 {
        self.edges.inner.len() as u32
    }

    pub fn insert_node(
        &mut self,
        allocator: &'a GraphAllocator,
        inner: N,
    ) -> Result<NodeIndex, GraphError> {
        let index = NodeIndex(self.nodes.inner.len() as u32);
        self.nodes
            .inner
            .push(Some(allocator.0.alloc(Node::new(inner))));
        Ok(index)
    }

    pub fn insert_edge<F>(
        &mut self,
        _allocator: &'a GraphAllocator,
        _inner: E,
        _from: NodeIndex,
        _to: NodeIndex,
        _cyclic_fn: F,
    ) -> Result<EdgeIndex, GraphError>
    where
        F: FnOnce(&mut Self, E) -> Result<EdgeIndex, GraphError>,
    {
        unimplemented!()
    }

    pub fn insert_edge_unchecked(
        &mut self,
        allocator: &'a GraphAllocator,
        inner: E,
        from: NodeIndex,
        to: NodeIndex,
    ) -> Result<EdgeIndex, GraphError> {
        if !self.nodes.exists(from) || !self.nodes.exists(to) {
            return Err(GraphError::InvalidIndex);
        }

        let index = EdgeIndex(self.edges.inner.len() as u32);
        self.edges
            .inner
            .push(Some(allocator.0.alloc(Edge::new(inner, from, to))));
        self.nodes.get_mut_unchecked(from).outgoing.push(index);
        self.nodes.get_mut_unchecked(to).incoming.push(index);
        Ok(index)
    }

    /// Get two nodes as mutable at the same time.
    /// Node ids must be different.
    pub fn node_pair_mut(
        &mut self,
        first: NodeIndex,
        second: NodeIndex,
    ) -> Result<(&mut N, &mut N), GraphError> {
        if first == second || !self.nodes.exists(first) || !self.nodes.exists(second) {
            return Err(GraphError::InvalidIndex);
        }
        let (first, second) = self.nodes.pair_mut(first, second);
        Ok((&mut first.inner, &mut second.inner))
    }

    /// Modify all children edges of source node to be children of `target` node.
    /// This leaves source node without children.
    ///
    /// This is essentially a fast path special case of `rewire_where`.
    pub fn rewire_children(
        &mut self,
        source: NodeIndex,
        target: NodeIndex,
    ) -> Result<(), GraphError> {
        if !self.nodes.exists(source) || !self.nodes.exists(target) {
            return Err(GraphError::InvalidIndex);
        }

        if source == target {
            return Ok(());
        }

        for edge_index in &self.nodes.get_unchecked(source).outgoing {
            self.edges.get_mut_unchecked(*edge_index).nodes.from = target;
        }

        let (target_node, source_node) = self.nodes.pair_mut(target, source);
        target_node.outgoing.extend(source_node.outgoing.drain());
        Ok(())
    }

    /// Modify incoming or outgoing edges of source node that match a predicate connecting them to `target` node instead.
    /// This leaves source node with edges for which the `predicate` returned false.
    pub fn rewire_where(
        &mut self,
        direction: Direction,
        source: NodeIndex,
        target: NodeIndex,
        mut predicate: impl FnMut(&mut E, NodeIndex) -> bool,
    ) -> Result<(), GraphError> {
        if !self.nodes.exists(source) || !self.nodes.exists(target) {
            return Err(GraphError::InvalidIndex);
        }

        if source == target {
            return Ok(());
        }

        let (target_node, source_node) = self.nodes.pair_mut(target, source);

        let source_vec = source_node.edges_mut(direction);
        let target_vec = target_node.edges_mut(direction);

        // test the edges for rewire and retain ones that don't pass
        let len = source_vec.len();
        let mut del = 0;
        {
            let v = &mut source_vec[..];

            for i in 0..len {
                let edge = self.edges.get_mut_unchecked(v[i]);
                let other_index = edge.nodes.get_dir(direction);
                if predicate(&mut edge.inner, other_index) {
                    // Move edges that passed to the new node.
                    del += 1;
                    edge.nodes.set_dir(direction.reverse(), target);
                    target_vec.push(v[i]);
                } else if del > 0 {
                    v.swap(i - del, i);
                }
            }
        }

        if del > 0 {
            source_vec.truncate(len - del);
        }

        Ok(())
    }

    pub fn update_edge(
        &mut self,
        edge_index: EdgeIndex,
        from: Option<NodeIndex>,
        to: Option<NodeIndex>,
    ) -> Result<(), GraphError> {
        let edge = self.edges.get_mut(edge_index)?;

        let nodes = &self.nodes;
        if from.map_or(false, |from| !nodes.exists(from))
            || to.map_or(false, |to| !nodes.exists(to))
        {
            return Err(GraphError::InvalidIndex);
        }

        if let Some(from) = from {
            let next_node = self.nodes.get_mut_unchecked(from);
            next_node.outgoing.push(edge_index);
            let prev_node = self.nodes.get_mut_unchecked(edge.nodes.from);
            prev_node.outgoing.retain(|i| *i != edge_index);
            edge.nodes.from = from;
        }

        if let Some(to) = to {
            let next_node = self.nodes.get_mut_unchecked(to);
            next_node.incoming.push(edge_index);
            let prev_node = self.nodes.get_mut_unchecked(edge.nodes.to);
            prev_node.incoming.retain(|i| *i != edge_index);
            edge.nodes.to = to;
        }

        Ok(())
    }

    pub fn nodes_iter(&self) -> impl Iterator<Item = &N> {
        self.nodes
            .inner
            .iter()
            .filter_map(|e| e.as_ref().map(|e| &e.inner))
    }
    pub fn nodes_iter_mut<'b: 'a>(&'b mut self) -> impl Iterator<Item = &mut N> + 'a + 'b {
        self.nodes
            .inner
            .iter_mut()
            .filter_map(|e| e.as_mut().map(|e| &mut e.inner))
    }
    pub fn edges_iter(&self) -> impl Iterator<Item = &E> {
        self.edges
            .inner
            .iter()
            .filter_map(|e| e.as_ref().map(|e| &e.inner))
    }
    pub fn edges_iter_mut<'b: 'a>(&'b mut self) -> impl Iterator<Item = &mut E> + 'a + 'b {
        self.edges
            .inner
            .iter_mut()
            .filter_map(|e| e.as_mut().map(|e| &mut e.inner))
    }

    pub fn nodes_indices_iter<'b: 'a>(&'b self) -> impl Iterator<Item = NodeIndex> + 'a {
        (0..self.nodes.inner.len() as u32)
            .map(|i| NodeIndex::new(i))
            .filter(move |i| self.nodes.exists(*i))
    }
    pub fn edges_indices_iter<'b: 'a>(&'b self) -> impl Iterator<Item = EdgeIndex> + 'a {
        (0..self.edges.inner.len() as u32)
            .map(|i| EdgeIndex::new(i))
            .filter(move |i| self.edges.exists(*i))
    }

    // Return false from the visitor to stop this branches traversal
    pub fn visit<V>(&self, node_index: NodeIndex, mut visitor: V) -> Result<(), GraphError>
    where
        V: FnMut(&Graph<N, E>, &N, &E) -> bool,
    {
        self.visit_traverse(node_index, &mut visitor)
    }

    // Return false from the visitor to stop this branches traversal
    fn visit_traverse<V>(&self, node_index: NodeIndex, visitor: &mut V) -> Result<(), GraphError>
    where
        V: FnMut(&Graph<N, E>, &N, &E) -> bool,
    {
        let node = self.nodes.get(node_index)?;
        for edge_index in &node.outgoing {
            let edge = self.edges.get_unchecked(*edge_index);
            if visitor(self, &node.inner, &edge.inner) {
                self.visit_traverse(edge.nodes.to, visitor)?;
            }
        }
        Ok(())
    }

    pub fn iter_children<'b: 'a>(
        &'b self,
        index: NodeIndex,
    ) -> Result<impl Iterator<Item = (&'a E, &'a N)> + 'b, GraphError> {
        let node = self.nodes.get(index)?;
        Ok(node.outgoing.iter().map(move |edge_index| {
            let edge = self.edges.get_unchecked(*edge_index);
            (&edge.inner, &node.inner)
        }))
    }

    pub fn iter_parents<'b: 'a>(
        &'b self,
        index: NodeIndex,
    ) -> Result<impl Iterator<Item = (&'a E, &'a N)> + 'b, GraphError> {
        let node = self.nodes.get(index)?;
        Ok(node.incoming.iter().map(move |edge_index| {
            let edge = self.edges.get_unchecked(*edge_index);
            (&edge.inner, &node.inner)
        }))
    }

    pub fn get_node(&self, index: NodeIndex) -> Result<&N, GraphError> {
        self.nodes.get(index).map(|node| &node.inner)
    }

    pub fn get_node_mut(&mut self, index: NodeIndex) -> Result<&mut N, GraphError> {
        self.nodes.get_mut(index).map(|node| &mut node.inner)
    }

    pub fn get_edge(&self, index: EdgeIndex) -> Result<&E, GraphError> {
        self.edges.get(index).map(|edge| &edge.inner)
    }

    pub fn get_edge_mut(&mut self, edge: EdgeIndex) -> Result<&mut E, GraphError> {
        self.edges.get_mut(edge).map(|edge| &mut edge.inner)
    }

    pub fn reset(&mut self, allocator: &'a mut GraphAllocator) {
        self.nodes.inner.clear();
        self.edges.inner.clear();
        allocator.0.reset();
    }
}

pub struct ChildrenWalker {
    node: NodeIndex,
    next: usize,
}

impl<'a, N, E> Walker<&Graph<'a, N, E>> for ChildrenWalker {
    type Item = (EdgeIndex, NodeIndex);

    #[inline(always)]
    fn walk_next(&mut self, graph: &Graph<'a, N, E>) -> Option<Self::Item> {
        if let Ok(node) = graph.nodes.get(self.node) {
            if let Some(edge_index) = node.outgoing.get(self.next) {
                let edge = graph.edges.get_unchecked(*edge_index);
                if edge.nodes.from == self.node {
                    self.next += 1;
                    return Some((*edge_index, edge.nodes.to));
                }
            }
        }
        None
    }
}

pub struct ParentsWalker {
    node: NodeIndex,
    next: usize,
}

impl<'a, N, E> Walker<&Graph<'a, N, E>> for ParentsWalker {
    type Item = (EdgeIndex, NodeIndex);

    #[inline(always)]
    fn walk_next(&mut self, graph: &Graph<'a, N, E>) -> Option<Self::Item> {
        if let Ok(node) = graph.nodes.get(self.node) {
            if let Some(edge_index) = node.incoming.get(self.next) {
                let edge = graph.edges.get_unchecked(*edge_index);
                self.next += 1;
                return Some((*edge_index, edge.nodes.from));
            }
        }
        None
    }
}

#[derive(Debug, Default)]
pub struct GraphAllocator(Bump);

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq)]
    pub struct TestNode {
        value: usize,
    }

    #[derive(Debug, Clone, PartialEq)]
    pub struct TestEdge {
        value: usize,
    }

    #[test]
    fn insertion() -> Result<(), GraphError> {
        let allocator = GraphAllocator::default();
        let mut graph = Graph::<TestNode, TestEdge>::default();

        let node1 = graph.insert_node(&allocator, TestNode { value: 1 })?;
        let node2 = graph.insert_node(&allocator, TestNode { value: 2 })?;
        let node3 = graph.insert_node(&allocator, TestNode { value: 3 })?;
        let node4 = graph.insert_node(&allocator, TestNode { value: 4 })?;
        let node5 = graph.insert_node(&allocator, TestNode { value: 5 })?;

        graph.insert_edge_unchecked(&allocator, TestEdge { value: 1 }, node1, node2)?;
        graph.insert_edge_unchecked(&allocator, TestEdge { value: 2 }, node2, node3)?;
        graph.insert_edge_unchecked(&allocator, TestEdge { value: 3 }, node3, node4)?;
        graph.insert_edge_unchecked(&allocator, TestEdge { value: 4 }, node3, node4)?;
        graph.insert_edge_unchecked(&allocator, TestEdge { value: 5 }, node4, node5)?;

        Ok(())
    }

    #[test]
    fn walk_directions() -> Result<(), GraphError> {
        let allocator = GraphAllocator::default();
        let mut graph = Graph::<TestNode, TestEdge>::default();

        let node1 = graph.insert_node(&allocator, TestNode { value: 1 })?;
        let node2 = graph.insert_node(&allocator, TestNode { value: 2 })?;
        let node3 = graph.insert_node(&allocator, TestNode { value: 3 })?;
        let node4 = graph.insert_node(&allocator, TestNode { value: 4 })?;
        let node5 = graph.insert_node(&allocator, TestNode { value: 5 })?;

        let edge1 = graph.insert_edge_unchecked(&allocator, TestEdge { value: 1 }, node1, node2)?;
        let edge2 = graph.insert_edge_unchecked(&allocator, TestEdge { value: 2 }, node1, node3)?;
        let edge3 = graph.insert_edge_unchecked(&allocator, TestEdge { value: 3 }, node1, node4)?;
        let edge4 = graph.insert_edge_unchecked(&allocator, TestEdge { value: 4 }, node4, node1)?;
        let edge5 = graph.insert_edge_unchecked(&allocator, TestEdge { value: 5 }, node5, node1)?;

        let mut children = node1.children();
        assert_eq!(Some((edge1, node2)), children.walk_next(&graph));
        assert_eq!(Some((edge2, node3)), children.walk_next(&graph));
        assert_eq!(Some((edge3, node4)), children.walk_next(&graph));
        assert_eq!(None, children.walk_next(&graph));

        let mut parents = node1.parents();
        assert_eq!(Some((edge4, node4)), parents.walk_next(&graph));
        assert_eq!(Some((edge5, node5)), parents.walk_next(&graph));
        assert_eq!(None, parents.walk_next(&graph));
        Ok(())
    }

    #[test]
    fn test_rewire_children() -> Result<(), GraphError> {
        let allocator = GraphAllocator::default();
        let mut graph = Graph::<TestNode, TestEdge>::default();

        let node1 = graph.insert_node(&allocator, TestNode { value: 1 })?;
        let node2 = graph.insert_node(&allocator, TestNode { value: 2 })?;
        let node3 = graph.insert_node(&allocator, TestNode { value: 3 })?;
        let node4 = graph.insert_node(&allocator, TestNode { value: 4 })?;
        let node5 = graph.insert_node(&allocator, TestNode { value: 5 })?;

        let edge1 = graph.insert_edge_unchecked(&allocator, TestEdge { value: 1 }, node1, node2)?;
        let edge2 = graph.insert_edge_unchecked(&allocator, TestEdge { value: 2 }, node1, node3)?;
        let edge3 = graph.insert_edge_unchecked(&allocator, TestEdge { value: 3 }, node3, node4)?;
        let edge4 = graph.insert_edge_unchecked(&allocator, TestEdge { value: 4 }, node3, node5)?;

        graph.rewire_children(node1, node3)?;

        let mut children = node1.children();
        assert_eq!(None, children.walk_next(&graph));

        let mut children = node3.children();
        assert_eq!(Some((edge3, node4)), children.walk_next(&graph));
        assert_eq!(Some((edge4, node5)), children.walk_next(&graph));
        assert_eq!(Some((edge1, node2)), children.walk_next(&graph));
        assert_eq!(Some((edge2, node3)), children.walk_next(&graph));
        assert_eq!(None, children.walk_next(&graph));
        Ok(())
    }

    #[test]
    fn test_rewire_parents_where() -> Result<(), GraphError> {
        let allocator = GraphAllocator::default();
        let mut graph = Graph::<TestNode, TestEdge>::default();

        let node1 = graph.insert_node(&allocator, TestNode { value: 1 })?;
        let node2 = graph.insert_node(&allocator, TestNode { value: 2 })?;
        let node3 = graph.insert_node(&allocator, TestNode { value: 3 })?;
        let node4 = graph.insert_node(&allocator, TestNode { value: 4 })?;
        let node5 = graph.insert_node(&allocator, TestNode { value: 5 })?;
        let node6 = graph.insert_node(&allocator, TestNode { value: 5 })?;
        let node7 = graph.insert_node(&allocator, TestNode { value: 5 })?;

        let edge1 = graph.insert_edge_unchecked(&allocator, TestEdge { value: 1 }, node2, node1)?;
        let edge2 = graph.insert_edge_unchecked(&allocator, TestEdge { value: 2 }, node3, node1)?;
        let edge3 = graph.insert_edge_unchecked(&allocator, TestEdge { value: 3 }, node4, node3)?;
        let edge4 = graph.insert_edge_unchecked(&allocator, TestEdge { value: 4 }, node5, node3)?;
        let edge5 = graph.insert_edge_unchecked(&allocator, TestEdge { value: 5 }, node6, node1)?;
        let edge6 = graph.insert_edge_unchecked(&allocator, TestEdge { value: 6 }, node7, node3)?;

        let mut parents = node1.parents();
        assert_eq!(Some((edge1, node2)), parents.walk_next(&graph));
        assert_eq!(Some((edge2, node3)), parents.walk_next(&graph));
        assert_eq!(Some((edge5, node6)), parents.walk_next(&graph));
        assert_eq!(None, parents.walk_next(&graph));

        let mut parents = node3.parents();
        assert_eq!(Some((edge3, node4)), parents.walk_next(&graph));
        assert_eq!(Some((edge4, node5)), parents.walk_next(&graph));
        assert_eq!(Some((edge6, node7)), parents.walk_next(&graph));
        assert_eq!(None, parents.walk_next(&graph));

        graph.rewire_where(Direction::Incoming, node1, node3, |e, _| e.value != 2)?;

        let mut parents = node1.parents();
        assert_eq!(Some((edge2, node3)), parents.walk_next(&graph));
        assert_eq!(None, parents.walk_next(&graph));

        let mut parents = node3.parents();
        assert_eq!(Some((edge3, node4)), parents.walk_next(&graph));
        assert_eq!(Some((edge4, node5)), parents.walk_next(&graph));
        assert_eq!(Some((edge6, node7)), parents.walk_next(&graph));
        assert_eq!(Some((edge1, node2)), parents.walk_next(&graph));
        assert_eq!(Some((edge5, node6)), parents.walk_next(&graph));
        assert_eq!(None, parents.walk_next(&graph));
        Ok(())
    }
}
