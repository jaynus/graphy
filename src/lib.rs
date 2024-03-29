pub mod walker;

use fixedbitset::FixedBitSet;
use purple::Arena;
use smallvec::SmallVec;
use std::fmt::Debug;

pub use walker::{ExactSizeWalker, Walker};
#[derive(Debug, Clone, PartialEq)]
struct Node<T> {
    inner: T,
    incoming: SmallVec<[EdgeIndex; 8]>,
    outgoing: SmallVec<[EdgeIndex; 8]>,
}
impl<T> Node<T> {
    pub(crate) fn new(inner: T) -> Self {
        Self {
            inner,
            incoming: SmallVec::default(),
            outgoing: SmallVec::default(),
        }
    }

    #[inline]
    pub(crate) fn edges_mut(&mut self, direction: Direction) -> &mut SmallVec<[EdgeIndex; 8]> {
        match direction {
            Direction::Incoming => &mut self.incoming,
            Direction::Outgoing => &mut self.outgoing,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EdgeNodes {
    pub from: NodeIndex,
    pub to: NodeIndex,
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
    #[inline]
    pub fn new(value: u32) -> Self {
        Self(value as _)
    }
    #[inline]
    pub fn end() -> Self {
        Self(u32::max_value())
    }
    #[inline]
    pub fn index(&self) -> usize {
        self.0 as _
    }
    #[inline]
    pub fn inner(&self) -> u32 {
        self.0
    }
    #[inline]
    pub fn parents(&self) -> ParentsWalker {
        ParentsWalker {
            node: *self,
            next: 0,
        }
    }
    #[inline]
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
    pub fn end() -> Self {
        Self(u32::max_value())
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

#[derive(Debug)]
pub struct Nodes<'a, N> {
    inner: SmallVec<[Option<&'a mut Node<N>>; 128]>,
}

impl<N> Default for Nodes<'_, N> {
    fn default() -> Self {
        Self {
            inner: SmallVec::default(),
        }
    }
}

impl<'a, N> Nodes<'a, N> {
    // :TakeShouldMove
    // @Incomplete: this would ideally move (return `Option<N>`), but it is currently not possible
    // as technically nodes are owned by the allocator and we only have exclusive references to them.
    // This should change when we will use `purple::collections::Vec` instead.
    // But for that it needs to be growable.
    #[inline]
    pub fn take(&mut self, index: NodeIndex) -> Option<&mut N> {
        self.inner
            .get_mut(index.index())
            .and_then(|n| n.take())
            .map(|n| &mut n.inner)
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &N> {
        self.inner
            .iter()
            .filter_map(|n| n.as_ref().map(|n| &n.inner))
    }

    #[inline]
    fn get(&self, index: NodeIndex) -> Option<&Node<N>> {
        self.inner
            .get(index.index())
            .and_then(|n| n.as_ref().map(|n| &**n))
    }

    #[inline]
    fn get_mut(&mut self, index: NodeIndex) -> Option<&mut Node<N>> {
        self.inner
            .get_mut(index.index())
            .and_then(|n| n.as_mut().map(|n| &mut **n))
    }

    #[inline]
    fn index_mut(&mut self, index: NodeIndex) -> &mut Node<N> {
        self.inner[index.index()].as_mut().unwrap()
    }

    #[inline]
    fn index(&self, index: NodeIndex) -> &Node<N> {
        self.inner[index.index()].as_ref().unwrap()
    }

    #[inline]
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

#[derive(Debug)]
pub struct Edges<'a, E> {
    inner: SmallVec<[Option<&'a mut Edge<E>>; 256]>,
}

impl<E> Default for Edges<'_, E> {
    fn default() -> Self {
        Self {
            inner: SmallVec::default(),
        }
    }
}

impl<'a, E> Edges<'a, E> {
    // :TakeShouldMove
    #[inline]
    pub fn take(&mut self, index: EdgeIndex) -> Option<&mut E> {
        self.inner
            .get_mut(index.index())
            .and_then(|n| n.take())
            .map(|n| &mut n.inner)
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &E> {
        self.inner
            .iter()
            .filter_map(|n| n.as_ref().map(|n| &n.inner))
    }

    #[inline]
    fn get(&self, index: EdgeIndex) -> Option<&Edge<E>> {
        self.inner
            .get(index.index())
            .and_then(|e| e.as_ref().map(|e| &**e))
    }

    #[inline]
    fn source(&self, index: EdgeIndex) -> Option<NodeIndex> {
        self.get(index).map(|edge| edge.nodes.from)
    }

    #[inline]
    fn destination(&self, index: EdgeIndex) -> Option<NodeIndex> {
        self.get(index).map(|edge| edge.nodes.to)
    }

    #[inline]
    fn get_mut(&mut self, index: EdgeIndex) -> Option<&mut Edge<E>> {
        self.inner
            .get_mut(index.index())
            .and_then(|e| e.as_mut().map(|e| &mut **e))
    }

    #[inline]
    fn index(&self, index: EdgeIndex) -> &Edge<E> {
        self.inner[index.index()].as_ref().unwrap()
    }

    #[inline]
    fn index_mut(&mut self, index: EdgeIndex) -> &mut Edge<E> {
        self.inner[index.index()].as_mut().unwrap()
    }

    #[inline]
    fn exists(&self, index: EdgeIndex) -> bool {
        self.inner.get(index.index()).map_or(false, |n| n.is_some())
    }

    #[inline]
    fn pair_mut(&mut self, first: EdgeIndex, second: EdgeIndex) -> (&mut Edge<E>, &mut Edge<E>) {
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
            panic!("Asked for pair of mutable references to the same edge")
        }
    }
}

#[derive(Debug)]
pub struct Graph<'a, N, E> {
    nodes: Nodes<'a, N>,
    edges: Edges<'a, E>,
}

impl<N, E> Default for Graph<'_, N, E> {
    fn default() -> Self {
        Self {
            nodes: Nodes::default(),
            edges: Edges::default(),
        }
    }
}

impl<'a, N, E> Graph<'a, N, E> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn remove_node(&mut self, index: NodeIndex) {
        if !self.nodes.exists(index) {
            return;
        }
        self.remove_nodes(&[index])
    }

    pub fn into_items(self) -> (Nodes<'a, N>, Edges<'a, E>) {
        (self.nodes, self.edges)
    }

    pub fn remove_nodes(&mut self, remove: &[NodeIndex]) {
        let (total_in, total_out) = remove.iter().fold((0, 0), |(i, o), index| {
            let (add_i, add_o) = self
                .nodes
                .get(*index)
                .map_or((0, 0), |n| (n.incoming.len(), n.outgoing.len()));
            (i + add_i, o + add_o)
        });

        let mut revisit_from = Vec::with_capacity(total_in);
        let mut revisit_to = Vec::with_capacity(total_out);

        for index in remove {
            if let Some(node) = &self.nodes.inner[index.index()] {
                for edge_index in node.incoming.iter() {
                    revisit_from.extend(self.edges.source(*edge_index));
                    self.edges.inner[edge_index.index()] = None;
                }
                for edge_index in node.outgoing.iter() {
                    revisit_to.extend(self.edges.destination(*edge_index));
                    self.edges.inner[edge_index.index()] = None;
                }
            }
            self.nodes.inner[index.index()] = None;
        }

        let edges = &self.edges;
        for index in revisit_from {
            if let Some(node) = &mut self.nodes.inner[index.index()] {
                node.outgoing.retain(|e| edges.exists(*e));
            }
        }
        for index in revisit_to {
            if let Some(node) = &mut self.nodes.inner[index.index()] {
                node.incoming.retain(|e| edges.exists(*e));
            }
        }
    }

    pub fn node_exists(&self, index: NodeIndex) -> bool {
        self.nodes.exists(index)
    }

    pub fn edge_exists(&self, index: EdgeIndex) -> bool {
        self.edges.exists(index)
    }

    pub fn node_count(&self) -> u32 {
        self.nodes.inner.len() as u32
    }

    pub fn edge_count(&self) -> u32 {
        self.edges.inner.len() as u32
    }

    pub fn insert_node(&mut self, allocator: &'a GraphAllocator, inner: N) -> NodeIndex {
        let index = NodeIndex(self.nodes.inner.len() as u32);
        self.nodes
            .inner
            .push(Some(allocator.0.alloc(Node::new(inner))));
        index
    }

    pub fn insert_child(
        &mut self,
        allocator: &'a GraphAllocator,
        parent: NodeIndex,
        edge: E,
        node: N,
    ) -> (EdgeIndex, NodeIndex) {
        let node_index = self.insert_node(allocator, node);
        let edge_index = self
            .insert_edge_unchecked(allocator, parent, node_index, edge)
            .unwrap();
        (edge_index, node_index)
    }

    pub fn insert_edge<F>(
        &mut self,
        _allocator: &'a GraphAllocator,
        _from: NodeIndex,
        _to: NodeIndex,
        _inner: E,
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
        from: NodeIndex,
        to: NodeIndex,
        inner: E,
    ) -> Result<EdgeIndex, GraphError> {
        if !self.nodes.exists(from) || !self.nodes.exists(to) {
            return Err(GraphError::InvalidIndex);
        }

        let index = EdgeIndex(self.edges.inner.len() as u32);
        self.edges
            .inner
            .push(Some(allocator.0.alloc(Edge::new(inner, from, to))));
        self.nodes.index_mut(from).outgoing.push(index);
        self.nodes.index_mut(to).incoming.push(index);
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

    /// Get two edges as mutable at the same time.
    /// Node ids must be different.
    pub fn edge_pair_mut(
        &mut self,
        first: EdgeIndex,
        second: EdgeIndex,
    ) -> Result<(&mut E, &mut E), GraphError> {
        if first == second || !self.edges.exists(first) || !self.edges.exists(second) {
            return Err(GraphError::InvalidIndex);
        }
        let (first, second) = self.edges.pair_mut(first, second);
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

        for edge_index in &self.nodes.index(source).outgoing {
            self.edges.index_mut(*edge_index).nodes.from = target;
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
                let edge = self.edges.index_mut(v[i]);
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
        let edge = self
            .edges
            .get_mut(edge_index)
            .ok_or(GraphError::InvalidIndex)?;

        let nodes = &self.nodes;
        if from.map_or(false, |from| !nodes.exists(from))
            || to.map_or(false, |to| !nodes.exists(to))
        {
            return Err(GraphError::InvalidIndex);
        }

        if let Some(from) = from {
            let next_node = self.nodes.index_mut(from);
            next_node.outgoing.push(edge_index);
            let prev_node = self.nodes.index_mut(edge.nodes.from);
            prev_node.outgoing.retain(|i| *i != edge_index);
            edge.nodes.from = from;
        }

        if let Some(to) = to {
            let next_node = self.nodes.index_mut(to);
            next_node.incoming.push(edge_index);
            let prev_node = self.nodes.index_mut(edge.nodes.to);
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
    pub fn edges_iter(&self) -> impl Iterator<Item = &E> {
        self.edges
            .inner
            .iter()
            .filter_map(|e| e.as_ref().map(|e| &e.inner))
    }
    pub fn nodes_indices_iter<'b>(&'b self) -> NodesIndicesIter<'a, 'b, N> {
        NodesIndicesIter {
            next: 0,
            nodes: &self.nodes,
        }
    }
    pub fn edges_indices_iter<'b>(&'b self) -> EdgesIndicesIter<'a, 'b, E> {
        EdgesIndicesIter {
            next: 0,
            edges: &self.edges,
        }
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
        let node = self.nodes.get(node_index).ok_or(GraphError::InvalidIndex)?;
        for edge_index in &node.outgoing {
            let edge = self.edges.index(*edge_index);
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
        let node = self.nodes.get(index).ok_or(GraphError::InvalidIndex)?;
        Ok(node.outgoing.iter().map(move |edge_index| {
            let edge = self.edges.index(*edge_index);
            (&edge.inner, &node.inner)
        }))
    }

    pub fn iter_parents<'b: 'a>(
        &'b self,
        index: NodeIndex,
    ) -> Result<impl Iterator<Item = (&'a E, &'a N)> + 'b, GraphError> {
        let node = self.nodes.get(index).ok_or(GraphError::InvalidIndex)?;
        Ok(node.incoming.iter().map(move |edge_index| {
            let edge = self.edges.index(*edge_index);
            (&edge.inner, &node.inner)
        }))
    }

    pub fn get_node(&self, index: NodeIndex) -> Option<&N> {
        self.nodes.get(index).map(|node| &node.inner)
    }

    pub fn get_node_mut(&mut self, index: NodeIndex) -> Option<&mut N> {
        self.nodes.get_mut(index).map(|node| &mut node.inner)
    }

    pub fn get_edge(&self, index: EdgeIndex) -> Option<&E> {
        self.edges.get(index).map(|edge| &edge.inner)
    }

    pub fn get_edge_mut(&mut self, edge: EdgeIndex) -> Option<&mut E> {
        self.edges.get_mut(edge).map(|edge| &mut edge.inner)
    }

    pub fn find_edge(&self, from: NodeIndex, to: NodeIndex) -> Option<EdgeIndex> {
        self.nodes
            .get(from)?
            .outgoing
            .iter()
            .find(|edge| self.edges.index(**edge).nodes.to == to)
            .copied()
    }

    pub fn has_edge<F>(&self, from: NodeIndex, to: NodeIndex, mut predicate: F) -> bool
    where
        F: FnMut(&E) -> bool,
    {
        self.nodes
            .index(from)
            .outgoing
            .iter()
            .find(|edge| {
                let edge = self.edges.index(**edge);
                edge.nodes.to == to && predicate(&edge.inner)
            })
            .is_some()
    }

    pub fn get_edge_endpoints(&self, index: EdgeIndex) -> Option<EdgeNodes> {
        self.edges.get(index).map(|edge| edge.nodes)
    }

    /// Remove all nodes that are not connected to a passed root node in incoming direction.
    pub fn trim(&mut self, root: NodeIndex) -> Result<(), GraphError> {
        if !self.nodes.exists(root) {
            return Err(GraphError::InvalidIndex);
        }

        let mut is_live = FixedBitSet::with_capacity(self.node_count() as _);
        let mut live = std::collections::VecDeque::with_capacity(self.node_count() as _);

        is_live.put(root.index());
        live.push_back(root);

        while let Some(index) = live.pop_front() {
            for (_, parent) in index.parents().iter(self) {
                if !is_live.put(parent.index()) {
                    live.push_back(parent);
                }
            }
        }

        // reuse same deque to save allocation
        let mut dead = live;
        dead.clear();

        for index in (0..self.node_count()).filter(|bit| !is_live[*bit as _]) {
            dead.push_back(NodeIndex::new(index));
        }

        let (left, right) = dead.as_slices();
        assert_eq!(0, right.len());
        if left.len() > 0 {
            self.remove_nodes(left);
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ChildrenWalker {
    node: NodeIndex,
    next: usize,
}

impl<'a, N, E> Walker<Graph<'a, N, E>> for ChildrenWalker {
    type Item = (EdgeIndex, NodeIndex);

    #[inline(always)]
    fn walk_next(&mut self, graph: &Graph<'a, N, E>) -> Option<Self::Item> {
        if let Some(node) = graph.nodes.get(self.node) {
            if let Some(edge_index) = node.outgoing.get(self.next) {
                let edge = graph.edges.index(*edge_index);
                if edge.nodes.from == self.node {
                    self.next += 1;
                    return Some((*edge_index, edge.nodes.to));
                }
            }
        }
        None
    }
}

impl<'a, N, E> ExactSizeWalker<Graph<'a, N, E>> for ChildrenWalker {
    #[inline(always)]
    fn len(&self, graph: &Graph<'a, N, E>) -> usize {
        graph
            .nodes
            .get(self.node)
            .map_or(0, |node| node.outgoing.len())
    }
}

#[derive(Debug, Clone)]
pub struct ParentsWalker {
    node: NodeIndex,
    next: usize,
}

impl<'a, N, E> Walker<Graph<'a, N, E>> for ParentsWalker {
    type Item = (EdgeIndex, NodeIndex);

    #[inline(always)]
    fn walk_next(&mut self, graph: &Graph<'a, N, E>) -> Option<Self::Item> {
        if let Some(node) = graph.nodes.get(self.node) {
            if let Some(edge_index) = node.incoming.get(self.next) {
                let edge = graph.edges.index(*edge_index);
                self.next += 1;
                return Some((*edge_index, edge.nodes.from));
            }
        }
        None
    }
}

impl<'a, N, E> ExactSizeWalker<Graph<'a, N, E>> for ParentsWalker {
    #[inline(always)]
    fn len(&self, graph: &Graph<'a, N, E>) -> usize {
        graph
            .nodes
            .get(self.node)
            .map_or(0, |node| node.incoming.len())
    }
}

pub struct NodesIndicesIter<'a, 'b, N> {
    next: u32,
    nodes: &'b Nodes<'a, N>,
}

impl<'a, 'b, N> Iterator for NodesIndicesIter<'a, 'b, N> {
    type Item = NodeIndex;
    fn next(&mut self) -> Option<Self::Item> {
        while self.next < self.nodes.inner.len() as u32 {
            let index = NodeIndex::new(self.next);
            self.next += 1;
            if self.nodes.exists(index) {
                return Some(index);
            }
        }
        None
    }
}

impl<'a, N, E> std::ops::Index<EdgeIndex> for Graph<'a, N, E> {
    type Output = E;
    #[inline]
    fn index(&self, index: EdgeIndex) -> &Self::Output {
        &self.edges.index(index).inner
    }
}

impl<'a, N, E> std::ops::IndexMut<EdgeIndex> for Graph<'a, N, E> {
    #[inline]
    fn index_mut(&mut self, index: EdgeIndex) -> &mut Self::Output {
        &mut self.edges.index_mut(index).inner
    }
}

impl<'a, N, E> std::ops::Index<NodeIndex> for Graph<'a, N, E> {
    type Output = N;
    #[inline]
    fn index(&self, index: NodeIndex) -> &Self::Output {
        &self.nodes.index(index).inner
    }
}

impl<'a, N, E> std::ops::IndexMut<NodeIndex> for Graph<'a, N, E> {
    #[inline]
    fn index_mut(&mut self, index: NodeIndex) -> &mut Self::Output {
        &mut self.nodes.index_mut(index).inner
    }
}

pub struct EdgesIndicesIter<'a, 'b, E> {
    next: u32,
    edges: &'b Edges<'a, E>,
}

impl<'a, 'b, E> Iterator for EdgesIndicesIter<'a, 'b, E> {
    type Item = EdgeIndex;
    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        while self.next < self.edges.inner.len() as u32 {
            let index = EdgeIndex::new(self.next);
            self.next += 1;
            if self.edges.exists(index) {
                return Some(index);
            }
        }
        None
    }
}
#[derive(Debug)]
pub struct GraphAllocator(pub Arena);
impl GraphAllocator {
    pub fn with_capacity(capacity: usize) -> Self {
        GraphAllocator(Arena::with_capacity(capacity))
    }

    // Safety: all references allocated since last reset must already be leaked.
    pub unsafe fn reset(&mut self) {
        self.0.reset();
    }
}

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
        let allocator = GraphAllocator::with_capacity(52_428_800);
        let mut graph = Graph::<TestNode, TestEdge>::default();

        let node1 = graph.insert_node(&allocator, TestNode { value: 1 });
        let node2 = graph.insert_node(&allocator, TestNode { value: 2 });
        let node3 = graph.insert_node(&allocator, TestNode { value: 3 });
        let node4 = graph.insert_node(&allocator, TestNode { value: 4 });
        let node5 = graph.insert_node(&allocator, TestNode { value: 5 });

        graph.insert_edge_unchecked(&allocator, node1, node2, TestEdge { value: 1 })?;
        graph.insert_edge_unchecked(&allocator, node2, node3, TestEdge { value: 2 })?;
        graph.insert_edge_unchecked(&allocator, node3, node4, TestEdge { value: 3 })?;
        graph.insert_edge_unchecked(&allocator, node3, node4, TestEdge { value: 4 })?;
        graph.insert_edge_unchecked(&allocator, node4, node5, TestEdge { value: 5 })?;

        Ok(())
    }

    #[test]
    fn walk_directions() -> Result<(), GraphError> {
        let allocator = GraphAllocator::with_capacity(52_428_800);
        let mut graph = Graph::<TestNode, TestEdge>::default();

        let node1 = graph.insert_node(&allocator, TestNode { value: 1 });
        let node2 = graph.insert_node(&allocator, TestNode { value: 2 });
        let node3 = graph.insert_node(&allocator, TestNode { value: 3 });
        let node4 = graph.insert_node(&allocator, TestNode { value: 4 });
        let node5 = graph.insert_node(&allocator, TestNode { value: 5 });

        let edge1 = graph.insert_edge_unchecked(&allocator, node1, node2, TestEdge { value: 1 })?;
        let edge2 = graph.insert_edge_unchecked(&allocator, node1, node3, TestEdge { value: 2 })?;
        let edge3 = graph.insert_edge_unchecked(&allocator, node1, node4, TestEdge { value: 3 })?;
        let edge4 = graph.insert_edge_unchecked(&allocator, node4, node1, TestEdge { value: 4 })?;
        let edge5 = graph.insert_edge_unchecked(&allocator, node5, node1, TestEdge { value: 5 })?;

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
        let allocator = GraphAllocator::with_capacity(52_428_800);
        let mut graph = Graph::<TestNode, TestEdge>::default();

        let node1 = graph.insert_node(&allocator, TestNode { value: 1 });
        let node2 = graph.insert_node(&allocator, TestNode { value: 2 });
        let node3 = graph.insert_node(&allocator, TestNode { value: 3 });
        let node4 = graph.insert_node(&allocator, TestNode { value: 4 });
        let node5 = graph.insert_node(&allocator, TestNode { value: 5 });

        let edge1 = graph.insert_edge_unchecked(&allocator, node1, node2, TestEdge { value: 1 })?;
        let edge2 = graph.insert_edge_unchecked(&allocator, node1, node3, TestEdge { value: 2 })?;
        let edge3 = graph.insert_edge_unchecked(&allocator, node3, node4, TestEdge { value: 3 })?;
        let edge4 = graph.insert_edge_unchecked(&allocator, node3, node5, TestEdge { value: 4 })?;

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
        let allocator = GraphAllocator::with_capacity(52_428_800);
        let mut graph = Graph::<TestNode, TestEdge>::default();

        let node1 = graph.insert_node(&allocator, TestNode { value: 1 });
        let node2 = graph.insert_node(&allocator, TestNode { value: 2 });
        let node3 = graph.insert_node(&allocator, TestNode { value: 3 });
        let node4 = graph.insert_node(&allocator, TestNode { value: 4 });
        let node5 = graph.insert_node(&allocator, TestNode { value: 5 });
        let node6 = graph.insert_node(&allocator, TestNode { value: 6 });
        let node7 = graph.insert_node(&allocator, TestNode { value: 7 });

        let edge1 = graph.insert_edge_unchecked(&allocator, node2, node1, TestEdge { value: 1 })?;
        let edge2 = graph.insert_edge_unchecked(&allocator, node3, node1, TestEdge { value: 2 })?;
        let edge3 = graph.insert_edge_unchecked(&allocator, node4, node3, TestEdge { value: 3 })?;
        let edge4 = graph.insert_edge_unchecked(&allocator, node5, node3, TestEdge { value: 4 })?;
        let edge5 = graph.insert_edge_unchecked(&allocator, node6, node1, TestEdge { value: 5 })?;
        let edge6 = graph.insert_edge_unchecked(&allocator, node7, node3, TestEdge { value: 6 })?;

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

    #[test]
    fn test_trim() -> Result<(), GraphError> {
        let allocator = GraphAllocator::with_capacity(52_428_800);
        let mut graph = Graph::<TestNode, TestEdge>::default();

        let node1 = graph.insert_node(&allocator, TestNode { value: 1 });
        let node2 = graph.insert_node(&allocator, TestNode { value: 2 });
        let node3 = graph.insert_node(&allocator, TestNode { value: 3 });
        let node4 = graph.insert_node(&allocator, TestNode { value: 4 });
        let node5 = graph.insert_node(&allocator, TestNode { value: 5 });
        let node6 = graph.insert_node(&allocator, TestNode { value: 6 });
        let node7 = graph.insert_node(&allocator, TestNode { value: 7 });
        let node8 = graph.insert_node(&allocator, TestNode { value: 8 });

        let edge1 = graph.insert_edge_unchecked(&allocator, node2, node1, TestEdge { value: 1 })?;
        let edge2 = graph.insert_edge_unchecked(&allocator, node4, node1, TestEdge { value: 2 })?;
        let edge3 = graph.insert_edge_unchecked(&allocator, node4, node3, TestEdge { value: 3 })?;
        let edge4 = graph.insert_edge_unchecked(&allocator, node5, node3, TestEdge { value: 4 })?;
        let edge5 = graph.insert_edge_unchecked(&allocator, node6, node1, TestEdge { value: 5 })?;
        let edge6 = graph.insert_edge_unchecked(&allocator, node7, node3, TestEdge { value: 6 })?;

        graph.trim(node1)?;

        assert_eq!(true, graph.node_exists(node1));
        assert_eq!(true, graph.node_exists(node2));
        assert_eq!(false, graph.node_exists(node3));
        assert_eq!(true, graph.node_exists(node4));
        assert_eq!(false, graph.node_exists(node5));
        assert_eq!(true, graph.node_exists(node6));
        assert_eq!(false, graph.node_exists(node7));
        assert_eq!(false, graph.node_exists(node8));

        assert_eq!(true, graph.edge_exists(edge1));
        assert_eq!(true, graph.edge_exists(edge2));
        assert_eq!(false, graph.edge_exists(edge3));
        assert_eq!(false, graph.edge_exists(edge4));
        assert_eq!(true, graph.edge_exists(edge5));
        assert_eq!(false, graph.edge_exists(edge6));

        Ok(())
    }
}
