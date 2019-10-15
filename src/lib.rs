mod iters;

use bumpalo::Bump;
use derivative::Derivative;
use smallvec::SmallVec;
use std::{fmt::Debug, sync::Arc};

pub trait NodeType: Debug + Clone + PartialEq {}
impl<T> NodeType for T where T: Debug + Clone + PartialEq {}

pub trait EdgeType: Debug + Clone + PartialEq {}
impl<T> EdgeType for T where T: Debug + Clone + PartialEq {}

// TODO: We should implement smallvec for BumpVec growth
// Lifetimes make this hard for now so whatever

#[derive(Debug, Clone, PartialEq)]
pub struct Node<T> {
    pub inner: T,
    pub incoming: SmallVec<[EdgeIndex; 8]>,
    pub outgoing: SmallVec<[EdgeIndex; 8]>,
}
impl<T: NodeType> Node<T> {
    pub fn new(inner: T) -> Self {
        Self {
            inner,
            incoming: SmallVec::default(),
            outgoing: SmallVec::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum Direction {
    In,
    Out,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EdgeNodes {
    from: NodeIndex,
    to: NodeIndex,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Edge<T> {
    pub inner: T,
    pub nodes: EdgeNodes,
}

impl<T: EdgeType> Edge<T> {
    pub fn new(inner: T, from: NodeIndex, to: NodeIndex) -> Self {
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

#[derive(Debug, Default, Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct EdgeIndex(u32);
impl std::ops::Add for EdgeIndex {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }
}

pub trait Index {
    fn new(value: u32) -> Self;
    fn index(&self) -> usize;
}
impl Index for NodeIndex {
    fn new(value: u32) -> Self {
        Self(value)
    }
    fn index(&self) -> usize {
        self.0 as usize
    }
}
impl Index for EdgeIndex {
    fn new(value: u32) -> Self {
        Self(value)
    }
    fn index(&self) -> usize {
        self.0 as usize
    }
}
#[derive(Derivative, Debug)]
#[derivative(Default(bound = ""))]
pub struct Graph<'a, N, E> {
    nodes: SmallVec<[&'a mut Node<N>; 128]>,
    edges: SmallVec<[&'a mut Edge<E>; 256]>,
}
impl<'a, N: NodeType, E: EdgeType> Graph<'a, N, E> {
    pub fn insert_node(
        &mut self,
        allocator: &'a GraphAllocator,
        inner: N,
    ) -> Result<NodeIndex, GraphError> {
        let index = NodeIndex(self.edges.len() as u32);

        self.nodes.push(allocator.0.alloc(Node::new(inner)));

        Ok(index)
    }

    pub fn insert_edge<F>(
        &mut self,
        allocator: &'a GraphAllocator,
        inner: E,
        from: NodeIndex,
        to: NodeIndex,
        cyclic_fn: F,
    ) -> Result<EdgeIndex, GraphError>
    where
        F: FnOnce(Self, E) -> Result<EdgeIndex, GraphError>,
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
        let index = EdgeIndex(self.edges.len() as u32);

        self.edges
            .push(allocator.0.alloc(Edge::new(inner, from, to)));

        self.nodes
            .get_mut(from.index())
            .ok_or_else(|| GraphError::InvalidIndex)?
            .outgoing
            .push(index);
        self.nodes
            .get_mut(to.index())
            .ok_or_else(|| GraphError::InvalidIndex)?
            .incoming
            .push(index);

        Ok(index)
    }

    pub fn update_edge(
        &mut self,
        edge_index: EdgeIndex,
        from: Option<NodeIndex>,
        to: Option<NodeIndex>,
    ) -> Result<(), GraphError> {
        let edge = self
            .edges
            .get_mut(edge_index.index())
            .ok_or_else(|| GraphError::InvalidIndex)?;

        if let Some(from) = from {
            self.nodes
                .get_mut(edge.nodes.from.index())
                .ok_or_else(|| GraphError::InvalidIndex)?
                .outgoing
                .remove(edge_index.index());
            edge.nodes.from;
        }
        if let Some(to) = to {
            self.nodes
                .get_mut(edge.nodes.to.index())
                .ok_or_else(|| GraphError::InvalidIndex)?
                .incoming
                .remove(edge_index.index());
            edge.nodes.to;
        }

        Ok(())
    }

    pub fn nodes_iter(&'a self) -> impl Iterator<Item = &'a Node<N>> {
        iters::NodeIter::new(self)
    }
    pub fn nodes_iter_mut(&'a mut self) -> impl Iterator<Item = &'a mut Node<N>> {
        iters::NodeIterMut::new(self)
    }

    pub fn edges_iter(&'a self) -> impl Iterator<Item = &'a Edge<E>> {
        iters::EdgeIter::new(self)
    }
    pub fn edges_iter_mut(&'a mut self) -> impl Iterator<Item = &'a mut Edge<E>> {
        iters::EdgeIterMut::new(self)
    }

    pub fn nodes_indices_iter(&'a self) -> impl Iterator<Item = NodeIndex> {
        iters::IndexIter::<NodeIndex>::new(self.nodes.len())
    }
    pub fn edges_indices_iter(&'a self) -> impl Iterator<Item = EdgeIndex> {
        iters::IndexIter::<EdgeIndex>::new(self.edges.len())
    }

    // Return false from the visitor to stop this branches traversal
    pub fn visit<V>(&self, node_index: NodeIndex, visitor: V) -> Result<(), GraphError>
    where
        V: FnMut(&Graph<N, E>, &Node<N>, &Edge<E>) -> bool,
    {
        self.visit_traverse(node_index, Arc::new(visitor))
    }

    // Return false from the visitor to stop this branches traversal
    fn visit_traverse<V>(
        &self,
        node_index: NodeIndex,
        mut visitor: Arc<V>,
    ) -> Result<(), GraphError>
    where
        V: FnMut(&Graph<N, E>, &Node<N>, &Edge<E>) -> bool,
    {
        let node = self
            .nodes
            .get(node_index.index())
            .ok_or_else(|| GraphError::InvalidIndex)?;
        for edge_index in &node.outgoing {
            let edge = self
                .edges
                .get(edge_index.index())
                .ok_or_else(|| GraphError::InvalidIndex)?;
            if (Arc::get_mut(&mut visitor).unwrap())(self, node, edge) {
                self.visit_traverse(edge.nodes.to, visitor.clone())?;
            }
        }

        Ok(())
    }

    pub fn get_neighbors(
        &'a self,
        node_index: NodeIndex,
    ) -> impl Iterator<Item = iters::RelationshipRef<N, E>> {
        iters::NeighborsRefIter::new(
            self,
            self.nodes
                .get(node_index.index())
                .unwrap()
                .incoming
                .iter()
                .map(move |edge_index| {
                    let edge: &Edge<E> = self.edges.get(edge_index.index()).unwrap();
                    let node = self.nodes.get(edge.nodes.from.index()).unwrap();
                    iters::RelationshipRef::Parent(edge, node)
                })
                .chain(
                    self.nodes
                        .get(node_index.index())
                        .unwrap()
                        .outgoing
                        .iter()
                        .map(move |edge_index| {
                            let edge: &Edge<E> = self.edges.get(edge_index.index()).unwrap();
                            let node = self.nodes.get(edge.nodes.from.index()).unwrap();
                            iters::RelationshipRef::Child(edge, node)
                        }),
                ),
        )
    }

    pub fn get_node(&self, node_index: NodeIndex) -> Option<&Node<N>> {
        self.nodes.get(node_index.index()).map(|node| &**node)
    }

    pub fn get_node_mut(&mut self, node_index: NodeIndex) -> Option<&mut Node<N>> {
        self.nodes
            .get_mut(node_index.index())
            .map(|node| &mut **node)
    }

    pub fn get_edge(&self, node_index: NodeIndex) -> Option<&Edge<E>> {
        self.edges.get(node_index.index()).map(|edge| &**edge)
    }

    pub fn get_edge_mut(&mut self, node_index: NodeIndex) -> Option<&mut Edge<E>> {
        self.edges
            .get_mut(node_index.index())
            .map(|edge| &mut **edge)
    }

    // TODO:
    fn reset(&mut self, allocator: &'a mut GraphAllocator) {
        allocator.0.reset();
        self.nodes.clear();
        self.edges.clear();
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

        let edge1 = graph.insert_edge_unchecked(&allocator, TestEdge { value: 1 }, node1, node2)?;
        let edge2 = graph.insert_edge_unchecked(&allocator, TestEdge { value: 2 }, node2, node3)?;
        let edge3 = graph.insert_edge_unchecked(&allocator, TestEdge { value: 3 }, node3, node4)?;
        let edge4 = graph.insert_edge_unchecked(&allocator, TestEdge { value: 4 }, node3, node4)?;
        let edge5 = graph.insert_edge_unchecked(&allocator, TestEdge { value: 5 }, node4, node5)?;

        Ok(())
    }
}
