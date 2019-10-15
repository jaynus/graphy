use crate::{Edge, EdgeType, Graph, Index, Node, NodeIndex, NodeType};
use smallvec::SmallVec;
use std::{marker::PhantomData, ops::Range};

#[derive(Debug)]
pub enum RelationshipRef<'a, N, E> {
    Parent(&'a Edge<E>, &'a Node<N>),
    Child(&'a Edge<E>, &'a Node<N>),
}

pub struct NeighborsRefIter<'a, N, E, I>
where
    I: Iterator<Item = RelationshipRef<'a, N, E>>,
{
    graph: &'a Graph<'a, N, E>,
    iter: I,
}
impl<'a, N: NodeType, E: EdgeType, I> NeighborsRefIter<'a, N, E, I>
where
    I: Iterator<Item = RelationshipRef<'a, N, E>>,
{
    pub(crate) fn new(graph: &'a Graph<'a, N, E>, iter: I) -> Self {
        Self { graph, iter }
    }
}
impl<'a, N: NodeType, E: EdgeType, I> Iterator for NeighborsRefIter<'a, N, E, I>
where
    I: Iterator<Item = RelationshipRef<'a, N, E>>,
{
    type Item = RelationshipRef<'a, N, E>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.graph.nodes.len()))
    }
}
/*
impl<'a, N: NodeType, E: EdgeType> DoubleEndedIterator for NeighborsRefIter<'a, N, E> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.cur -= 1;
        self.graph.nodes.get(self.cur).map(|node| &**node)
    }
}
impl<'a, N: NodeType, E: EdgeType> ExactSizeIterator for NeighborsRefIter<'a, N, E> {}
*/

pub struct IndexIter<I> {
    range: Range<usize>,
    cur: usize,
    _marker: PhantomData<I>,
}
impl<I: Index> IndexIter<I> {
    pub(crate) fn new(len: usize) -> Self {
        Self {
            range: Range { start: 0, end: len },
            cur: 0,
            _marker: Default::default(),
        }
    }
}
impl<I: Index> Iterator for IndexIter<I> {
    type Item = I;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cur >= self.range.end {
            None
        } else {
            let index = I::new(self.cur as u32);
            self.cur += 1;
            Some(index)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.range.end))
    }
}
impl<I: Index> DoubleEndedIterator for IndexIter<I> {
    fn next_back(&mut self) -> Option<Self::Item> {
        unimplemented!()
    }
}
impl<I: Index> ExactSizeIterator for IndexIter<I> {}

pub struct NodeIter<'a, N, E> {
    graph: &'a Graph<'a, N, E>,
    cur: usize,
}
impl<'a, N: NodeType, E: EdgeType> NodeIter<'a, N, E> {
    pub(crate) fn new(graph: &'a Graph<'a, N, E>) -> Self {
        Self { graph, cur: 0 }
    }
}
impl<'a, N: NodeType, E: EdgeType> Iterator for NodeIter<'a, N, E> {
    type Item = &'a Node<N>;

    fn next(&mut self) -> Option<Self::Item> {
        self.cur += 1;
        self.graph.nodes.get(self.cur).map(|node| &**node)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.graph.nodes.len()))
    }
}
impl<'a, N: NodeType, E: EdgeType> DoubleEndedIterator for NodeIter<'a, N, E> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.cur -= 1;
        self.graph.nodes.get(self.cur).map(|node| &**node)
    }
}
impl<'a, N: NodeType, E: EdgeType> ExactSizeIterator for NodeIter<'a, N, E> {}

pub struct EdgeIter<'a, N, E> {
    graph: &'a Graph<'a, N, E>,
    cur: usize,
}
impl<'a, N: NodeType, E: EdgeType> EdgeIter<'a, N, E> {
    pub(crate) fn new(graph: &'a Graph<'a, N, E>) -> Self {
        Self { graph, cur: 0 }
    }
}
impl<'a, N: NodeType, E: EdgeType> Iterator for EdgeIter<'a, N, E> {
    type Item = &'a Edge<E>;

    fn next(&mut self) -> Option<Self::Item> {
        self.cur += 1;
        self.graph.edges.get(self.cur).map(|edges| &**edges)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.graph.edges.len()))
    }
}
impl<'a, N: NodeType, E: EdgeType> DoubleEndedIterator for EdgeIter<'a, N, E> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.cur -= 1;
        self.graph.edges.get(self.cur).map(|edges| &**edges)
    }
}
impl<'a, N: NodeType, E: EdgeType> ExactSizeIterator for EdgeIter<'a, N, E> {}

/// SAFETY: This iterator is safe because we are maintaing the mutable borrow on graph for the iterator
pub struct NodeIterMut<'a, N, E> {
    pub(crate) graph: &'a mut Graph<'a, N, E>,
    pub(crate) cur: usize,
}
impl<'a, N: NodeType, E: EdgeType> NodeIterMut<'a, N, E> {
    pub(crate) fn new(graph: &'a mut Graph<'a, N, E>) -> Self {
        Self { graph, cur: 0 }
    }
}
impl<'a, N: NodeType, E: EdgeType> Iterator for NodeIterMut<'a, N, E> {
    type Item = &'a mut Node<N>;

    fn next(&mut self) -> Option<Self::Item> {
        self.cur += 1;
        self.graph
            .nodes
            .get_mut(self.cur)
            .map(|node| unsafe { &mut *(*node as *mut Node<N>) })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.graph.nodes.len()))
    }
}
impl<'a, N: NodeType, E: EdgeType> DoubleEndedIterator for NodeIterMut<'a, N, E> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.cur -= 1;
        self.graph
            .nodes
            .get_mut(self.cur)
            .map(|node| unsafe { &mut *(*node as *mut Node<N>) })
    }
}
impl<'a, N: NodeType, E: EdgeType> ExactSizeIterator for NodeIterMut<'a, N, E> {}

/// SAFETY: This iterator is safe because we are maintaing the mutable borrow on graph f  or the iterator
pub struct EdgeIterMut<'a, N, E> {
    pub(crate) graph: &'a mut Graph<'a, N, E>,
    pub(crate) cur: usize,
}
impl<'a, N: NodeType, E: EdgeType> EdgeIterMut<'a, N, E> {
    pub(crate) fn new(graph: &'a mut Graph<'a, N, E>) -> Self {
        Self { graph, cur: 0 }
    }
}
impl<'a, N: NodeType, E: EdgeType> Iterator for EdgeIterMut<'a, N, E> {
    type Item = &'a mut Edge<E>;

    fn next(&mut self) -> Option<Self::Item> {
        self.cur += 1;
        self.graph
            .edges
            .get_mut(self.cur)
            .map(|edge| unsafe { &mut *(*edge as *mut Edge<E>) })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.graph.edges.len()))
    }
}
impl<'a, N: NodeType, E: EdgeType> DoubleEndedIterator for EdgeIterMut<'a, N, E> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.cur -= 1;
        self.graph
            .edges
            .get_mut(self.cur)
            .map(|edge| unsafe { &mut *(*edge as *mut Edge<E>) })
    }
}
impl<'a, N: NodeType, E: EdgeType> ExactSizeIterator for EdgeIterMut<'a, N, E> {}
