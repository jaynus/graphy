use crate::{Graph, EdgeType, NodeType, Node, Edge};

pub struct NodeIter<'a, N, E> {
    pub(crate) graph: &'a Graph<'a, N, E>,
    pub(crate) cur: usize,
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
    pub(crate) graph: &'a Graph<'a, N, E>,
    pub(crate) cur: usize,
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
impl<'a, N: NodeType, E: EdgeType> Iterator for NodeIterMut<'a, N, E> {
    type Item = &'a mut Node<N>;

    fn next(&mut self) -> Option<Self::Item> {
        self.cur += 1;
        self.graph.nodes.get_mut(self.cur).map(|node| unsafe { &mut *(*node as *mut Node<N>) })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.graph.nodes.len()))
    }
}
impl<'a, N: NodeType, E: EdgeType> DoubleEndedIterator for NodeIterMut<'a, N, E> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.cur -= 1;
        self.graph.nodes.get_mut(self.cur).map(|node| unsafe { &mut *(*node as *mut Node<N>) })
    }
}
impl<'a, N: NodeType, E: EdgeType> ExactSizeIterator for NodeIterMut<'a, N, E> {}


/// SAFETY: This iterator is safe because we are maintaing the mutable borrow on graph for the iterator
pub struct EdgeIterMut<'a, N, E> {
    pub(crate) graph: &'a mut Graph<'a, N, E>,
    pub(crate) cur: usize,
}
impl<'a, N: NodeType, E: EdgeType> Iterator for EdgeIterMut<'a, N, E> {
    type Item = &'a mut Edge<E>;

    fn next(&mut self) -> Option<Self::Item> {
        self.cur += 1;
        self.graph.edges.get_mut(self.cur).map(|edge| unsafe { &mut *(*edge as *mut Edge<E>) })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.graph.edges.len()))
    }
}
impl<'a, N: NodeType, E: EdgeType> DoubleEndedIterator for EdgeIterMut<'a, N, E> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.cur -= 1;
        self.graph.edges.get_mut(self.cur).map(|edge| unsafe { &mut *(*edge as *mut Edge<E>) })
    }
}
impl<'a, N: NodeType, E: EdgeType> ExactSizeIterator for EdgeIterMut<'a, N, E> {}
