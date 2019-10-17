/// A walker is a traversal state, but where part of the traversal
/// information is supplied manually to each next call.
///
/// This for example allows graph traversals that don't hold a borrow of the
/// graph they are traversing.
pub trait Walker<Context> {
    type Item;
    /// Advance to the next item
    fn walk_next(&mut self, context: &Context) -> Option<Self::Item>;

    /// Create an iterator out of the walker and given `context`.
    #[inline]
    fn iter<'a>(self, context: &'a Context) -> WalkerIter<'a, Self, Context>
    where
        Self: Sized,
    {
        WalkerIter {
            walker: self,
            context: &context,
        }
    }

    #[inline]
    fn filter<P>(self, predicate: P) -> Filter<Self, P>
    where
        Self: Sized,
        P: FnMut(&Context, &Self::Item) -> bool,
    {
        Filter {
            walker: self,
            predicate,
        }
    }

    #[inline]
    fn filter_map<P, I>(self, predicate: P) -> FilterMap<Self, P>
    where
        Self: Sized,
        P: FnMut(&Context, &Self::Item) -> Option<I>,
    {
        FilterMap {
            walker: self,
            predicate,
        }
    }
}

pub trait ExactSizeWalker<Context> {
    fn len(&self, context: &Context) -> usize;
}

/// A walker and its context wrapped into an iterator.
#[derive(Clone, Debug)]
pub struct WalkerIter<'a, W, C> {
    walker: W,
    context: &'a C,
}

impl<'a, W, C> WalkerIter<'a, W, C>
where
    W: Walker<C>,
{
    #[inline]
    pub fn context(&self) -> &C {
        &self.context
    }

    #[inline]
    pub fn inner_ref(&self) -> &W {
        &self.walker
    }

    #[inline]
    pub fn inner_mut(&mut self) -> &mut W {
        &mut self.walker
    }
}

impl<'a, W, C> Iterator for WalkerIter<'a, W, C>
where
    W: Walker<C>,
{
    type Item = W::Item;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.walker.walk_next(self.context)
    }
}

impl<W, C> std::iter::ExactSizeIterator for WalkerIter<'_, W, C>
where
    W: Walker<C> + ExactSizeWalker<C>,
{
    fn len(&self) -> usize {
        self.walker.len(self.context.clone())
    }
}

#[derive(Clone, Debug)]
pub struct FilterMap<W, P> {
    walker: W,
    predicate: P,
}

impl<C, W, I, P> Walker<C> for FilterMap<W, P>
where
    W: Walker<C>,
    P: FnMut(&C, &W::Item) -> Option<I>,
{
    type Item = I;
    #[inline]
    fn walk_next(&mut self, context: &C) -> Option<Self::Item> {
        while let Some(item) = self.walker.walk_next(context) {
            if let Some(mapped) = (self.predicate)(context, &item) {
                return Some(mapped);
            }
        }
        None
    }
}

#[derive(Clone, Debug)]
pub struct Filter<W, P> {
    walker: W,
    predicate: P,
}

impl<C, W, P> Walker<C> for Filter<W, P>
where
    W: Walker<C>,
    P: FnMut(&C, &W::Item) -> bool,
{
    type Item = W::Item;
    #[inline]
    fn walk_next(&mut self, context: &C) -> Option<Self::Item> {
        while let Some(item) = self.walker.walk_next(context) {
            if (self.predicate)(context, &item) {
                return Some(item);
            }
        }
        None
    }
}
