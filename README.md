asdf

- no allocation per node/edge (just a single big pool of memory would be ideal)
- able to clear data and reuse that memory pool in next frame
- graph must be directed and acyclic (inseting an edge can result in an error result that informs about cycle)
- both nodes and edges can hold data
- edges are iterated in insertion order
- i need to be able to ask for node's parents very efficiently (this is the most used operation)
- i need to be able to ask for node's children fairly fast (used once per replacement operation)
- i need to be able to conceptually remove nodes/edges, but not necessarily need to deallocate them immediately