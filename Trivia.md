### Shallow copy

One method of copying an object is the _shallow copy_. In that case a new object B is created, and the fields values of A are copied over to B.This is also known as a _field-by-field copy_,_field-for-field copy_, or _field copy_If the field value is a reference to an object (e.g., a memory address) it copies the reference, hence referring to the same object as A does, and if the field value is a primitive type, it copies the value of the primitive type. In languages without primitive types (where everything is an object), all fields of the copy B are references to the same objects as the fields of original A. The referenced objects are thus _shared_, so if one of these objects is modified (from A or B), the change is visible in the other. Shallow copies are simple and typically cheap, as they can usually be implemented by simply copying the bits exactly.
### Deep copy

An alternative is a deep copy, meaning that fields are dereferenced: rather than references to objects being copied, new copy of objects are created for any referenced objects, and references to these are placed in B. Later modifications to the contents of either remain unique to A or B, as the contents are not shared.

### Queue:
In java, queue is not a class, its an interface so an object cannot be initialised. If you want a queue object then you must initialise using "new PriorityQueue<>()" or "new LinkedList<>()"

Also, the method poll() is preferred over remove() to get queue top because when you run remove() on empty queue then it throws an exception but poll returns null.

#### Graphs:
Two types: undirected and directed

Graphs can have cycles in them

A path is an order of traversal that reaches a lot of nodes but maximum once

Degree of node means number of nodes connected to it

Degree of graph is twice the number of edges

![[Pasted image 20250821223321.png]]
Visualisation of graph

#### Priority Queue:
Stored in array but represented as binary tree.
root = 1
Every index i has 
parent = i//2
left child = i\*2
right child = i\*2 + 1
height = log N

Insert: **MIN HEAP** ->
while child < parent: swap

Delete: min heap ->
lets say you delete head node, now you put last node to head and do the following:
min = left or right
swap temp head with min
continue until  temp head < both of its children

Why do we use array or arraylist for implementing priority queue i.e. heap? because linkedlist insert at index takes O(n) but array takes O(log n)

by default java priority queue is min heap but we can make max heap using:
1. PriorityQueue\<Integer> heap = new PriorityQueue<\Integer>(Collections.reverseOrder());
2. PriorityQueue\<Integer> heap = new PriorityQueue<\Integer>((a, b) -> b-a);
#### Trees:
in a binary tree: nodes **after n/2 are all leaf nodes**

