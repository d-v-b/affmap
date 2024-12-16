# Affmap: affine transformations with named axes

## Motivation

The matrix representation of an affine transformation can be ambiguous if the reader and the writer disagree on the array indexing order of the transform, or the order of the dimensions of the data. Bugs caused by this confusion are not very fun. This library prevents these bugs by representing affine transformations as a data structure that is independent of array indexing conventions.