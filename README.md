`StaticRc` is a safe reference-counted pointer, similar to `Rc` or `Arc`, though performing its reference-counting at
compile-time rather than run-time, and therefore avoiding most run-time overhead.

#   Motivating Example

A number of collections, such as linked-lists, binary-trees, or B-Trees are most easily implemented with aliasing
pointers.

Traditionally, this requires either `unsafe` raw pointers, or using `Rc` or `Arc` depending on the scenario. A key
observation, however, is that in those collections the exact number of aliases is known at compile-time:

-   A doubly linked-list has 2 pointers to each node.
-   A binary-tree has 3 pointers to each node: one from the parent, and one from each child.
-   A B-Tree of cardinality N has N+1 pointers to each node.

In this type of scenario, `static-rc` offers the safety of `Rc` and `Arc`, with the performance of `unsafe` raw
pointers.


#   Goals

Provide safe and efficient reference-counting:

-   Efficiency: most associated functions boil down to copying a `NonNull<T>`, a trivial operation.
    -   One key exception are `join` functions: a run-time check must be performed to ensure the instances being joined
        refer to the same pointer. Unsafe unchecked variants are available if their overhead is too high.
-   Safety: most associated functions are safe to use.
    -   The few unsafe functions are strictly optional.


#   Maturity

This crate is still very much experimental.

Review:

-   Minimally reviewed.
-   Not audited.
-   Not formally proven.

Documentation:

-   All `StaticRc` associated functions are documented, with example.
-   All `StaticRcRef` associated functions are documented, with example.

Testing:

-   All compile-time assertions are tested with compile-fail tests.
-   All panics are tested with panic tests.
-   Miri runs the test-suite without any complain.


#   Debug checks

This library contains a number of additional checks when building with `debug_assertions`, in particular the `Drop`
implementation of `StaticRc` will catch any attempt at destroying a `StaticRc<T, N, D>` where `N <> D`, as this would
typically result in a leak.

Those checks are not strictly necessary for safety, they are included to help point out logic errors.

From experience, the `Drop` check on top of an extensive test-suite will help catch all those instances where one path
accidentally let a pointer drop.


#   That's all folks!

And thanks for reading.
