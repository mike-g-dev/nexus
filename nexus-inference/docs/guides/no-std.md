# no_std Support

nexus-inference requires `std`. The `no_std`, `alloc`, and `libm`
feature flags were removed — the crate always links against the
standard library.
