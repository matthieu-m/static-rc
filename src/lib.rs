//! `StaticRc`, resp. `StaticRcRef`, use Rust's affine type system and const generics to track the shared ownership
//!  of a heap-allocated, resp. reference, value safely at compile-time, with no run-time overhead.
//!
//! The amount of `unsafe` used within is minimal, `StaticRc` mostly leverages `Box` for most of the heavy-duty
//! operations.
//!
//! #   Example of usage.
//!
//! ```
//! use static_rc::StaticRc;
//!
//! type Full<T> = StaticRc<T, 3, 3>;
//! type TwoThird<T> = StaticRc<T, 2, 3>;
//! type OneThird<T> = StaticRc<T, 1, 3>;
//!
//! let mut full = Full::new("Hello, world!".to_string());
//!
//! assert_eq!("Hello, world!", &*full);
//!
//! //  Mutation is allowed when having full ownership, just like for `Box`.
//! *full = "Hello, you!".to_string();
//!
//! assert_eq!("Hello, you!", &*full);
//!
//! //  Mutation is no longer allowed from now on, due to aliasing, just like for `Rc`.
//! let (two_third, one_third) = Full::split::<2, 1>(full);
//!
//! assert_eq!("Hello, you!", &*two_third);
//! assert_eq!("Hello, you!", &*one_third);
//!
//! let mut full = Full::join(one_third, two_third);
//!
//! assert_eq!("Hello, you!", &*full);
//!
//! //  Mutation is allowed again, since `full` has full ownership.
//! *full = "Hello, world!".to_string();
//!
//! assert_eq!("Hello, world!", &*full);
//!
//! //  Finally, the value is dropped when `full` is.
//! ```
//!
//! #   Options
//!
//! The crate is defined for `no_std` environment and only relies on `core` and `alloc` by default.
//!
//! The `alloc` crate can be opted out of, though this disables `StaticRc`.
//!
//! The crate only uses stable features by default, with a MSRV of 1.51 due to the use of const generics.
//!
//! Additional, the crate offers several optional features which unlock additional capabilities by using nightly.
//! Please see `Cargo.toml` for an up-to-date list of features, and their effects.

//  Regular features
#![cfg_attr(not(test), no_std)]

//  Nightly features
#![cfg_attr(feature = "compile-time-ratio", allow(incomplete_features))]
#![cfg_attr(feature = "compile-time-ratio", feature(const_generics, const_evaluatable_checked))]
#![cfg_attr(feature = "nightly-async-stream", feature(async_stream))]
#![cfg_attr(feature = "nightly-coerce-unsized", feature(coerce_unsized))]
#![cfg_attr(feature = "nightly-dispatch-from-dyn", feature(dispatch_from_dyn))]
#![cfg_attr(any(feature = "nightly-dispatch-from-dyn", feature = "nightly-coerce-unsized"), feature(unsize))]
#![cfg_attr(feature = "nightly-generator-trait", feature(generator_trait))]

//  Lints
#![deny(missing_docs)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[macro_use]
mod utils;

#[cfg(feature = "alloc")]
mod rc;
mod rcref;

#[cfg(feature = "alloc")]
pub use self::rc::StaticRc;
pub use self::rcref::StaticRcRef;

/// Lifts `root` into the slot provided by `fun`; returns the previous value of the slot, if any.
///
/// This function is useful, for example, to "tie the knot" when appending 2 linked-lists: it is easy to splice the
/// the head of the back linked-list at the back of the front linked-list, but then one has lost the head pointer
/// and can no longer splice the tail of the front linked-list to it.
///
/// #   Experimental
///
/// This function is highly experimental, see the ongoing discussion at
/// https://users.rust-lang.org/t/can-you-break-the-lift/58858.
#[cfg(feature = "experimental-lift")]
pub fn lift<F, R>(root: R, fun: F) -> R
where
    F: for<'a> FnOnce(&'a R) -> &'a mut R,
{
    let slot = fun(&root) as *mut R;

    debug_assert_ne!(slot as *const _, &root as *const _);

    //  Safety:
    //  -   `root` is still alive, hence any reference linked to `root` is _also_ alive.
    //  -   `slot` is necessarily different from `root`, being mutable.
    core::mem::replace(unsafe { &mut *slot }, root)
}
