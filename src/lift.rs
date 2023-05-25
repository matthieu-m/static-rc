//! Lifting allows tying the note when linear or affine types -- such as `StaticRc` -- are used to implement cyclic
//! data-structures such as linked-lists.

use core::{mem::ManuallyDrop, ptr};

/// Lifts `root` into the slot provided by `fun`; returns the previous value of the slot, if any.
///
/// This function is useful, for example, to "tie the knot" when appending 2 linked-lists: it is easy to splice the
/// the head of the back linked-list at the back of the front linked-list, but then one has lost the head pointer
/// and can no longer splice the tail of the front linked-list to it.
///
/// #   Panics
///
/// If `fun` panics, then `root` is forgotten. This may result in a resource leak.
///
/// #   Experimental
///
/// This function is highly experimental, see the ongoing discussion at
/// https://users.rust-lang.org/t/can-you-break-the-lift/58858.
pub fn lift<F, R>(root: R, fun: F) -> R
where
    F: for<'a> FnOnce(&'a R) -> &'a mut R,
{
    //  The move into Manually Drop must happen _before_ to appease `MIRIFLAGS=-Zmiri-track-raw-pointers`
    let root = ManuallyDrop::new(root);
    let slot = fun(&root) as *mut R as *mut ManuallyDrop<R>;

    //  Safety:
    //  -   `root` is still alive, hence any reference linked to `root` is _also_ alive.
    //  -   `root` is valid for reads.
    //  -   `slot` is valid for reads & writes.
    unsafe { replace(slot, &root as *const _) }
}

/// Lifts `root` into the slot provided by `fun`; returns the previous value of the slot, if any.
///
/// #   Panics
///
/// If `fun` panics, then `root` is forgotten. This may result in a resource leak.
///
/// #   Note
///
/// This function is similar to `lift`. The `extra` argument is required to work-around `for<'a>` not otherwise
/// appropriately constraining its range of lifetime.
pub fn lift_with<F, E, R>(root: R, extra: &E, fun: F) -> R
where
    F: for<'a> FnOnce(&'a R, &'a E) -> &'a mut R,
{
    //  The move into Manually Drop must happen _before_ to appease `MIRIFLAGS=-Zmiri-track-raw-pointers`
    let root = ManuallyDrop::new(root);
    let slot = fun(&root, extra) as *mut R as *mut ManuallyDrop<R>;

    //  Safety:
    //  -   `root` is still alive, hence any reference linked to `root` is _also_ alive.
    //  -   `root` is valid for reads.
    //  -   `slot` is valid for reads & writes.
    unsafe { replace(slot, &root as *const _) }
}

/// Lifts `root` into the slot provided by `fun`; returns the previous value of the slot, if any.
///
/// #   Panics
///
/// If `fun` panics, then `root` is forgotten. This may result in a resource leak.
///
/// #   Note
///
/// This function is similar to `lift`. The `extra` argument is required to work-around `for<'a>` not otherwise
/// appropriately constraining its range of lifetime.
pub fn lift_with_mut<F, E, R>(root: R, extra: &mut E, fun: F) -> R
where
    F: for<'a> FnOnce(&'a R, &'a mut E) -> &'a mut R,
{
    //  The move into Manually Drop must happen _before_ to appease `MIRIFLAGS=-Zmiri-track-raw-pointers`
    let root = ManuallyDrop::new(root);
    let slot = fun(&root, extra) as *mut R as *mut ManuallyDrop<R>;

    //  Safety:
    //  -   `root` is still alive, hence any reference linked to `root` is _also_ alive.
    //  -   `root` is valid for reads.
    //  -   `slot` is valid for reads & writes.
    unsafe { replace(slot, &root as *const _) }
}

//  Replaces the content of `dest` with that of `src`, returns the content of `dest`.
//
//  #   Safety
//
//  -   `dest` is valid for both reads and writes.
//  -   `src` is valid for reads.
//
//  It is perfectly safe for `src` and `dest` to alias.
unsafe fn replace<T>(dest: *mut ManuallyDrop<T>, src: *const ManuallyDrop<T>) -> T {
    //  Swap, manually.
    let result = ptr::read(dest);
    ptr::copy(src, dest, 1);

    ManuallyDrop::into_inner(result)
}

#[cfg(test)]
mod tests {
    use core::{cell, mem};

    use super::*;

    //  Example from https://github.com/matthieu-m/static-rc/issues/14 by @noamtashma.
    //
    //  Using `UnsafeCell` instead of `GhostCell` to avoid 3rd-party dependency.
    fn lift_self_impl(cell_ref: &cell::UnsafeCell<Box<i32>>) -> &mut cell::UnsafeCell<Box<i32>> {
        //  cell_ref.borrow_mut(token) with GhostCell.
        let borrowed: &mut Box<i32> = unsafe { &mut *cell_ref.get() };

        //  GhostCell::from_mut
        let transmuted: &mut cell::UnsafeCell<Box<i32>> = unsafe { mem::transmute(borrowed) };

        transmuted
    }

    #[test]
    fn lift_self() {
        let cell = cell::UnsafeCell::new(Box::new(7));

        lift(cell, lift_self_impl);
    }

    #[test]
    fn lift_with_self() {
        let cell = cell::UnsafeCell::new(Box::new(7));

        lift_with(cell, &(), |cell_ref, _| lift_self_impl(cell_ref));
    }

    #[test]
    fn lift_with_mut_self() {
        let cell = cell::UnsafeCell::new(Box::new(7));

        lift_with_mut(cell, &mut (), |cell_ref, _| lift_self_impl(cell_ref));
    }

    //  Example from https://users.rust-lang.org/t/can-you-break-the-lift/58858/19 by @steffahn.
    //
    //  Arranged and annotated by yours truly.
    struct Struct;

    #[allow(clippy::mut_from_ref)]
    trait LeakBorrow {
        fn foo(&self) -> &mut Box<dyn LeakBorrow>;
    }

    impl LeakBorrow for cell::RefCell<Box<dyn LeakBorrow>> {
        fn foo(&self) -> &mut Box<dyn LeakBorrow> {
            let refmut: cell::RefMut<'_, _> = self.borrow_mut();
            let refmut_refmut: &mut cell::RefMut<'_, _> = Box::leak(Box::new(refmut));
            will_leak(&*refmut_refmut);

            &mut **refmut_refmut
        }
    }

    impl LeakBorrow for Struct {
        fn foo(&self) -> &mut Box<dyn LeakBorrow> {
            unimplemented!()
        }
    }

    #[test]
    fn lift_leak_borrow() {
        let root = Box::new(Struct) as Box<dyn LeakBorrow>;
        let root = Box::new(cell::RefCell::new(root)) as Box<dyn LeakBorrow>;
        lift(root, |b| b.foo());
    }

    #[test]
    fn lift_with_leak_borrow() {
        let root = Box::new(Struct) as Box<dyn LeakBorrow>;
        let root = Box::new(cell::RefCell::new(root)) as Box<dyn LeakBorrow>;
        lift_with(root, &(), |b, _| b.foo());
    }

    #[test]
    fn lift_with_mut_leak_borrow() {
        let root = Box::new(Struct) as Box<dyn LeakBorrow>;
        let root = Box::new(cell::RefCell::new(root)) as Box<dyn LeakBorrow>;
        lift_with_mut(root, &mut (), |b, _| b.foo());
    }

    //  Indicates that the pointed to memory will be leaked, to avoid it being reported.
    fn will_leak<T>(_t: &T) {
        #[cfg(miri)]
        {
            unsafe { miri_static_root(_t as *const _ as *const u8) };
        }
    }

    #[cfg(miri)]
    extern "Rust" {
        /// Miri-provided extern function to mark the block `ptr` points to as a "root"
        /// for some static memory. This memory and everything reachable by it is not
        /// considered leaking even if it still exists when the program terminates.
        ///
        /// `ptr` has to point to the beginning of an allocated block.
        fn miri_static_root(ptr: *const u8);
    }
} // mod tests
