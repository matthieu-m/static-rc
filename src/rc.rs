//! `StaticRc` is a compile-time referenced counted heap-allocated pointer.

use core::{
    any,
    borrow,
    cmp,
    convert,
    fmt,
    future,
    hash,
    iter,
    marker,
    mem::{self, MaybeUninit},
    ops,
    pin,
    ptr::{self, NonNull},
    task,
};

use alloc::boxed::Box;

#[cfg(feature = "nightly-async-iterator")]
use core::async_iter;

#[cfg(feature = "nightly-coerce-unsized")]
use core::ops::CoerceUnsized;

#[cfg(feature = "nightly-dispatch-from-dyn")]
use core::ops::DispatchFromDyn;

/// A compile-time reference-counted pointer.
///
/// The inherent methods of `StaticRc` are all associated functions to avoid conflicts with the the methods of the
/// inner type `T` which are brought into scope by the `Deref` implementation.
///
/// The parameters `NUM` and `DEN` DENote the ratio (`NUM / DEN`) of ownership of the pointer:
///
/// -   The ratio is always in the (0, 1] interval, that is: `NUM > 0` and `NUM <= DEN`.
/// -   When the ratio is equal to 1, that is when `NUM == DEN`, then the instance has full ownership of the pointee
///     and extra capabilities are unlocked.
pub struct StaticRc<T: ?Sized, const NUM: usize, const DEN: usize> {
    pointer: NonNull<T>,
}

impl<T, const N: usize> StaticRc<T, N, N> {
    /// Constructs a new `StaticRc<T, N, N>`.
    ///
    /// This uses `Box` under the hood.
    ///
    /// #   Example
    ///
    /// ```rust
    /// use static_rc::StaticRc;
    ///
    /// type Full = StaticRc<i32, 1, 1>;
    ///
    /// let rc = Full::new(42);
    /// assert_eq!(42, *rc);
    /// ```
    #[inline(always)]
    pub fn new(value: T) -> Self
    where
        AssertLeType!(1, N): Sized,
    {
        #[cfg(not(feature = "compile-time-ratio"))]
        assert!(N > 0);

        let pointer = NonNull::from(Box::leak(Box::new(value)));
        Self { pointer }
    }

    /// Constructs a new `Pin<StaticRc<T, N, N>>`.
    ///
    /// #   Example
    ///
    /// ```rust
    /// use static_rc::StaticRc;
    ///
    /// type Full = StaticRc<i32, 1, 1>;
    ///
    /// let rc = Full::pin(42);
    /// assert_eq!(42, *rc);
    /// ```
    #[inline(always)]
    pub fn pin(value: T) -> pin::Pin<Self>
    where
        AssertLeType!(1, N): Sized,
    {
        #[cfg(not(feature = "compile-time-ratio"))]
        assert!(N > 0);

        //  Safety:
        //  -   The `value` is placed on the heap, and cannot be moved out of the heap without full ownership.
        unsafe { pin::Pin::new_unchecked(Self::new(value)) }
    }

    /// Returns the inner value.
    ///
    /// #   Example
    ///
    /// ```rust
    /// use static_rc::StaticRc;
    ///
    /// type Full = StaticRc<i32, 1, 1>;
    ///
    /// let rc = Full::new(42);
    /// assert_eq!(42, Full::into_inner(rc));
    /// ```
    #[inline(always)]
    pub fn into_inner(this: Self) -> T {
        //  Safety:
        //  -   Ratio = 1, hence full ownership.
        let boxed = unsafe { Box::from_raw(this.pointer.as_ptr()) };
        mem::forget(this);

        *boxed
    }
}

impl<T: ?Sized, const N: usize> StaticRc<T, N, N> {
    /// Returns a mutable reference into the given `StaticRc`.
    ///
    /// #   Example
    ///
    /// ```rust
    /// use static_rc::StaticRc;
    ///
    /// type Full = StaticRc<i32, 1, 1>;
    ///
    /// let mut rc = Full::new(42);
    /// let r: &mut i32 = Full::get_mut(&mut rc);
    /// *r = 33;
    /// assert_eq!(33, *rc);
    /// ```
    #[inline(always)]
    pub fn get_mut(this: &mut Self) -> &mut T {
        //  Safety:
        //  -   Ratio = 1, hence full ownership.
        unsafe { this.pointer.as_mut() }
    }

    /// Returns the inner value, boxed
    ///
    /// #   Example
    ///
    /// ```rust
    /// use static_rc::StaticRc;
    ///
    /// type Full = StaticRc<i32, 1, 1>;
    ///
    /// let mut rc = Full::new(42);
    /// let boxed: Box<_> = Full::into_box(rc);
    /// assert_eq!(42, *boxed);
    /// ```
    #[inline(always)]
    pub fn into_box(this: Self) -> Box<T> {
        let pointer = this.pointer;
        mem::forget(this);

        //  Safety:
        //  -   Ratio = 1, hence full ownership.
        //  -   `pointer` was allocated by Box.
        unsafe { Box::from_raw(pointer.as_ptr()) }
    }
}

impl<T: ?Sized, const NUM: usize, const DEN: usize> StaticRc<T, NUM, DEN> {
    /// Consumes the `StaticRc`, returning the wrapped pointer.
    ///
    /// To avoid a memory leak, the pointer must be converted back to `Self` using `StaticRc::from_raw`.
    ///
    /// #   Example
    ///
    /// ```rust
    /// use static_rc::StaticRc;
    ///
    /// type Full = StaticRc<i32, 1, 1>;
    ///
    /// let rc = Full::new(42);
    /// let leaked = Full::into_raw(rc);
    ///
    /// let rc = unsafe { Full::from_raw(leaked) };
    /// assert_eq!(42, *rc);
    /// ```
    #[inline(always)]
    pub fn into_raw(this: Self) -> NonNull<T> {
        let pointer = this.pointer;
        mem::forget(this);

        pointer
    }

    /// Provides a raw pointer to the data.
    ///
    /// `StaticRc` is not consumed or affected in any way, the pointer is valid as long as there are shared owners of
    /// the value.
    ///
    /// #   Example
    ///
    /// ```rust
    /// use static_rc::StaticRc;
    ///
    /// type Full = StaticRc<i32, 1, 1>;
    ///
    /// let rc = Full::new(42);
    /// let pointer = Full::as_ptr(&rc);
    /// assert_eq!(42, unsafe { *pointer.as_ref() });
    /// ```
    #[inline(always)]
    pub fn as_ptr(this: &Self) -> NonNull<T> { this.pointer }

    /// Provides a reference to the data.
    ///
    /// #   Example
    ///
    /// ```rust
    /// use static_rc::StaticRc;
    ///
    /// type Full = StaticRc<i32, 1, 1>;
    ///
    /// let rc = Full::new(42);
    /// assert_eq!(42, *Full::get_ref(&rc));
    /// ```
    #[inline(always)]
    pub fn get_ref(this: &Self) -> &T {
        //  Safety:
        //  -   The data is valid for as long as `this` lives.
        unsafe { this.pointer.as_ref() }
    }

    /// Constructs a `StaticRc<T, NUM, DEN>` from a raw pointer.
    ///
    /// #   Safety
    ///
    /// The raw pointer must have been previously returned by a call to `StaticRc<U, N, D>::into_raw`:
    ///
    /// -   If `U` is different from `T`, then specific restrictions on size and alignment apply. See `mem::transmute`
    ///     for the restrictions applying to transmuting references.
    /// -   If `N / D` is different from `NUM / DEN`, then specific restrictions apply. The user is responsible for
    ///     ensuring proper management of the ratio of shares, and ultimately that the value is not dropped twice.
    ///
    /// #   Example
    ///
    /// ```rust
    /// use static_rc::StaticRc;
    ///
    /// type Full = StaticRc<i32, 2, 2>;
    /// type Half = StaticRc<i32, 1, 2>;
    ///
    /// let rc = Full::new(42);
    /// let leaked = Full::into_raw(rc);
    ///
    /// let (one, two) = unsafe { (Half::from_raw(leaked), Half::from_raw(leaked)) };
    /// let rc = Full::join(one, two);
    ///
    /// assert_eq!(42, *rc);
    /// ```
    #[inline(always)]
    pub unsafe fn from_raw(pointer: NonNull<T>) -> Self
    where
        AssertLeType!(1, NUM): Sized,
    {
        #[cfg(not(feature = "compile-time-ratio"))]
        assert!(NUM > 0);

        Self { pointer }
    }

    /// Returns true if the two `StaticRc` point to the same allocation.
    ///
    /// #   Example
    ///
    /// ```rust
    /// use static_rc::StaticRc;
    ///
    /// type Full = StaticRc<i32, 2, 2>;
    ///
    /// let rc = Full::new(42);
    /// let (one, two) = Full::split::<1, 1>(rc);
    ///
    /// assert!(StaticRc::ptr_eq(&one, &two));
    ///
    /// Full::join(one, two);
    /// ```
    #[inline(always)]
    pub fn ptr_eq<const N: usize, const D: usize>(this: &Self, other: &StaticRc<T, N, D>) -> bool {
        ptr::eq(StaticRc::as_ptr(this).as_ptr(), StaticRc::as_ptr(other).as_ptr())
    }

    /// Adjusts the NUMerator and DENUMerator of the ratio of the instance, preserving the ratio.
    ///
    /// #   Panics
    ///
    /// If the compile-time-ratio feature is not used, and the ratio is not preserved; that is `N / D <> NUM / DEN`.
    ///
    /// #   Example
    ///
    /// ```rust
    /// use static_rc::StaticRc;
    ///
    /// type Full = StaticRc<i32, 2, 2>;
    ///
    /// let rc = Full::new(42);
    /// let rc = Full::adjust::<1, 1>(rc);
    ///
    /// assert_eq!(42, *rc);
    /// ```
    #[inline(always)]
    pub fn adjust<const N: usize, const D: usize>(this: Self) -> StaticRc<T, N, D>
    where
        AssertLeType!(1, N): Sized,
        AssertEqType!(N * DEN, NUM * D): Sized,
    {
        #[cfg(not(feature = "compile-time-ratio"))]
        {
            assert!(N > 0);
            assert_eq!(NUM * D, N * DEN, "{NUM} / {DEN} != {N} / {D}");
        }

        let pointer = this.pointer;
        mem::forget(this);

        StaticRc { pointer }
    }

    /// Converts an instance into a [`StaticRcRef`](super::StaticRcRef).
    ///
    /// The current instance is mutably borrowed for the duration the result can be used.
    ///
    /// #   Example
    ///
    /// ```rust
    /// use static_rc::StaticRc;
    /// use static_rc::StaticRcRef;
    /// let rc: StaticRc<_, 2, 2> = StaticRc::new(5);
    /// let (mut rc1, mut rc2) = StaticRc::split::<1, 1>(rc);
    /// {
    ///     // Modify without moving `rc1`, `rc2`.
    ///     let rcref1 = StaticRc::as_rcref(&mut rc1);
    ///     let rcref2 = StaticRc::as_rcref(&mut rc2);
    ///     let mut rcref_owning: StaticRcRef<_, 2, 2> = StaticRcRef::join(rcref1, rcref2);
    ///     *rcref_owning = 9;
    ///     // Refs not used anymore, original rcs can be used again
    /// }
    /// let rc: StaticRc<_, 2, 2> = StaticRc::join(rc1, rc2);
    /// assert_eq!(*rc, 9);
    /// assert_eq!(*StaticRc::into_box(rc), 9);
    /// ```
    #[inline(always)]
    pub fn as_rcref<'a>(this: &'a mut Self) -> super::StaticRcRef<'a, T, NUM, DEN>
    where
        AssertLeType!(1, NUM): Sized,
    {
        //  Safety:
        //  -   The public documentation says that `StaticRcRef::from_raw`
        //      can only be called on pointers returned from `StaticRcRef::into_raw`.
        //      which this isn't.
        //  -   However, internally the library knows that `rc` and `rcref` have the same invariants:
        //      -  `this.pointer` is a valid aligned pointer into a valid value of `T`.
        //  -   The result is only usable for lifetime `'a`, and for the duration
        //      of the lifetime `'a` `this` is mutably borrowed.
        //  -   `this` has NUM/DEN of the ownership. So it can lend NUM/DEN
        //      of the right to mutate the value. Therefore, this is semantically sound
        //      according to the general principle of this library.
        //
        //  This is safe for generally the same reason `StaticRcRef::reborrow` is safe.
        //
        //  `StaticRcRef::from_raw` has to have a comment documenting
        //  internally that such a use is allowed.
        let ptr = this.pointer;
        unsafe {
            super::StaticRcRef::from_raw(ptr)
        }
    }

    /// Splits the current instance into two instances with the specified NUMerators.
    ///
    /// #   Panics
    ///
    /// If the compile-time-ratio feature is not used, and the ratio is not preserved; that is `A + B <> NUM`.
    ///
    /// #   Example
    ///
    /// ```rust
    /// use static_rc::StaticRc;
    ///
    /// type Full = StaticRc<i32, 2, 2>;
    /// type Half = StaticRc<i32, 1, 2>;
    ///
    /// let rc = Full::new(42);
    /// let (one, two): (Half, Half) = Full::split::<1, 1>(rc);
    ///
    /// assert_eq!(42, *one);
    ///
    /// Full::join(one, two);
    /// ```
    #[inline(always)]
    pub fn split<const A: usize, const B: usize>(this: Self) -> (StaticRc<T, A, DEN>, StaticRc<T, B, DEN>)
    where
        AssertLeType!(1, A): Sized,
        AssertLeType!(1, B): Sized,
        AssertEqType!(A + B, NUM): Sized,
    {
        #[cfg(not(feature = "compile-time-ratio"))]
        {
            assert!(A > 0);
            assert!(B > 0);
            assert_eq!(NUM, A + B, "{NUM} != {A} + {B}");
        }

        let pointer = this.pointer;
        mem::forget(this);

        (StaticRc { pointer }, StaticRc { pointer })
    }

    /// Splits the current instance into `DIM` instances with the specified Numerators and Denominators.
    ///
    /// #   Panics
    ///
    /// If the compile-time-ratio feature is not used, and the ratio is not preserved; that is `N * DIM <> NUM`.
    ///
    /// #   Example
    ///
    /// ```rust
    /// use static_rc::StaticRc;
    ///
    /// type Full = StaticRc<i32, 2, 2>;
    ///
    /// let rc = Full::new(42);
    /// let array = Full::split_array::<1, 2>(rc);
    ///
    /// assert_eq!(42, *array[0]);
    ///
    /// Full::join_array(array);
    /// ```
    #[inline(always)]
    pub fn split_array<const N: usize, const DIM: usize>(this: Self) -> [StaticRc<T, N, DEN>; DIM]
    where
        AssertEqType!(N * DIM, NUM ): Sized,
        AssertLeType!(mem::size_of::<[StaticRc<T, N, DEN>; DIM]>(), usize::MAX / 2 + 1): Sized,
    {
        #[cfg(not(feature = "compile-time-ratio"))]
        assert_eq!(NUM, N * DIM, "{NUM} != {N} * {DIM}");

        #[cfg(not(feature = "compile-time-ratio"))]
        assert!(mem::size_of::<[StaticRc<T, N, DEN>; DIM]>() <= (isize::MAX as usize),
            "Size of result ({}) exceeeds isize::MAX", mem::size_of::<[StaticRc<T, N, DEN>; DIM]>());

        let pointer = this.pointer;
        mem::forget(this);

        let mut array = MaybeUninit::uninit();

        for i in 0..DIM {
            //  Safety:
            //  -   `destination` within bounds of allocated array (< DIM).
            //  -   Offset doesn't overflow `isize`, as per array-size assertion.
            //  -   Offset doesn't wrap around, as per array-size assertion.
            let destination = unsafe { (array.as_mut_ptr() as *mut StaticRc<T, N, DEN>).add(i) };

            //  Safety:
            //  -   `destination` is valid for writes.
            //  -   `destination` is correctly aligned.
            unsafe { ptr::write(destination, StaticRc { pointer }); }
        }

        //  Safety:
        //  -   Every element of the array is now initialized.
        unsafe { array.assume_init() }
    }

    /// Joins two instances into a single instance.
    ///
    /// #   Panics
    ///
    /// If the two instances do no point to the same allocation, as determined by `StaticRc::ptr_eq`.
    ///
    /// If the compile-time-ratio feature is not used and the ratio is not preserved; that is `A + B <> NUM`.
    ///
    /// #   Example
    ///
    /// ```rust
    /// use static_rc::StaticRc;
    ///
    /// type Full = StaticRc<i32, 3, 3>;
    ///
    /// let rc = Full::new(42);
    /// let (one, two) = Full::split::<1, 2>(rc);
    ///
    /// let rc = Full::join(one, two);
    /// assert_eq!(42, *rc);
    /// ```
    #[inline(always)]
    pub fn join<const A: usize, const B: usize>(left: StaticRc<T, A, DEN>, right: StaticRc<T, B, DEN>) -> Self
    where
        AssertEqType!(NUM, A + B): Sized,
    {
        let (left, right) = Self::validate_pair(left, right);

        //  Safety:
        //  -   `left` and `right` point to the same pointer.
        unsafe { Self::join_impl(left, right) }
    }

    /// Joins two instances into a single instance without checking whether they point to the same allocation.
    ///
    /// Unless `compile-time-ratio` is activated, the ratios are checked nevertheless.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that those instances point to the same allocation.
    ///
    /// #   Panics
    ///
    /// If the compile-time-ratio feature is not used and the ratio is not preserved; that is `A + B <> NUM`.
    ///
    /// In debug, if the two instances do not point to the same allocation, as determined by `StaticRc::ptr_eq`.
    ///
    /// #   Example
    ///
    /// ```rust
    /// use static_rc::StaticRc;
    ///
    /// type Full = StaticRc<i32, 3, 3>;
    ///
    /// let rc = Full::new(42);
    /// let (one, two) = Full::split::<1, 2>(rc);
    ///
    /// let rc = unsafe { Full::join_unchecked(one, two) };
    /// assert_eq!(42, *rc);
    /// ```
    #[inline(always)]
    pub unsafe fn join_unchecked<const A: usize, const B: usize>(
        left: StaticRc<T, A, DEN>,
        right: StaticRc<T, B, DEN>,
    ) -> Self
    where
        AssertEqType!(NUM, A + B): Sized,
    {
        #[cfg(debug_assertions)]
        let (left, right) = Self::validate_pair(left, right);

        Self::join_impl(left, right)
    }

    /// Joins DIM instances into a single instance.
    ///
    /// #   Panics
    ///
    /// If all instances do not point to the same allocation, as determined by `StaticRc::ptr_eq`.
    ///
    /// If the compile-time-ratio feature is not used and the ratio is not preserved; that is `N * DIM <> NUM`.
    ///
    /// #   Example
    ///
    /// ```rust
    /// use static_rc::StaticRc;
    ///
    /// type Full = StaticRc<i32, 2, 2>;
    ///
    /// let rc = Full::new(42);
    /// let array = Full::split_array::<1, 2>(rc);
    /// let rc = Full::join_array(array);
    ///
    /// assert_eq!(42, *rc);
    /// ```
    #[inline(always)]
    pub fn join_array<const N: usize, const DIM: usize>(array: [StaticRc<T, N, DEN>; DIM]) -> Self
    where
        AssertLeType!(1, NUM): Sized,
        AssertEqType!(N * DIM, NUM): Sized,
    {
        let array = Self::validate_array(array);

        unsafe { Self::join_array_impl(array) }
    }

    /// Joins DIM instances into a single instance.
    ///
    /// #   Safety
    ///
    /// All instances must point to the same allocation, as determined by `StaticRc::ptr_eq`.
    ///
    /// #   Panics
    ///
    /// If the compile-time-ratio feature is not used and the ratio is not preserved; that is `N * DIM <> NUM`.
    ///
    /// In debug, if all instances do not point to the same allocation, as determined by `StaticRc::ptr_eq`.
    ///
    /// #   Example
    ///
    /// ```rust
    /// use static_rc::StaticRc;
    ///
    /// type Full = StaticRc<i32, 2, 2>;
    ///
    /// let rc = Full::new(42);
    /// let array = Full::split_array::<1, 2>(rc);
    /// let rc = unsafe { Full::join_array_unchecked(array) };
    ///
    /// assert_eq!(42, *rc);
    /// ```
    #[inline(always)]
    pub unsafe fn join_array_unchecked<const N: usize, const DIM: usize>(array: [StaticRc<T, N, DEN>; DIM])
        -> Self
    where
        AssertLeType!(1, NUM): Sized,
        AssertEqType!(N * DIM, NUM): Sized,
    {
        #[cfg(debug_assertions)]
        let array = Self::validate_array(array);

        Self::join_array_impl(array)
    }

    //  Internal; joins without validating origin.
    #[inline(always)]
    unsafe fn join_impl<const A: usize, const B: usize>(
        left: StaticRc<T, A, DEN>,
        right: StaticRc<T, B, DEN>,
    ) -> Self
    where
        AssertEqType!(NUM, A + B): Sized,
    {
        #[cfg(not(feature = "compile-time-ratio"))]
        #[allow(non_fmt_panics)]
        if NUM != A + B {
            mem::forget(left);
            mem::forget(right);

            panic!("{NUM} != {A} + {B}");
        }

        let pointer = left.pointer;
        mem::forget(left);
        mem::forget(right);

        Self { pointer }
    }

    //  Internal; joins without validating origin.
    #[inline(always)]
    unsafe fn join_array_impl<const N: usize, const DIM: usize>(array: [StaticRc<T, N, DEN>; DIM])
        -> Self
    where
        AssertLeType!(1, NUM): Sized,
        AssertEqType!(N * DIM, NUM): Sized,
    {
        #[cfg(not(feature = "compile-time-ratio"))]
        #[allow(non_fmt_panics)]
        {
            if NUM == 0 {
                mem::forget(array);

                panic!("NUM == 0");
            }

            if NUM != N * DIM {
                mem::forget(array);

                panic!("{NUM} != {N} * {DIM}");
            }
        }

        let pointer = array[0].pointer;
        mem::forget(array);

        Self { pointer, }
    }

    fn validate_pair<const A: usize, const B: usize>(left: StaticRc<T, A, DEN>, right: StaticRc<T, B, DEN>)
        -> (StaticRc<T, A, DEN>, StaticRc<T, B, DEN>)
    {
        if StaticRc::ptr_eq(&left, &right) {
            return (left, right);
        }

        let left = StaticRc::into_raw(left);
        let right = StaticRc::into_raw(right);

        panic!("Cannot join pair with multiple origins: {:?} != {:?}", left.as_ptr(), right.as_ptr());
    }

    fn validate_array<const N: usize, const DIM: usize>(array: [StaticRc<T, N, DEN>; DIM]) -> [StaticRc<T, N, DEN>; DIM] {
        let first = &array[0];
        let divergent = array[1..].iter().find(|e| !StaticRc::ptr_eq(first, e));

        if let Some(divergent) = divergent {
            let first = first.pointer.as_ptr();
            let divergent = divergent.pointer.as_ptr();

            mem::forget(array);

            panic!("Cannot join array with multiple origins: {:?} != {:?}", first, divergent);
        }

        array
    }
}

impl<const NUM: usize, const DEN: usize> StaticRc<dyn any::Any, NUM, DEN> {
    /// Attempts to downcast `Self` to a concrete type.
    pub fn downcast<T: any::Any>(self) -> Result<StaticRc<T, NUM, DEN>, Self> {
        if Self::get_ref(&self).is::<T>() {
            let pointer = Self::into_raw(self).cast::<T>();
            Ok(StaticRc { pointer })
        } else {
            Err(self)
        }
    }
}

impl<T: ?Sized, const NUM: usize, const DEN: usize> Drop for StaticRc<T, NUM, DEN> {
    #[inline(always)]
    fn drop(&mut self) {
        debug_assert_eq!(NUM, DEN, "{NUM} != {DEN}");

        if NUM == DEN {
            //  Safety:
            //  -   Ratio = 1, hence full ownership.
            //  -   `self.pointer` was allocated by Box.
            let _ = unsafe { Box::from_raw(self.pointer.as_ptr()) };
        }
    }
}

impl<T: ?Sized, const N: usize> convert::AsMut<T> for StaticRc<T, N, N> {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut T { Self::get_mut(self) }
}

impl<T: ?Sized, const NUM: usize, const DEN: usize> convert::AsRef<T> for StaticRc<T, NUM, DEN> {
    #[inline(always)]
    fn as_ref(&self) -> &T { Self::get_ref(self) }
}

impl<T: ?Sized, const NUM: usize, const DEN: usize> borrow::Borrow<T> for StaticRc<T, NUM, DEN> {
    #[inline(always)]
    fn borrow(&self) -> &T { Self::get_ref(self) }
}

impl<T: ?Sized, const N: usize> borrow::BorrowMut<T> for StaticRc<T, N, N> {
    #[inline(always)]
    fn borrow_mut(&mut self) -> &mut T { Self::get_mut(self) }
}

#[cfg(feature = "nightly-coerce-unsized")]
impl<T, U, const NUM: usize, const DEN: usize> CoerceUnsized<StaticRc<U, NUM, DEN>> for StaticRc<T, NUM, DEN>
where
    T: ?Sized + marker::Unsize<U>,
    U: ?Sized,
{}

impl<T: ?Sized + fmt::Debug, const NUM: usize, const DEN: usize> fmt::Debug for StaticRc<T, NUM, DEN> {
    #[inline(always)]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        fmt::Debug::fmt(Self::get_ref(self), f)
    }
}

impl<T: Default, const N: usize> Default for StaticRc<T, N, N>
where
    AssertLeType!(1, N): Sized,
{
    #[inline(always)]
    fn default() -> Self { Self::new(T::default()) }
}

impl<T: ?Sized, const NUM: usize, const DEN: usize> ops::Deref for StaticRc<T, NUM, DEN> {
    type Target = T;

    #[inline(always)]
    fn deref(&self) -> &T { Self::get_ref(self) }
}

impl<T: ?Sized, const N: usize> ops::DerefMut for StaticRc<T, N, N> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut T { Self::get_mut(self) }
}

impl<T: ?Sized + fmt::Display, const NUM: usize, const DEN: usize> fmt::Display for StaticRc<T, NUM, DEN> {
    #[inline(always)]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        fmt::Display::fmt(Self::get_ref(self), f)
    }
}

#[cfg(feature = "nightly-dispatch-from-dyn")]
impl<T, U, const NUM: usize, const DEN: usize> DispatchFromDyn<StaticRc<U, NUM, DEN>> for StaticRc<T, NUM, DEN>
where
    T: ?Sized + marker::Unsize<U>,
    U: ?Sized,
{}

impl<I: iter::DoubleEndedIterator + ?Sized, const N: usize> iter::DoubleEndedIterator for StaticRc<I, N, N> {
    #[inline(always)]
    fn next_back(&mut self) -> Option<I::Item> { Self::get_mut(self).next_back() }

    #[inline(always)]
    fn nth_back(&mut self, n: usize) -> Option<I::Item> { Self::get_mut(self).nth_back(n) }
}

impl<T: ?Sized + cmp::Eq, const NUM: usize, const DEN: usize> cmp::Eq for StaticRc<T, NUM, DEN> {}

impl<I: iter::ExactSizeIterator + ?Sized, const N: usize> iter::ExactSizeIterator for StaticRc<I, N, N> {
    #[inline(always)]
    fn len(&self) -> usize { Self::get_ref(self).len() }
}

impl<T: ?Sized, const N: usize> From<Box<T>> for StaticRc<T, N, N> {
    #[inline(always)]
    fn from(value: Box<T>) -> Self {
        let pointer = NonNull::from(Box::leak(value));
        Self { pointer }
    }
}

impl<T: Copy, const N: usize> From<&'_ [T]> for StaticRc<[T], N, N> {
    #[inline(always)]
    fn from(value: &[T]) -> Self { Self::from(Box::from(value)) }
}

impl<const N: usize> From<&'_ str> for StaticRc<str, N, N> {
    #[inline(always)]
    fn from(value: &str) -> Self { Self::from(Box::from(value)) }
}

impl<T, const LEN: usize, const N: usize> From<[T; LEN]> for StaticRc<[T], N, N> {
    #[inline(always)]
    fn from(value: [T; LEN]) -> Self { Self::from(Box::from(value)) }
}

impl<T: Copy, const N: usize> From<alloc::borrow::Cow<'_, [T]>> for StaticRc<[T], N, N> {
    #[inline(always)]
    fn from(value: alloc::borrow::Cow<'_, [T]>) -> Self { Self::from(Box::from(value)) }
}

impl<const N: usize> From<alloc::borrow::Cow<'_, str>> for StaticRc<str, N, N> {
    #[inline(always)]
    fn from(value: alloc::borrow::Cow<'_, str>) -> Self { Self::from(Box::from(value)) }
}

impl<const N: usize> From<alloc::string::String> for StaticRc<str, N, N> {
    #[inline(always)]
    fn from(value: alloc::string::String) -> Self { Self::from(Box::from(value)) }
}

impl<T, const N: usize> From<T> for StaticRc<T, N, N> {
    #[inline(always)]
    fn from(value: T) -> Self { Self::from(Box::from(value)) }
}

impl<T, const N: usize> From<alloc::vec::Vec<T>> for StaticRc<[T], N, N> {
    #[inline(always)]
    fn from(value: alloc::vec::Vec<T>) -> Self { Self::from(Box::from(value)) }
}

impl<T, const N: usize> From<StaticRc<[T], N, N>> for alloc::vec::Vec<T> {
    #[inline(always)]
    fn from(value: StaticRc<[T], N, N>) -> Self { Self::from(StaticRc::into_box(value)) }
}

impl<T: ?Sized, const N: usize> From<StaticRc<T, N, N>> for alloc::rc::Rc<T> {
    #[inline(always)]
    fn from(value: StaticRc<T, N, N>) -> Self { Self::from(StaticRc::into_box(value)) }
}

impl<T: ?Sized, const N: usize> From<StaticRc<T, N, N>> for alloc::sync::Arc<T> {
    #[inline(always)]
    fn from(value: StaticRc<T, N, N>) -> Self { Self::from(StaticRc::into_box(value)) }
}

impl<const N: usize> From<StaticRc<str, N, N>> for alloc::string::String {
    #[inline(always)]
    fn from(value: StaticRc<str, N, N>) -> Self { Self::from(StaticRc::into_box(value)) }
}

impl<const NUM: usize, const DEN: usize> From<StaticRc<str, NUM, DEN>> for StaticRc<[u8], NUM, DEN> {
    #[inline(always)]
    fn from(value: StaticRc<str, NUM, DEN>) -> Self {
        let pointer = value.pointer.as_ptr() as *mut [u8];
        mem::forget(value);

        //  Safety:
        //  -   `value.pointer` was not null, hence `pointer` is not null.
        debug_assert!(!pointer.is_null());
        let pointer = unsafe { NonNull::new_unchecked(pointer) };

        Self { pointer }
    }
}

impl<const N: usize> iter::FromIterator<StaticRc<str, N, N>> for alloc::string::String {
    #[inline(always)]
    fn from_iter<I: IntoIterator<Item = StaticRc<str, N, N>>>(iter: I) -> Self {
        Self::from_iter(iter.into_iter().map(StaticRc::into_box))
    }
}

impl<T, const N: usize> iter::FromIterator<T> for StaticRc<[T], N, N> {
    #[inline(always)]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self { Self::from(Box::from_iter(iter)) }
}

impl<I: iter::FusedIterator + ?Sized, const N: usize> iter::FusedIterator for StaticRc<I, N, N> {}

impl<F: ?Sized + future::Future + marker::Unpin, const N: usize> future::Future for StaticRc<F, N, N> {
    type Output = F::Output;

    fn poll(mut self: pin::Pin<&mut Self>, cx: &mut task::Context<'_>) -> task::Poll<Self::Output> {
        F::poll(pin::Pin::new(&mut *self), cx)
    }
}

#[cfg(feature = "nightly-generator-trait")]
impl<G: ?Sized + ops::Generator<R> + marker::Unpin, R, const N: usize> ops::Generator<R> for StaticRc<G, N, N> {
    type Yield = G::Yield;
    type Return = G::Return;

        fn resume(mut self: pin::Pin<&mut Self>, arg: R) -> ops::GeneratorState<Self::Yield, Self::Return> {
            G::resume(pin::Pin::new(&mut *self), arg)
        }
}

#[cfg(feature = "nightly-generator-trait")]
impl<G: ?Sized + ops::Generator<R>, R, const N: usize> ops::Generator<R> for pin::Pin<StaticRc<G, N, N>> {
    type Yield = G::Yield;
    type Return = G::Return;

        fn resume(mut self: pin::Pin<&mut Self>, arg: R) -> ops::GeneratorState<Self::Yield, Self::Return> {
            G::resume((*self).as_mut(), arg)
        }
}

impl<T: ?Sized + hash::Hash, const NUM: usize, const DEN: usize> hash::Hash for StaticRc<T, NUM, DEN> {
    #[inline(always)]
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        Self::get_ref(self).hash(state);
    }
}

impl<I: iter::Iterator + ?Sized, const N: usize> iter::Iterator for StaticRc<I, N, N> {
    type Item = I::Item;

    #[inline(always)]
    fn next(&mut self) -> Option<I::Item> { Self::get_mut(self).next() }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) { Self::get_ref(self).size_hint() }

    #[inline(always)]
    fn nth(&mut self, n: usize) -> Option<I::Item> { Self::get_mut(self).nth(n) }

    #[inline(always)]
    fn last(self) -> Option<I::Item> { Self::into_box(self).last() }
}

impl<T: ?Sized + cmp::Ord, const NUM: usize, const DEN: usize> cmp::Ord for StaticRc<T, NUM, DEN> {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        if Self::ptr_eq(self, other) {
            cmp::Ordering::Equal
        } else {
            Self::get_ref(self).cmp(Self::get_ref(other))
        }
    }
}

impl<T, const NUM: usize, const DEN: usize, const N: usize, const D: usize> cmp::PartialEq<StaticRc<T, N, D>>
    for StaticRc<T, NUM, DEN>
where
    T: ?Sized + PartialEq<T>
{
    #[inline(always)]
    fn eq(&self, other: &StaticRc<T, N, D>) -> bool { Self::get_ref(self).eq(StaticRc::get_ref(other)) }

    #[inline(always)]
    #[allow(clippy::partialeq_ne_impl)]
    fn ne(&self, other: &StaticRc<T, N, D>) -> bool { Self::get_ref(self).ne(StaticRc::get_ref(other)) }
}

impl<T, const NUM: usize, const DEN: usize, const N: usize, const D: usize> cmp::PartialOrd<StaticRc<T, N, D>>
    for StaticRc<T, NUM, DEN>
where
    T: ?Sized + PartialOrd<T>
{
    #[inline(always)]
    fn partial_cmp(&self, other: &StaticRc<T, N, D>) -> Option<cmp::Ordering> {
        Self::get_ref(self).partial_cmp(StaticRc::get_ref(other))
    }

    #[inline(always)]
    fn lt(&self, other: &StaticRc<T, N, D>) -> bool {
        Self::get_ref(self).lt(StaticRc::get_ref(other))
    }

    #[inline(always)]
    fn le(&self, other: &StaticRc<T, N, D>) -> bool {
        Self::get_ref(self).le(StaticRc::get_ref(other))
    }

    #[inline(always)]
    fn gt(&self, other: &StaticRc<T, N, D>) -> bool {
        Self::get_ref(self).gt(StaticRc::get_ref(other))
    }

    #[inline(always)]
    fn ge(&self, other: &StaticRc<T, N, D>) -> bool {
        Self::get_ref(self).ge(StaticRc::get_ref(other))
    }
}

impl<T: ?Sized, const NUM: usize, const DEN: usize> fmt::Pointer for StaticRc<T, NUM, DEN> {
    #[inline(always)]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Pointer::fmt(&Self::as_ptr(self).as_ptr(), f)
    }
}

#[cfg(feature = "nightly-async-iterator")]
impl<S: ?Sized + async_iter::AsyncIterator + marker::Unpin, const N: usize> async_iter::AsyncIterator for StaticRc<S, N, N> {
    type Item = S::Item;

    fn poll_next(mut self: pin::Pin<&mut Self>, cx: &mut task::Context<'_>) -> task::Poll<Option<Self::Item>> {
        pin::Pin::new(&mut **self).poll_next(cx)
    }

    fn size_hint(&self) -> (usize, Option<usize>) { (**self).size_hint() }
}

impl<T: ?Sized, const NUM: usize, const DEN: usize> marker::Unpin for StaticRc<T, NUM, DEN> {}

unsafe impl<T: ?Sized + marker::Send, const NUM: usize, const DEN: usize> marker::Send for StaticRc<T, NUM, DEN> {}

unsafe impl<T: ?Sized + marker::Sync, const NUM: usize, const DEN: usize> marker::Sync for StaticRc<T, NUM, DEN> {}

#[doc(hidden)]
pub mod compile_tests {

/// ```compile_fail,E0505
/// let mut a = String::from("foo");
/// let mut rc = static_rc::StaticRc::<_,1,1>::new(a);
///
/// let mut reborrow = static_rc::StaticRc::as_rcref(&mut rc);
/// std::mem::drop(rc);
/// assert_eq!(*reborrow, "foo"); // This should fail to compile.
/// ```
pub fn rc_reborrow_and_move() {}

/// ```compile_fail,E0502
/// let mut a = String::from("foo");
/// let mut rc = static_rc::StaticRc::<_,1,1>::new(a);
///
/// let mut reborrow = static_rc::StaticRc::as_rcref(&mut rc);
/// assert_eq!(*rc, "foo");
/// assert_eq!(*reborrow, "foo"); // This should fail to compile.
/// ```
pub fn rc_reborrow_and_use() {}

} // mod compile_tests

#[doc(hidden)]
#[cfg(feature = "compile-time-ratio")]
pub mod compile_ratio_tests {

/// ```compile_fail,E0080
/// type Zero = static_rc::StaticRc<i32, 0, 0>;
///
/// Zero::new(42);
/// ```
pub fn rc_new_zero() {}

/// ```compile_fail,E0080
/// type Zero = static_rc::StaticRc<i32, 0, 0>;
///
/// Zero::pin(42);
/// ```
pub fn rc_pin_zero() {}

/// ```compile_fail,E0080
/// type Zero = static_rc::StaticRc<i32, 0, 0>;
///
/// let pointer = core::ptr::NonNull::dangling();
///
/// unsafe { Zero::from_raw(pointer) };
/// ```
pub fn rc_from_raw_zero() {}

/// ```compile_fail,E0080
/// type One = static_rc::StaticRc<i32, 1, 1>;
///
/// let rc = One::new(42);
///
/// One::adjust::<0, 0>(rc);
/// ```
pub fn rc_adjust_zero() {}

/// ```compile_fail,E0080
/// type One = static_rc::StaticRc<i32, 1, 1>;
///
/// let rc = One::new(42);
///
/// One::adjust::<2, 3>(rc);
/// ```
pub fn rc_adjust_ratio() {}

/// ```compile_fail,E0080
/// type Two = static_rc::StaticRc<i32, 2, 2>;
///
/// let rc = Two::new(42);
///
/// Two::split::<0, 2>(rc);
/// ```
pub fn rc_split_zero_first() {}

/// ```compile_fail,E0080
/// type Two = static_rc::StaticRc<i32, 2, 2>;
///
/// let rc = Two::new(42);
///
/// Two::split::<2, 0>(rc);
/// ```
pub fn rc_split_zero_second() {}

/// ```compile_fail,E0080
/// type Two = static_rc::StaticRc<i32, 2, 2>;
///
/// let rc = Two::new(42);
///
/// Two::split::<1, 2>(rc);
/// ```
pub fn rc_split_sum() {}

/// ```compile_fail,E0080
/// type Two = static_rc::StaticRc<i32, 2, 2>;
///
/// let rc = Two::new(42);
///
/// Two::split_array::<2, 2>(rc);
/// ```
pub fn rc_split_array_ratio() {}

/// ```compile_fail,E0080
/// type Two = static_rc::StaticRc<i32, 2, 2>;
///
/// let rc = Two::new(42);
/// let (one, two) = Two::split::<1, 1>(rc);
///
/// static_rc::StaticRc::<_, 1, 2>::join(one, two);
/// ```
pub fn rc_join_ratio() {}

/// ```compile_fail,E0080
/// type Two = static_rc::StaticRc<i32, 2, 2>;
///
/// let rc = Two::new(42);
/// let (one, two) = Two::split::<1, 1>(rc);
///
/// unsafe { static_rc::StaticRc::<_, 1, 2>::join_unchecked(one, two) };
/// ```
pub fn rc_join_unchecked_ratio() {}

/// ```compile_fail,E0080
/// type Two = static_rc::StaticRc<i32, 2, 2>;
///
/// let rc = Two::new(42);
/// let array: [_; 2] = Two::split_array::<1, 2>(rc);
///
/// static_rc::StaticRc::<_, 1, 2>::join_array(array);
/// ```
pub fn rc_join_array_ratio() {}

/// ```compile_fail,E0080
/// type Two = static_rc::StaticRc<i32, 2, 2>;
///
/// let rc = Two::new(42);
/// let array: [_; 2] = Two::split_array::<1, 2>(rc);
///
/// unsafe { static_rc::StaticRc::<_, 1, 2>::join_array_unchecked(array) };
/// ```
pub fn rc_join_array_unchecked_ratio() {}

} // mod compile_ratio_tests

#[cfg(all(test, not(feature = "compile-time-ratio")))]
mod panic_ratio_tests {

use super::*;

type Zero = StaticRc<i32, 0, 0>;
type One = StaticRc<i32, 1, 1>;
type Two = StaticRc<i32, 2, 2>;

#[test]
#[should_panic]
fn rc_new_zero() {
    Zero::new(42);
}

#[test]
#[should_panic]
fn rc_pin_zero() {
    Zero::pin(42);
}

#[test]
#[should_panic]
fn rc_from_raw_zero() {
    let pointer = NonNull::dangling();

    unsafe { Zero::from_raw(pointer) };
}

#[test]
#[should_panic]
fn rc_adjust_zero() {
    let rc = One::new(42);

    One::adjust::<0, 0>(rc);
}

#[test]
#[should_panic]
fn rc_adjust_ratio() {
    let rc = One::new(42);

    One::adjust::<2, 3>(rc);
}

#[test]
#[should_panic]
fn rc_split_zero_first() {
    let rc = Two::new(42);

    Two::split::<0, 2>(rc);
}

#[test]
#[should_panic]
fn rc_split_zero_second() {
    let rc = Two::new(42);

    Two::split::<0, 2>(rc);
}

#[test]
#[should_panic]
fn rc_split_sum() {
    let rc = Two::new(42);

    Two::split::<1, 2>(rc);
}

#[test]
#[should_panic]
fn rc_split_array_ratio() {
    let rc = Two::new(42);

    Two::split_array::<2, 2>(rc);
}

#[test]
#[should_panic]
fn rc_join_ratio() {
    let rc = Two::new(42);
    will_leak(&rc);

    let (one, two) = Two::split::<1, 1>(rc);

    StaticRc::<_, 1, 2>::join(one, two);
}

#[test]
#[should_panic]
fn rc_join_different() {
    let (rc, other) = (Two::new(42), Two::new(33));
    will_leak(&rc);
    will_leak(&other);

    let (one, two) = Two::split::<1, 1>(rc);
    let (other_one, other_two) = Two::split::<1, 1>(other);

    mem::forget([two, other_two]);

    Two::join(one, other_one);
}

#[test]
#[should_panic]
fn rc_join_unchecked_ratio() {
    let rc = Two::new(42);
    will_leak(&rc);

    let (one, two) = Two::split::<1, 1>(rc);

    unsafe { StaticRc::<_, 1, 2>::join_unchecked(one, two) };
}

#[test]
#[should_panic]
fn rc_join_array_ratio() {
    let rc = Two::new(42);
    will_leak(&rc);

    let array: [_; 2] = Two::split_array::<1, 2>(rc);

    StaticRc::<_, 1, 2>::join_array(array);
}

#[test]
#[should_panic]
fn rc_join_array_different() {
    let (rc, other) = (Two::new(42), Two::new(33));
    will_leak(&rc);
    will_leak(&other);

    let (one, two) = Two::split::<1, 1>(rc);
    let (other_one, other_two) = Two::split::<1, 1>(other);

    mem::forget([two, other_two]);

    Two::join_array([one, other_one]);
}

#[test]
#[should_panic]
fn rc_join_array_unchecked_ratio() {
    let rc = Two::new(42);
    will_leak(&rc);

    let array = Two::split_array::<1, 2>(rc);

    unsafe { StaticRc::<_, 1, 2>::join_array_unchecked(array) };
}

//  Indicates that the pointed to memory will be leaked, to avoid it being reported.
fn will_leak<T, const NUM: usize, const DEN: usize>(_rc: &StaticRc<T, NUM, DEN>) {
    #[cfg(miri)]
    {
        unsafe { miri_static_root(StaticRc::as_ptr(_rc).as_ptr() as *const u8) };
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

} // mod panic_ratio_tests
