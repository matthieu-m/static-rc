//! `StaticRcRef` is a compile-time referenced counted access to a mutable reference.

use core::{
    any,
    borrow,
    cmp,
    convert,
    fmt,
    future,
    hash,
    iter,
    marker::{self, PhantomData},
    mem::{self, MaybeUninit},
    ops,
    pin,
    ptr::{self, NonNull},
    task,
};

#[cfg(feature = "nightly-async-stream")]
use core::stream;

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
pub struct StaticRcRef<'a, T: ?Sized, const NUM: usize, const DEN: usize> {
    pointer: NonNull<T>,
    _marker: PhantomData<&'a mut T>,
}

impl<'a, T: ?Sized, const N: usize> StaticRcRef<'a, T, N, N> {
    /// Constructs a new `StaticRcRef<'a, T, N, N>`.
    ///
    /// #   Example
    ///
    /// ```rust
    /// use static_rc::StaticRcRef;
    ///
    /// type Full<'a> = StaticRcRef<'a, i32, 1, 1>;
    ///
    /// let mut value = 42;
    /// let rc = Full::new(&mut value);
    /// assert_eq!(42, *rc);
    /// ```
    #[inline(always)]
    pub fn new(value: &'a mut T) -> Self
    where
        AssertLeType!(1, N): Sized,
    {
        #[cfg(not(feature = "compile-time-ratio"))]
        assert!(N > 0);

        let pointer = NonNull::from(value);
        Self { pointer, _marker: PhantomData }
    }

    /// Constructs a new `Pin<StaticRcRef<'a, T, N, N>>`.
    ///
    /// #   Example
    ///
    /// ```rust
    /// use static_rc::StaticRcRef;
    ///
    /// type Full<'a> = StaticRcRef<'a, i32, 1, 1>;
    ///
    /// let mut value = 42;
    /// let rc = Full::pin(&mut value);
    /// assert_eq!(42, *rc);
    /// ```
    #[inline(always)]
    pub fn pin(value: &'a mut T) -> pin::Pin<Self>
    where
        AssertLeType!(1, N): Sized,
    {
        #[cfg(not(feature = "compile-time-ratio"))]
        assert!(N > 0);

        //  Safety:
        //  -   The `value` is mutably borrowed, and cannot be moved out of without full ownership.
        unsafe { pin::Pin::new_unchecked(Self::new(value)) }
    }

    /// Returns the inner value.
    ///
    /// #   Example
    ///
    /// ```rust
    /// use static_rc::StaticRcRef;
    ///
    /// type Full<'a> = StaticRcRef<'a, i32, 1, 1>;
    ///
    /// let mut value = 42;
    /// let rc = Full::new(&mut value);
    /// let inner: &mut i32 = Full::into_inner(rc);
    /// assert_eq!(42, *inner);
    /// ```
    #[inline(always)]
    pub fn into_inner(this: Self) -> &'a mut T {
        //  Safety:
        //  -   Ratio = 1, hence full ownership.
        //  -   Original lifetime is restored.
        unsafe { &mut *this.pointer.as_ptr() }
    }

    /// Returns a mutable reference into the given `StaticRcRef`.
    ///
    /// #   Example
    ///
    /// ```rust
    /// use static_rc::StaticRcRef;
    ///
    /// type Full<'a> = StaticRcRef<'a, i32, 1, 1>;
    ///
    /// let mut value = 42;
    /// let mut rc = Full::new(&mut value);
    /// assert_eq!(42, *Full::get_mut(&mut rc));
    /// ```
    #[inline(always)]
    pub fn get_mut(this: &mut Self) -> &mut T {
        //  Safety:
        //  -   Ratio = 1, hence full ownership.
        unsafe { this.pointer.as_mut() }
    }
}

impl<'a, T: ?Sized, const NUM: usize, const DEN: usize> StaticRcRef<'a, T, NUM, DEN> {
    /// Consumes the `StaticRcRef`, returning the wrapped pointer.
    ///
    /// #   Example
    ///
    /// ```rust
    /// use static_rc::StaticRcRef;
    ///
    /// type Full<'a> = StaticRcRef<'a, i32, 1, 1>;
    ///
    /// let mut value = 42;
    /// let rc = Full::new(&mut value);
    /// let pointer = Full::into_raw(rc);
    /// assert_eq!(&mut value as *mut _, pointer.as_ptr());
    /// ```
    #[inline(always)]
    pub fn into_raw(this: Self) -> NonNull<T> { this.pointer }

    /// Provides a raw pointer to the data.
    ///
    /// `StaticRcRef` is not consumed or affected in any way, the pointer is valid as long as the original value is.
    ///
    /// #   Example
    ///
    /// ```rust
    /// use static_rc::StaticRcRef;
    ///
    /// type Full<'a> = StaticRcRef<'a, i32, 1, 1>;
    ///
    /// let mut value = 42;
    /// let pointer = &mut value as *mut _;
    ///
    /// let rc = Full::new(&mut value);
    /// let other_pointer = Full::as_ptr(&rc);
    ///
    /// assert_eq!(pointer, other_pointer.as_ptr());
    /// ```
    #[inline(always)]
    pub fn as_ptr(this: &Self) -> NonNull<T> { this.pointer }

    /// Provides a reference to the data.
    ///
    /// #   Example
    ///
    /// ```rust
    /// use static_rc::StaticRcRef;
    ///
    /// type Full<'a> = StaticRcRef<'a, i32, 1, 1>;
    ///
    /// let mut value = 42;
    /// let rc = Full::new(&mut value);
    ///
    /// let r = Full::get_ref(&rc);
    ///
    /// assert_eq!(42, *r);
    /// ```
    #[inline(always)]
    pub fn get_ref(this: &Self) -> &T {
        //  Safety:
        //  -   The data is valid for as long as `this` lives.
        unsafe { this.pointer.as_ref() }
    }

    /// Constructs a `StaticRcRef<'a, T, NUM, DEN>` from a raw pointer.
    ///
    /// The raw pointer must have been previously returned by a call to `StaticRcRef<'a, U, N, D>::into_raw`:
    ///
    /// -   If `U` is different from `T`, then specific restrictions on size and alignment apply. See `mem::transmute`
    ///     for the restrictions applying to transmuting references.
    /// -   If `N / D` is different from `NUM / DEN`, then specific restrictions apply. The user is responsible for
    ///     ensuring proper management of the ratio of shares, and ultimately that the value is not dropped twice.
    //  Internal comment: Internally, calling `from_raw` in the specific case of `StaticRc::as_rcref`
    //  is allowed. This isn't allowed as an external user of the library.
    #[inline(always)]
    pub unsafe fn from_raw(pointer: NonNull<T>) -> Self
    where
        AssertLeType!(1, NUM): Sized,
    {
        #[cfg(not(feature = "compile-time-ratio"))]
        assert!(NUM > 0);

        Self { pointer, _marker: PhantomData, }
    }

    /// Returns true if the two `StaticRcRef` point to the same allocation.
    ///
    /// #   Example
    ///
    /// ```rust
    /// use static_rc::StaticRcRef;
    ///
    /// type Full<'a> = StaticRcRef<'a, i32, 2, 2>;
    ///
    /// let mut value = 42;
    /// let rc = Full::new(&mut value);
    /// let (one, two) = Full::split::<1, 1>(rc);
    ///
    /// assert!(StaticRcRef::ptr_eq(&one, &two));
    /// ```
    #[inline(always)]
    pub fn ptr_eq<const N: usize, const D: usize>(this: &Self, other: &StaticRcRef<'a, T, N, D>) -> bool {
        StaticRcRef::as_ptr(this) == StaticRcRef::as_ptr(other)
    }

    /// Adjusts the NUMerator and DENumerator of the ratio of the instance, preserving the ratio.
    ///
    /// #   Panics
    ///
    /// If the compile-time-ratio feature is not used, and the ratio is not preserved; that is `N / D <> NUM / DEN`.
    ///
    /// #   Example
    ///
    /// ```rust
    /// use static_rc::StaticRcRef;
    ///
    /// type Full<'a> = StaticRcRef<'a, i32, 2, 2>;
    ///
    /// let mut value = 42;
    /// let rc = Full::new(&mut value);
    /// let rc = Full::adjust::<1, 1>(rc);
    ///
    /// assert_eq!(42, *rc);
    /// ```
    #[inline(always)]
    pub fn adjust<const N: usize, const D: usize>(this: Self) -> StaticRcRef<'a, T, N, D>
    where
        AssertLeType!(1, N): Sized,
        AssertEqType!(N * DEN, NUM * D): Sized,
    {
        #[cfg(not(feature = "compile-time-ratio"))]
        {
            assert!(N > 0);
            assert_eq!(NUM * D, N * DEN, "{} / {} != {} / {}", NUM, DEN, N, D);
        }

        StaticRcRef { pointer: this.pointer, _marker: PhantomData }
    }

    /// Reborrows into another [`StaticRcRef`].
    /// 
    /// The current instance is mutably borrowed for the duration the result can be used.
    /// 
    /// #   Example
    /// 
    /// ```rust
    /// use static_rc::StaticRcRef;
    /// let mut x = 5;
    /// let rc_full: StaticRcRef<i32, 2, 2> = StaticRcRef::new(&mut x);
    /// let (mut rc1, mut rc2) = StaticRcRef::split::<1, 1>(rc_full);
    /// {
    ///     // Modify without moving `rc1`, `rc2`.
    ///     let rc_borrow1 = StaticRcRef::reborrow(&mut rc1);
    ///     let rc_borrow2 = StaticRcRef::reborrow(&mut rc2);
    ///     let mut rcref_full: StaticRcRef<_, 2, 2> = StaticRcRef::join(rc_borrow1, rc_borrow2);
    ///     *rcref_full = 9;
    ///     // Reborrow ends, can use the original refs again
    /// }
    /// let rc_full: StaticRcRef<_, 2, 2> = StaticRcRef::join(rc1, rc2);
    /// assert_eq!(*rc_full, 9);
    /// assert_eq!(x, 9);
    /// ```
    #[inline(always)]
    pub fn reborrow<'reborrow>(this: &'reborrow mut Self) -> StaticRcRef<'reborrow, T, NUM, DEN> {
        //  Safety (even though this doesn't use the `unsafe` keyword):
        //  -  `this.pointer` is a valid aligned pointer into a valid value of `T`.
        //  -   The result is only usable for lifetime `'a`, and for the duration
        //      of the lifetime `'a` `this` is mutably borrowed.
        //  -   `this` has NUM/DEN of the right to mutate the value. So it can lend NUM/DEN
        //      of the right to mutate the value. Therefore, this is semantically sound
        //      according to the general principle of this library.
        StaticRcRef {
            pointer: this.pointer,
            _marker: PhantomData::default(),
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
    /// use static_rc::StaticRcRef;
    ///
    /// type Full<'a> = StaticRcRef<'a, i32, 2, 2>;
    ///
    /// let mut value = 42;
    /// let rc = Full::new(&mut value);
    /// let (one, two) = Full::split::<1, 1>(rc);
    ///
    /// assert_eq!(42, *one);
    /// assert_eq!(42, *two);
    /// ```
    #[inline(always)]
    pub fn split<const A: usize, const B: usize>(this: Self) -> (StaticRcRef<'a, T, A, DEN>, StaticRcRef<'a, T, B, DEN>)
    where
        AssertLeType!(1, A): Sized,
        AssertLeType!(1, B): Sized,
        AssertEqType!(A + B, NUM): Sized,
    {
        #[cfg(not(feature = "compile-time-ratio"))]
        {
            assert!(A > 0);
            assert!(B > 0);
            assert_eq!(NUM, A + B, "{} != {} + {}", NUM, A, B);
        }

        let pointer = this.pointer;
        let _marker = PhantomData;

        (StaticRcRef { pointer, _marker, }, StaticRcRef { pointer, _marker, })
    }

    /// Splits the current instance into DIM instances with the specified Numerators.
    ///
    /// #   Panics
    ///
    /// If the compile-time-ratio feature is not used, and the ratio is not preserved; that is `N * DIM <> NUM`.
    ///
    /// #   Example
    ///
    /// ```rust
    /// use static_rc::StaticRcRef;
    ///
    /// type Full<'a> = StaticRcRef<'a, i32, 2, 2>;
    ///
    /// let mut value = 42;
    /// let rc = Full::new(&mut value);
    /// let array = Full::split_array::<1, 2>(rc);
    ///
    /// assert_eq!(42, *array[0]);
    /// assert_eq!(42, *array[1]);
    /// ```
    #[inline(always)]
    pub fn split_array<const N: usize, const DIM: usize>(this: Self) -> [StaticRcRef<'a, T, N, DEN>; DIM]
    where
        T: 'a,
        AssertEqType!(N * DIM, NUM): Sized,
        AssertLeType!(mem::size_of::<[StaticRcRef<'a, T, N, DEN>; DIM]>(), usize::MAX / 2 + 1): Sized,
    {
        #[cfg(not(feature = "compile-time-ratio"))]
        {
            assert_eq!(NUM, N * DIM, "{} != {} * {}", NUM, N, DIM);

            assert!(mem::size_of::<[StaticRcRef<T, N, DEN>; DIM]>() <= (isize::MAX as usize),
                "Size of result ({}) exceeeds isize::MAX", mem::size_of::<[StaticRcRef<T, N, DEN>; DIM]>());
        }

        let pointer = this.pointer;
        let _marker = PhantomData;

        let mut array = MaybeUninit::uninit();

        for i in 0..DIM {
            //  Safety:
            //  -   `destination` within bounds of allocated array (< DIM).
            //  -   Offset doesn't overflow `isize`, as per array-size assertion.
            //  -   Offset doesn't wrap around, as per array-size assertion.
            let destination = unsafe { (array.as_mut_ptr() as *mut StaticRcRef<T, N, DEN>).add(i) };

            //  Safety:
            //  -   `destination` is valid for writes.
            //  -   `destination` is correctly aligned.
            unsafe { ptr::write(destination, StaticRcRef { pointer, _marker, }); }
        }

        //  Safety:
        //  -   Every element of the array is now initialized.
        unsafe { array.assume_init() }
    }

    /// Joins two instances into a single instance.
    ///
    /// #   Panics
    ///
    /// If the two instances do no point to the same allocation, as determined by `StaticRcRef::ptr_eq`.
    ///
    /// If the compile-time-ratio feature is not used and the ratio is not preserved; that is `A + B <> NUM`.
    ///
    /// #   Example
    ///
    /// ```rust
    /// use static_rc::StaticRcRef;
    ///
    /// type Full<'a> = StaticRcRef<'a, i32, 2, 2>;
    ///
    /// let mut value = 42;
    /// let rc = Full::new(&mut value);
    /// let (one, two) = Full::split::<1, 1>(rc);
    /// let rc = Full::join(one, two);
    ///
    /// assert_eq!(42, *rc);
    /// ```
    #[inline(always)]
    pub fn join<const A: usize, const B: usize>(left: StaticRcRef<'a, T, A, DEN>, right: StaticRcRef<'a, T, B, DEN>) -> Self
    //  FIXME: re-enable when https://github.com/rust-lang/rust/issues/77708 fixed
    //  where
    //      AssertEqType!(NUM, A + B): Sized,
    {
        assert!(StaticRcRef::ptr_eq(&left, &right), "{:?} != {:?}", left.pointer.as_ptr(), right.pointer.as_ptr());

        //  Safety:
        //  -   `left` and `right` point to the same pointer.
        unsafe { Self::join_unchecked(left, right) }
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
    /// In debug, if the two instances do no point to the same allocation, as determined by `StaticRcRef::ptr_eq`.
    ///
    /// #   Example
    ///
    /// ```rust
    /// use static_rc::StaticRcRef;
    ///
    /// type Full<'a> = StaticRcRef<'a, i32, 2, 2>;
    ///
    /// let mut value = 42;
    /// let rc = Full::new(&mut value);
    /// let (one, two) = Full::split::<1, 1>(rc);
    /// let rc = unsafe { Full::join_unchecked(one, two) };
    ///
    /// assert_eq!(42, *rc);
    /// ```
    #[inline(always)]
    pub unsafe fn join_unchecked<const A: usize, const B: usize>(
        left: StaticRcRef<'a, T, A, DEN>,
        _right: StaticRcRef<'a, T, B, DEN>,
    ) -> Self
    //  FIXME: re-enable when https://github.com/rust-lang/rust/issues/77708 fixed
    //  where
    //      AssertEqType!(NUM, A + B): Sized,
    {
        //  FIXME: re-enable when https://github.com/rust-lang/rust/issues/77708 fixed
        //  #[cfg(not(feature = "compile-time-ratio"))]
        assert_eq!(NUM, A + B, "{} != {} + {}", NUM, A, B);

        debug_assert!(StaticRcRef::ptr_eq(&left, &_right), "{:?} != {:?}", left.pointer.as_ptr(), _right.pointer.as_ptr());

        Self { pointer: left.pointer, _marker: PhantomData, }
    }

    /// Joins DIM instances into a single instance.
    ///
    /// #   Panics
    ///
    /// If all instances do not point to the same allocation, as determined by `StaticRcRef::ptr_eq`.
    ///
    /// If the compile-time-ratio feature is not used and the ratio is not preserved; that is `N * DIM <> NUM`.
    ///
    /// #   Example
    ///
    /// ```rust
    /// use static_rc::StaticRcRef;
    ///
    /// type Full<'a> = StaticRcRef<'a, i32, 2, 2>;
    ///
    /// let mut value = 42;
    /// let rc = Full::new(&mut value);
    /// let array = Full::split_array::<1, 2>(rc);
    /// let rc = Full::join_array(array);
    ///
    /// assert_eq!(42, *rc);
    /// ```
    #[inline(always)]
    pub fn join_array<const N: usize, const DIM: usize>(array: [StaticRcRef<'a, T, N, DEN>; DIM]) -> Self
    //  FIXME: re-enable when https://github.com/rust-lang/rust/issues/77708 fixed
    //  where
    //      AssertLeType!(1, NUM): Sized,
    //      AssertEqType!(N * DIM, NUM): Sized,
    {
        let first = &array[0];
        for successive in &array[1..] {
            assert!(StaticRcRef::ptr_eq(&first, &successive),
                "{:?} != {:?}", first.pointer.as_ptr(), successive.pointer.as_ptr());
        }

        unsafe { Self::join_array_unchecked(array) }
    }

    /// Joins DIM instances into a single instance.
    ///
    /// #   Panics
    ///
    /// If the compile-time-ratio feature is not used and the ratio is not preserved; that is `N * DIM <> NUM`.
    ///
    /// In debug, if all instances do not point to the same allocation, as determined by `StaticRcRef::ptr_eq`.
    ///
    /// #   Example
    ///
    /// ```rust
    /// use static_rc::StaticRcRef;
    ///
    /// type Full<'a> = StaticRcRef<'a, i32, 2, 2>;
    ///
    /// let mut value = 42;
    /// let rc = Full::new(&mut value);
    /// let array = Full::split_array::<1, 2>(rc);
    /// let rc = unsafe { Full::join_array_unchecked(array) };
    ///
    /// assert_eq!(42, *rc);
    /// ```
    #[inline(always)]
    pub unsafe fn join_array_unchecked<const N: usize, const DIM: usize>(array: [StaticRcRef<'a, T, N, DEN>; DIM])
        -> Self
    //  FIXME: re-enable when https://github.com/rust-lang/rust/issues/77708 fixed
    //  where
    //      AssertLeType!(1, NUM): Sized,
    //      AssertEqType!(N * DIM, NUM): Sized,
    {
        //  FIXME: re-enable when https://github.com/rust-lang/rust/issues/77708 fixed
        //  #[cfg(not(feature = "compile-time-ratio"))]
        {
            assert!(NUM > 0);
            assert_eq!(NUM, N * DIM, "{} != {} * {}", NUM, N, DIM);
        }

        let _first = &array[0];
        for _successive in &array[1..] {
            debug_assert!(StaticRcRef::ptr_eq(&_first, &_successive),
                "{:?} != {:?}", _first.pointer.as_ptr(), _successive.pointer.as_ptr());
        }

        Self { pointer: array[0].pointer, _marker: PhantomData, }
    }

    
}

impl<'a, const NUM: usize, const DEN: usize> StaticRcRef<'a, dyn any::Any, NUM, DEN> {
    /// Attempts to downcast `Self` to a concrete type.
    pub fn downcast<T: any::Any>(self) -> Result<StaticRcRef<'a, T, NUM, DEN>, Self> {
        if Self::get_ref(&self).is::<T>() {
            let pointer = Self::into_raw(self).cast::<T>();
            Ok(StaticRcRef { pointer, _marker: PhantomData, })
        } else {
            Err(self)
        }
    }
}

impl<'a, T: ?Sized, const N: usize> convert::AsMut<T> for StaticRcRef<'a, T, N, N> {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut T { Self::get_mut(self) }
}

impl<'a, T: ?Sized, const NUM: usize, const DEN: usize> convert::AsRef<T> for StaticRcRef<'a, T, NUM, DEN> {
    #[inline(always)]
    fn as_ref(&self) -> &T { Self::get_ref(self) }
}

impl<'a, T: ?Sized, const NUM: usize, const DEN: usize> borrow::Borrow<T> for StaticRcRef<'a, T, NUM, DEN> {
    #[inline(always)]
    fn borrow(&self) -> &T { Self::get_ref(self) }
}

impl<'a, T: ?Sized, const N: usize> borrow::BorrowMut<T> for StaticRcRef<'a, T, N, N> {
    #[inline(always)]
    fn borrow_mut(&mut self) -> &mut T { Self::get_mut(self) }
}

#[cfg(feature = "nightly-coerce-unsized")]
impl<'a, T, U, const NUM: usize, const DEN: usize> CoerceUnsized<StaticRcRef<'a, U, NUM, DEN>> for StaticRcRef<'a, T, NUM, DEN>
where
    T: ?Sized + marker::Unsize<U>,
    U: ?Sized,
{}

impl<'a, T: ?Sized + fmt::Debug, const NUM: usize, const DEN: usize> fmt::Debug for StaticRcRef<'a, T, NUM, DEN> {
    #[inline(always)]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        fmt::Debug::fmt(Self::get_ref(self), f)
    }
}

impl<'a, T: ?Sized, const NUM: usize, const DEN: usize> ops::Deref for StaticRcRef<'a, T, NUM, DEN> {
    type Target = T;

    #[inline(always)]
    fn deref(&self) -> &T { Self::get_ref(self) }
}

impl<'a, T: ?Sized, const N: usize> ops::DerefMut for StaticRcRef<'a, T, N, N> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut T { Self::get_mut(self) }
}

impl<'a, T: ?Sized + fmt::Display, const NUM: usize, const DEN: usize> fmt::Display for StaticRcRef<'a, T, NUM, DEN> {
    #[inline(always)]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        fmt::Display::fmt(Self::get_ref(self), f)
    }
}

#[cfg(feature = "nightly-dispatch-from-dyn")]
impl<'a, T, U, const NUM: usize, const DEN: usize> DispatchFromDyn<StaticRcRef<'a, U, NUM, DEN>> for StaticRcRef<'a, T, NUM, DEN>
where
    T: ?Sized + marker::Unsize<U>,
    U: ?Sized,
{}

impl<'a, I: iter::DoubleEndedIterator + ?Sized, const N: usize> iter::DoubleEndedIterator for StaticRcRef<'a, I, N, N> {
    #[inline(always)]
    fn next_back(&mut self) -> Option<I::Item> { Self::get_mut(self).next_back() }

    #[inline(always)]
    fn nth_back(&mut self, n: usize) -> Option<I::Item> { Self::get_mut(self).nth_back(n) }
}

impl<'a, T: ?Sized + cmp::Eq, const NUM: usize, const DEN: usize> cmp::Eq for StaticRcRef<'a, T, NUM, DEN> {}

impl<'a, I: iter::ExactSizeIterator + ?Sized, const N: usize> iter::ExactSizeIterator for StaticRcRef<'a, I, N, N> {
    #[inline(always)]
    fn len(&self) -> usize { Self::get_ref(self).len() }
}

impl<'a, const NUM: usize, const DEN: usize> From<StaticRcRef<'a, str, NUM, DEN>> for StaticRcRef<'a, [u8], NUM, DEN> {
    #[inline(always)]
    fn from(value: StaticRcRef<'a, str, NUM, DEN>) -> Self {
        let pointer = value.pointer.as_ptr() as *mut [u8];

        //  Safety:
        //  -   `value.pointer` was not null, hence `pointer` is not null.
        debug_assert!(!pointer.is_null());
        let pointer = unsafe { NonNull::new_unchecked(pointer) };

        Self { pointer, _marker: PhantomData, }
    }
}

impl<'a, I: iter::FusedIterator + ?Sized, const N: usize> iter::FusedIterator for StaticRcRef<'a, I, N, N> {}

impl<'a, F: ?Sized + future::Future + marker::Unpin, const N: usize> future::Future for StaticRcRef<'a, F, N, N> {
    type Output = F::Output;

    fn poll(mut self: pin::Pin<&mut Self>, cx: &mut task::Context<'_>) -> task::Poll<Self::Output> {
        F::poll(pin::Pin::new(&mut *self), cx)
    }
}

#[cfg(feature = "nightly-generator-trait")]
impl<'a, G: ?Sized + ops::Generator<R> + marker::Unpin, R, const N: usize> ops::Generator<R> for StaticRcRef<'a, G, N, N> {
    type Yield = G::Yield;
    type Return = G::Return;

        fn resume(mut self: pin::Pin<&mut Self>, arg: R) -> ops::GeneratorState<Self::Yield, Self::Return> {
            G::resume(pin::Pin::new(&mut *self), arg)
        }
}

#[cfg(feature = "nightly-generator-trait")]
impl<'a, G: ?Sized + ops::Generator<R>, R, const N: usize> ops::Generator<R> for pin::Pin<StaticRcRef<'a, G, N, N>> {
    type Yield = G::Yield;
    type Return = G::Return;

        fn resume(mut self: pin::Pin<&mut Self>, arg: R) -> ops::GeneratorState<Self::Yield, Self::Return> {
            G::resume((*self).as_mut(), arg)
        }
}

impl<'a, T: ?Sized + hash::Hash, const NUM: usize, const DEN: usize> hash::Hash for StaticRcRef<'a, T, NUM, DEN> {
    #[inline(always)]
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        Self::get_ref(self).hash(state);
    }
}

impl<'a, I: iter::Iterator + ?Sized, const N: usize> iter::Iterator for StaticRcRef<'a, I, N, N> {
    type Item = I::Item;

    #[inline(always)]
    fn next(&mut self) -> Option<I::Item> { Self::get_mut(self).next() }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) { Self::get_ref(self).size_hint() }

    #[inline(always)]
    fn nth(&mut self, n: usize) -> Option<I::Item> { Self::get_mut(self).nth(n) }

    #[inline(always)]
    fn last(mut self) -> Option<I::Item> { Self::get_mut(&mut self).last() }
}

impl<'a, T: ?Sized + cmp::Ord, const NUM: usize, const DEN: usize> cmp::Ord for StaticRcRef<'a, T, NUM, DEN> {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        if Self::ptr_eq(self, other) {
            cmp::Ordering::Equal
        } else {
            Self::get_ref(self).cmp(Self::get_ref(other))
        }
    }
}

impl<'a, T, const NUM: usize, const DEN: usize, const N: usize, const D: usize> cmp::PartialEq<StaticRcRef<'a, T, N, D>>
    for StaticRcRef<'a, T, NUM, DEN>
where
    T: ?Sized + PartialEq<T>
{
    #[inline(always)]
    fn eq(&self, other: &StaticRcRef<'a, T, N, D>) -> bool { Self::get_ref(self).eq(StaticRcRef::get_ref(other)) }

    #[inline(always)]
    fn ne(&self, other: &StaticRcRef<'a, T, N, D>) -> bool { Self::get_ref(self).ne(StaticRcRef::get_ref(other)) }
}

impl<'a, T, const NUM: usize, const DEN: usize, const N: usize, const D: usize> cmp::PartialOrd<StaticRcRef<'a, T, N, D>>
    for StaticRcRef<'a, T, NUM, DEN>
where
    T: ?Sized + PartialOrd<T>
{
    #[inline(always)]
    fn partial_cmp(&self, other: &StaticRcRef<'a, T, N, D>) -> Option<cmp::Ordering> {
        Self::get_ref(self).partial_cmp(StaticRcRef::get_ref(other))
    }

    #[inline(always)]
    fn lt(&self, other: &StaticRcRef<'a, T, N, D>) -> bool {
        Self::get_ref(self).lt(StaticRcRef::get_ref(other))
    }

    #[inline(always)]
    fn le(&self, other: &StaticRcRef<'a, T, N, D>) -> bool {
        Self::get_ref(self).le(StaticRcRef::get_ref(other))
    }

    #[inline(always)]
    fn gt(&self, other: &StaticRcRef<'a, T, N, D>) -> bool {
        Self::get_ref(self).gt(StaticRcRef::get_ref(other))
    }

    #[inline(always)]
    fn ge(&self, other: &StaticRcRef<'a, T, N, D>) -> bool {
        Self::get_ref(self).ge(StaticRcRef::get_ref(other))
    }
}

impl<'a, T: ?Sized, const NUM: usize, const DEN: usize> fmt::Pointer for StaticRcRef<'a, T, NUM, DEN> {
    #[inline(always)]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Pointer::fmt(&Self::as_ptr(self).as_ptr(), f)
    }
}

#[cfg(feature = "nightly-async-stream")]
impl<'a, S: ?Sized + stream::Stream + marker::Unpin, const N: usize> stream::Stream for StaticRcRef<'a, S, N, N> {
    type Item = S::Item;

    fn poll_next(mut self: pin::Pin<&mut Self>, cx: &mut task::Context<'_>) -> task::Poll<Option<Self::Item>> {
        pin::Pin::new(&mut **self).poll_next(cx)
    }

    fn size_hint(&self) -> (usize, Option<usize>) { (**self).size_hint() }
}

impl<'a, T: ?Sized, const NUM: usize, const DEN: usize> marker::Unpin for StaticRcRef<'a, T, NUM, DEN> {}

unsafe impl<'a, T: ?Sized + marker::Send, const NUM: usize, const DEN: usize> marker::Send for StaticRcRef<'a, T, NUM, DEN> {}

unsafe impl<'a, T: ?Sized + marker::Sync, const NUM: usize, const DEN: usize> marker::Sync for StaticRcRef<'a, T, NUM, DEN> {}

#[doc(hidden)]
pub mod compile_tests {

/// ```compile_fail,E0597
/// let a = String::from("foo");
/// let mut a_ref = &a;
/// let mut rc = static_rc::StaticRcRef::<'_, _,1,1>::new(&mut a_ref);
/// {
///     let b = String::from("bar");
///     *rc = &b; // a_ref now points to b
/// }
/// // b is now dropped
/// assert_ne!(a_ref, "bar");  // This should fail to compile.
/// ```
pub fn rcref_prevent_use_after_free() {}

/// ```compile_fail,E0505
/// let mut a = String::from("foo");
/// let mut rc = static_rc::StaticRcRef::<'_, _,1,1>::new(&mut a);
/// 
/// let mut reborrow = static_rc::StaticRcRef::reborrow(&mut rc);
/// std::mem::drop(rc);
/// assert_eq!(*reborrow, "foo"); // This should fail to compile.
/// ```
pub fn rcref_reborrow_and_move() {}

/// ```compile_fail,E0502
/// let mut a = String::from("foo");
/// let mut rc = static_rc::StaticRcRef::<'_, _,1,1>::new(&mut a);
/// 
/// let mut reborrow = static_rc::StaticRcRef::reborrow(&mut rc);
/// assert_eq!(*rc, "foo");
/// assert_eq!(*reborrow, "foo"); // This should fail to compile.
/// ```
pub fn rcref_reborrow_and_use() {}

} // mod compile_tests

#[doc(hidden)]
#[cfg(feature = "compile-time-ratio")]
pub mod compile_ratio_tests {

/// ```compile_fail,E0080
/// type Zero<'a> = static_rc::StaticRcRef<'a, i32, 0, 0>;
///
/// let mut value = 42;
///
/// Zero::new(&mut value);
/// ```
pub fn rcref_new_zero() {}

/// ```compile_fail,E0080
/// type Zero<'a> = static_rc::StaticRcRef<'a, i32, 0, 0>;
///
/// let mut value = 42;
///
/// Zero::pin(&mut value);
/// ```
pub fn rcref_pin_zero() {}

/// ```compile_fail,E0080
/// type Zero<'a> = static_rc::StaticRcRef<'a, i32, 0, 0>;
///
/// let pointer = core::ptr::NonNull::dangling();
///
/// unsafe { Zero::from_raw(pointer) };
/// ```
pub fn rcref_from_raw_zero() {}

/// ```compile_fail,E0080
/// type One<'a> = static_rc::StaticRcRef<'a, i32, 1, 1>;
///
/// let mut value = 42;
/// let rc = One::new(&mut value);
///
/// One::adjust::<0, 0>(rc);
/// ```
pub fn rcref_adjust_zero() {}

/// ```compile_fail,E0080
/// type One<'a> = static_rc::StaticRcRef<'a, i32, 1, 1>;
///
/// let mut value = 42;
/// let rc = One::new(&mut value);
///
/// One::adjust::<2, 3>(rc);
/// ```
pub fn rcref_adjust_ratio() {}

/// ```compile_fail,E0080
/// type Two<'a> = static_rc::StaticRcRef<'a, i32, 2, 2>;
///
/// let mut value = 42;
/// let rc = Two::new(&mut value);
///
/// Two::split::<0, 2>(rc);
/// ```
pub fn rcref_split_zero_first() {}

/// ```compile_fail,E0080
/// type Two<'a> = static_rc::StaticRcRef<'a, i32, 2, 2>;
///
/// let mut value = 42;
/// let rc = Two::new(&mut value);
///
/// Two::split::<2, 0>(rc);
/// ```
pub fn rcref_split_zero_second() {}

/// ```compile_fail,E0080
/// type Two<'a> = static_rc::StaticRcRef<'a, i32, 2, 2>;
///
/// let mut value = 42;
/// let rc = Two::new(&mut value);
///
/// Two::split::<1, 2>(rc);
/// ```
pub fn rcref_split_sum() {}

/// ```compile_fail,E0080
/// type Two<'a> = static_rc::StaticRcRef<'a, i32, 2, 2>;
///
/// let mut value = 42;
/// let rc = Two::new(&mut value);
///
/// Two::split_array::<2, 2>(rc);
/// ```
pub fn rcref_split_array_ratio() {}

//  FIXME: should be "compile_fail,E0080"
/// ```should_panic
/// type Two<'a> = static_rc::StaticRcRef<'a, i32, 2, 2>;
///
/// let mut value = 42;
/// let rc = Two::new(&mut value);
/// let (one, two) = Two::split::<1, 1>(rc);
///
/// static_rc::StaticRcRef::<'_, _, 1, 2>::join(one, two);
/// ```
pub fn rcref_join_ratio() {}

//  FIXME: should be "compile_fail,E0080"
/// ```should_panic
/// type Two<'a> = static_rc::StaticRcRef<'a, i32, 2, 2>;
///
/// let mut value = 42;
/// let rc = Two::new(&mut value);
/// let (one, two) = Two::split::<1, 1>(rc);
///
/// unsafe { static_rc::StaticRcRef::<'_, _, 1, 2>::join_unchecked(one, two) };
/// ```
pub fn rcref_join_unchecked_ratio() {}

//  FIXME: should be "compile_fail,E0080"
/// ```should_panic
/// type Two<'a> = static_rc::StaticRcRef<'a, i32, 2, 2>;
///
/// let mut value = 42;
/// let rc = Two::new(&mut value);
/// let array: [_; 2] = Two::split_array::<1, 2>(rc);
///
/// static_rc::StaticRcRef::<'_, _, 1, 2>::join_array(array);
/// ```
pub fn rcref_join_array_ratio() {}

//  FIXME: should be "compile_fail,E0080"
/// ```should_panic
/// type Two<'a> = static_rc::StaticRcRef<'a, i32, 2, 2>;
///
/// let mut value = 42;
/// let rc = Two::new(&mut value);
/// let array: [_; 2] = Two::split_array::<1, 2>(rc);
///
/// unsafe { static_rc::StaticRcRef::<'_, _, 1, 2>::join_array_unchecked(array) };
/// ```
pub fn rcref_join_array_unchecked_ratio() {}

} // mod compile_ratio_tests

#[cfg(all(test, not(feature = "compile-time-ratio")))]
mod panic_ratio_tests {

use super::*;

type Zero<'a> = StaticRcRef<'a, i32, 0, 0>;
type One<'a> = StaticRcRef<'a, i32, 1, 1>;
type Two<'a> = StaticRcRef<'a, i32, 2, 2>;

#[test]
#[should_panic]
fn rcref_new_zero() {
    let mut value = 42;

    Zero::new(&mut value);
}

#[test]
#[should_panic]
fn rcref_pin_zero() {
    let mut value = 42;

    Zero::pin(&mut value);
}

#[test]
#[should_panic]
fn rcref_from_raw_zero() {
    let pointer = NonNull::dangling();

    unsafe { Zero::from_raw(pointer) };
}

#[test]
#[should_panic]
fn rcref_adjust_zero() {
    let mut value = 42;
    let rc = One::new(&mut value);

    One::adjust::<0, 0>(rc);
}

#[test]
#[should_panic]
fn rcref_adjust_ratio() {
    let mut value = 42;
    let rc = One::new(&mut value);

    One::adjust::<2, 3>(rc);
}

#[test]
#[should_panic]
fn rcref_split_zero_first() {
    let mut value = 42;
    let rc = Two::new(&mut value);

    Two::split::<0, 2>(rc);
}

#[test]
#[should_panic]
fn rcref_split_zero_second() {
    let mut value = 42;
    let rc = Two::new(&mut value);

    Two::split::<0, 2>(rc);
}

#[test]
#[should_panic]
fn rcref_split_sum() {
    let mut value = 42;
    let rc = Two::new(&mut value);

    Two::split::<1, 2>(rc);
}

#[test]
#[should_panic]
fn rcref_split_array_ratio() {
    let mut value = 42;
    let rc = Two::new(&mut value);

    Two::split_array::<2, 2>(rc);
}

#[test]
#[should_panic]
fn rcref_join_ratio() {
    let mut value = 42;
    let rc = Two::new(&mut value);
    let (one, two) = Two::split::<1, 1>(rc);

    StaticRcRef::<'_, _, 1, 2>::join(one, two);
}

#[test]
#[should_panic]
fn rcref_join_different() {
    let (mut value, mut other_value) = (42, 33);
    let (rc, other) = (Two::new(&mut value), Two::new(&mut other_value));
    let (one, _two) = Two::split::<1, 1>(rc);
    let (other_one, _other_two) = Two::split::<1, 1>(other);

    Two::join(one, other_one);
}

#[test]
#[should_panic]
fn rcref_join_unchecked_ratio() {
    let mut value = 42;
    let rc = Two::new(&mut value);
    let (one, two) = Two::split::<1, 1>(rc);

    unsafe { StaticRcRef::<'_, _, 1, 2>::join_unchecked(one, two) };
}

#[test]
#[should_panic]
fn rcref_join_array_ratio() {
    let mut value = 42;
    let rc = Two::new(&mut value);
    let array: [_; 2] = Two::split_array::<1, 2>(rc);

    StaticRcRef::<'_, _, 1, 2>::join_array(array);
}

#[test]
#[should_panic]
fn rcref_join_array_different() {
    let (mut value, mut other_value) = (42, 33);
    let (rc, other) = (Two::new(&mut value), Two::new(&mut other_value));
    let (one, _two) = Two::split::<1, 1>(rc);
    let (other_one, _other_two) = Two::split::<1, 1>(other);

    Two::join_array([one, other_one]);
}

#[test]
#[should_panic]
fn rcref_join_array_unchecked_ratio() {
    let mut value = 42;
    let rc = Two::new(&mut value);
    let array = Two::split_array::<1, 2>(rc);

    unsafe { StaticRcRef::<'_, _, 1, 2>::join_array_unchecked(array) };
}

} // mod panic_ratio_tests
