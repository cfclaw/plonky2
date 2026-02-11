#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
#![deny(rustdoc::broken_intra_doc_links)]
#![deny(missing_debug_implementations)]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
pub extern crate alloc;

/// Re-export of `plonky2_field`.
#[doc(inline)]
pub use plonky2_field as field;

pub mod batch_fri;
pub mod fri;
pub mod gadgets;
pub mod gates;
pub mod hash;
pub mod iop;
pub mod plonk;
pub mod recursion;
pub mod util;
pub mod serialization_helpers;

/// Minimal synchronous executor for futures that complete without yielding.
/// Used by `CircuitData::prove()` when `async_prover` is enabled to drive
/// the (trivially-async) CPU prover future to completion.
/// Panics if the future yields (i.e. returns `Pending`).
#[cfg(feature = "async_prover")]
pub fn block_on_simple<T>(future: impl core::future::Future<Output = T>) -> T {
    use core::pin::pin;
    use core::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

    fn noop_raw_waker() -> RawWaker {
        fn no_op(_: *const ()) {}
        fn clone(_: *const ()) -> RawWaker { noop_raw_waker() }
        RawWaker::new(core::ptr::null(), &RawWakerVTable::new(clone, no_op, no_op, no_op))
    }

    let waker = unsafe { Waker::from_raw(noop_raw_waker()) };
    let mut cx = Context::from_waker(&waker);
    let mut future = pin!(future);

    match future.as_mut().poll(&mut cx) {
        Poll::Ready(val) => val,
        Poll::Pending => panic!(
            "block_on_simple: future yielded Pending. \
             Use plonky2::plonk::prover::prove() directly with .await for async provers."
        ),
    }
}

#[cfg(test)]
mod lookup_test;
