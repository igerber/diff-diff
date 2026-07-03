//! Feature-gated counting allocator for memory-footprint measurement builds.
//!
//! Compiled ONLY under `--features alloc-profile` (never in shipped wheels).
//! Wraps the system allocator with atomic current/high-water byte counters so
//! before/after footprint claims can be verified at the allocation level -
//! resident-set metrics are unreliable for this purpose on macOS, where the
//! memory compressor evicts cold pages from RSS (see docs/performance-plan.md).
//!
//! Scope: counts RUST-side allocations in this extension only; numpy/Python
//! allocations are invisible - which is exactly the surface under measurement.

use pyo3::prelude::*;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct CountingAlloc;

static CURRENT: AtomicUsize = AtomicUsize::new(0);
static HIGH_WATER: AtomicUsize = AtomicUsize::new(0);

fn track_alloc(size: usize) {
    let now = CURRENT.fetch_add(size, Ordering::Relaxed) + size;
    // Racy max update is fine for a diagnostic: retry while below.
    let mut hw = HIGH_WATER.load(Ordering::Relaxed);
    while now > hw {
        match HIGH_WATER.compare_exchange_weak(hw, now, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(actual) => hw = actual,
        }
    }
}

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            track_alloc(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        CURRENT.fetch_sub(layout.size(), Ordering::Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            CURRENT.fetch_sub(layout.size(), Ordering::Relaxed);
            track_alloc(new_size);
        }
        new_ptr
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

/// Reset the high-water mark to the current live-byte count.
#[pyfunction]
pub fn reset_alloc_high_water() {
    HIGH_WATER.store(CURRENT.load(Ordering::Relaxed), Ordering::Relaxed);
}

/// Peak rust-side allocated bytes since the last reset.
#[pyfunction]
pub fn alloc_high_water_bytes() -> usize {
    HIGH_WATER.load(Ordering::Relaxed)
}
