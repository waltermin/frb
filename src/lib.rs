//! # frb — fast random bijection
//!
//! A seedable, O(1)-memory pseudorandom bijection (permutation) over
//! `[0, size)` for arbitrary `size`. Given a seed, [`Bijection`] produces a
//! function that maps every input in the range to a unique output in the
//! same range, with no internal table and no precomputation proportional to
//! `size`.
//!
//! ## What it's for
//!
//! Anywhere you want the *effect* of shuffling without paying to materialize
//! the shuffled set:
//!
//! - Stream a shuffled view of `0..N` for huge `N` without allocating
//!   `O(N)` memory.
//! - Assign deterministic but unpredictable IDs from a dense counter.
//! - Sample without replacement from a large range by iterating
//!   `map(0), map(1), …` — every output is guaranteed distinct.
//! - Partition a stream of indices across workers while preserving a
//!   reproducible random order per seed.
//!
//! ## How it works
//!
//! Each instance composes two stages on the next power-of-two domain above
//! `size`:
//!
//! 1. A full-period **LCG** (Hull–Dobell parameters derived from the seed)
//!    gives a cheap bijection over `[0, 2^bits)`.
//! 2. A **splitmix-style xor-shift–multiply cascade** with shift constants
//!    precomputed per bit-width via exhaustive avalanche search provides
//!    the bit-mixing that makes the output look random.
//!
//! When `size` is not a power of two, **cycle walking** handles the gap: if
//! the permutation lands outside `[0, size)`, it is re-applied until it
//! does. Because the domain is at most 2× `size`, the expected number of
//! applications per call is under 2, and the chance of needing 10+ retries
//! is below 1-in-1000.
//!
//! ## Example
//!
//! ```
//! use frb::Bijection;
//!
//! let b = Bijection::new(0xC0FFEE, 1_000_000);
//!
//! // Walk 0..N and emit a shuffled permutation of the same range.
//! let shuffled: Vec<u64> = (0..10).map(|i| b.map(i)).collect();
//!
//! // Every output is distinct and lies in [0, 1_000_000).
//! let mut sorted = shuffled.clone();
//! sorted.sort();
//! sorted.dedup();
//! assert_eq!(sorted.len(), shuffled.len());
//! assert!(shuffled.iter().all(|&x| x < 1_000_000));
//! ```
//!
//! ## Properties
//!
//! - **O(1) memory.** A [`Bijection`] is a handful of `u64`s. No table is
//!   stored, regardless of `size`.
//! - **O(1) expected time per call.** Constant work in the power-of-two
//!   case; under 2× on average with cycle walking.
//! - **Deterministic.** The same `(seed, size)` always produces the same
//!   permutation.
//! - **Not cryptographic.** The mixer is tuned for speed and avalanche
//!   quality, not for resisting an adversary who can observe outputs and
//!   recover the seed. Do not use it where unpredictability matters for
//!   security.
//!
//! ## Limits
//!
//! `size` must satisfy `0 < size <= 2^63`. See [`Bijection::new`].

/// Optimal xor-shift constants per bit width, found by exhaustive
/// avalanche search (two-phase: screen + refine).
/// Indexed by `bits - 1`.
///
/// Each entry is `[s1, s2, s3]` for the mix function:
/// ```text
/// x = (x ^ (x >> s1)) * C0  &  mask;
/// x = (x ^ (x >> s2)) * C1  &  mask;
/// x = (x ^ (x >> s3)) * C2  &  mask;
/// x ^ (x >> s3)
/// ```
const OPTIMAL_SHIFTS: [[u8; 3]; 64] = [
    [1, 1, 1],    //  1 bits
    [1, 1, 1],    //  2 bits
    [1, 1, 2],    //  3 bits
    [1, 2, 2],    //  4 bits
    [1, 4, 3],    //  5 bits
    [1, 4, 2],    //  6 bits
    [3, 2, 3],    //  7 bits
    [4, 7, 3],    //  8 bits
    [6, 2, 3],    //  9 bits
    [7, 4, 4],    // 10 bits
    [5, 4, 5],    // 11 bits
    [7, 4, 5],    // 12 bits
    [7, 2, 8],    // 13 bits
    [7, 5, 6],    // 14 bits
    [9, 4, 6],    // 15 bits
    [8, 2, 8],    // 16 bits
    [8, 6, 8],    // 17 bits
    [9, 4, 9],    // 18 bits
    [11, 5, 8],   // 19 bits
    [10, 4, 9],   // 20 bits
    [11, 5, 9],   // 21 bits
    [11, 1, 10],  // 22 bits
    [9, 5, 10],   // 23 bits
    [17, 4, 10],  // 24 bits
    [16, 6, 11],  // 25 bits
    [14, 14, 8],  // 26 bits
    [7, 8, 11],   // 27 bits
    [16, 8, 10],  // 28 bits
    [11, 8, 11],  // 29 bits
    [13, 8, 9],   // 30 bits
    [15, 13, 11], // 31 bits
    [13, 18, 12], // 32 bits
    [8, 17, 16],  // 33 bits
    [24, 13, 11], // 34 bits
    [12, 13, 22], // 35 bits
    [16, 15, 17], // 36 bits
    [20, 11, 18], // 37 bits
    [20, 22, 25], // 38 bits
    [15, 13, 12], // 39 bits
    [26, 20, 19], // 40 bits
    [26, 21, 25], // 41 bits
    [18, 17, 15], // 42 bits
    [20, 27, 19], // 43 bits
    [14, 28, 14], // 44 bits
    [28, 18, 22], // 45 bits
    [15, 22, 22], // 46 bits
    [19, 29, 30], // 47 bits
    [21, 18, 25], // 48 bits
    [26, 21, 20], // 49 bits
    [26, 21, 20], // 50 bits
    [32, 22, 24], // 51 bits
    [20, 21, 20], // 52 bits
    [31, 23, 30], // 53 bits
    [24, 25, 25], // 54 bits
    [25, 31, 32], // 55 bits
    [24, 30, 28], // 56 bits
    [31, 29, 28], // 57 bits
    [31, 33, 24], // 58 bits
    [26, 37, 33], // 59 bits
    [23, 33, 34], // 60 bits
    [39, 20, 37], // 61 bits
    [21, 23, 37], // 62 bits
    [36, 35, 39], // 63 bits
    [35, 28, 39], // 64 bits
];

/// A seedable, O(1)-memory pseudorandom bijection over `[0, size)`.
///
/// Internally composes an LCG (full-cycle guarantee) with a splitmix-style
/// xor-shift-multiply cascade (avalanche), using cycle walking to handle
/// domain sizes that aren't powers of two.
///
/// Shift constants for the xor-shift-multiply cascade are precomputed per
/// bit width via exhaustive avalanche optimization (see `OPTIMAL_SHIFTS`).
///
/// Generating each element requires one or more applications of the
/// underlying permutation. The permutation operates on the next power-of-two
/// domain, which is at most 2× the actual size, so at least half of all
/// outputs are valid. On average this means fewer than 2 applications per
/// element. Needing even 10 retries has odds below 1-in-1000.
#[derive(Debug, Clone, Copy)]
pub struct Bijection {
    size: u64,
    mask: u64,
    shifts: [u32; 3],
    a: u64,
    c: u64,
}

impl Bijection {
    /// Create a new bijection over `[0, size)` seeded by `seed`.
    ///
    /// # Panics
    /// Panics if `size` is 0 or greater than `2^63`.
    pub fn new(seed: u64, size: u64) -> Self {
        assert!(size > 0, "size must be > 0");
        assert!(size <= (1u64 << 63), "size must be <= 2^63");

        let bits = (64 - (size - 1).leading_zeros()).max(1); // ceil(log2(size)), min 1
        let mask = if bits >= 64 {
            u64::MAX
        } else {
            (1u64 << bits) - 1
        };

        // Derive LCG constants from seed.
        // Hull-Dobell: a ≡ 1 (mod 4), c is odd → full period mod 2^bits.
        let ha = Self::mix64(seed);
        let hc = Self::mix64(seed ^ 0x9e3779b97f4a7c15);

        let a = ((ha & !3) | 1) & mask;
        let c = (hc | 1) & mask;

        let shifts = OPTIMAL_SHIFTS[bits as usize - 1].map(|x| x as u32);

        Self {
            size,
            mask,
            shifts,
            a,
            c,
        }
    }

    /// Map `x` in `[0, size)` to a unique output in `[0, size)`.
    ///
    /// Every input maps to a distinct output — this is a permutation.
    ///
    /// # Panics
    /// Panics if `x >= size`.
    pub fn map(&self, x: u64) -> u64 {
        assert!(x < self.size, "x must be < size");
        self.map_inner(x)
    }

    /// Map `x` without bounds checking.
    ///
    /// # Safety
    /// Caller must ensure `x < size`. Passing `x >= size` is not
    /// memory-unsafe, but the output is meaningless — it may fall
    /// outside `[0, size)` and the bijection guarantee is void.
    #[inline]
    pub unsafe fn map_unchecked(&self, x: u64) -> u64 {
        self.map_inner(x)
    }

    /// Map `x`, returning `None` if `x >= size`.
    #[inline]
    pub fn try_map(&self, x: u64) -> Option<u64> {
        if x < self.size {
            Some(self.map_inner(x))
        } else {
            None
        }
    }

    #[inline]
    fn map_inner(&self, mut val: u64) -> u64 {
        loop {
            val = self.permute(val);
            if val < self.size {
                return val;
            }
        }
    }

    /// The core permutation on `[0, 2^bits)`: LCG then splitmix-style mix.
    #[inline]
    fn permute(&self, x: u64) -> u64 {
        let lcg = (self.a.wrapping_mul(x).wrapping_add(self.c)) & self.mask;
        self.mix(lcg)
    }

    /// Splitmix-style xor-shift-multiply cascade with per-width optimized
    /// shift constants.
    ///
    /// Three rounds instead of splitmix64's two — the extra round is needed
    /// because our inputs are a structured LCG arithmetic progression rather
    /// than the already-decorrelated Weyl sequence splitmix64 was designed for.
    /// Shift constants are precomputed by exhaustive avalanche search (see
    /// `OPTIMAL_SHIFTS`).
    #[inline]
    fn mix(&self, mut x: u64) -> u64 {
        let [s1, s2, s3] = self.shifts;
        x = (x ^ (x >> s1)).wrapping_mul(0xbf58476d1ce4e5b9) & self.mask;
        x = (x ^ (x >> s2)).wrapping_mul(0x94d049bb133111eb) & self.mask;
        x = (x ^ (x >> s3)).wrapping_mul(0xd6e8feb86659fd93) & self.mask;
        x ^ (x >> s3)
    }

    /// Splitmix64 finalizer, used to derive LCG constants.
    fn mix64(mut x: u64) -> u64 {
        x ^= x >> 30;
        x = x.wrapping_mul(0xbf58476d1ce4e5b9);
        x ^= x >> 27;
        x = x.wrapping_mul(0x94d049bb133111eb);
        x ^= x >> 31;
        x
    }
}

#[test]
fn foo() {
    for seed in 10000..11000 {
        let len = 4;
        let bijection = Bijection::new(seed, len);

        for i in 0..len {
            println!(
                "seed={} | {}, {}, {}, {}",
                seed,
                bijection.map(0),
                bijection.map(1),
                bijection.map(2),
                bijection.map(3),
            );
        }
    }
}

#[cfg(test)]
mod tests;
#[cfg(test)]
mod uniformity_tests;
