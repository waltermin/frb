//! Two-phase search for optimal xor-shift constants at each bit width.
//!
//! Phase 1: Screen all candidate (s1, s2, s3) triples with a small sample
//!          count. Keep the top N finalists.
//! Phase 2: Re-evaluate finalists with a much larger sample count to find
//!          the true best.
//!
//! Parallelized with rayon across triples within each bit width.
//!
//! Add to Cargo.toml:
//!   [dependencies]
//!   rayon = "1"
//!
//! Run with: cargo run --release

use rayon::prelude::*;
use std::time::Instant;

// ── mixer under test ────────────────────────────────────────────────

const C0: u64 = 0xbf58476d1ce4e5b9;
const C1: u64 = 0x94d049bb133111eb;
const C2: u64 = 0xd6e8feb86659fd93;

#[inline(always)]
fn mix(mut x: u64, mask: u64, s1: u32, s2: u32, s3: u32) -> u64 {
    x = (x ^ (x >> s1)).wrapping_mul(C0) & mask;
    x = (x ^ (x >> s2)).wrapping_mul(C1) & mask;
    x = (x ^ (x >> s3)).wrapping_mul(C2) & mask;
    x ^ (x >> s3)
}

// ── avalanche scoring ───────────────────────────────────────────────

fn avalanche_score(bits: u32, s1: u32, s2: u32, s3: u32, samples: &[u64]) -> f64 {
    let mask = if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    };
    let nb = bits as usize;
    let n = samples.len();

    let mut counts = vec![0u32; nb * nb];

    for &x in samples {
        let base = mix(x, mask, s1, s2, s3);
        for ib in 0..nb {
            let diff = base ^ mix(x ^ (1u64 << ib), mask, s1, s2, s3);
            let mut d = diff;
            while d != 0 {
                let ob = d.trailing_zeros() as usize;
                counts[ib * nb + ob] += 1;
                d &= d - 1;
            }
        }
    }

    let ideal = n as f64 / 2.0;
    let mut total: f64 = 0.0;
    for &c in &counts {
        let dev = c as f64 - ideal;
        total += dev * dev;
    }

    total / (ideal * ideal * (nb * nb) as f64)
}

// ── sample generation ───────────────────────────────────────────────

fn make_samples(bits: u32, max_samples: usize) -> Vec<u64> {
    let domain_size: u128 = 1u128 << bits;

    if domain_size <= max_samples as u128 {
        (0..domain_size as u64).collect()
    } else {
        let mask = if bits >= 64 {
            u64::MAX
        } else {
            (1u64 << bits) - 1
        };
        let mut state = 0x123456789abcdef0u64;
        (0..max_samples)
            .map(|_| {
                state = state.wrapping_add(0x9e3779b97f4a7c15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
                z = z ^ (z >> 31);
                z & mask
            })
            .collect()
    }
}

// ── search ──────────────────────────────────────────────────────────

/// Build the list of all candidate (s1, s2, s3) triples for a given bit width.
fn candidate_triples(bits: u32) -> Vec<[u32; 3]> {
    let (lo, hi) = if bits <= 24 {
        (1u32, bits)
    } else if bits <= 40 {
        ((bits / 4).max(1), (bits * 3 / 4).min(bits))
    } else {
        ((bits / 3).max(1), (bits * 2 / 3).min(bits))
    };

    let mut triples = Vec::new();
    for s1 in lo..hi {
        for s2 in lo..hi {
            for s3 in lo..hi {
                triples.push([s1, s2, s3]);
            }
        }
    }
    triples
}

/// Phase 1: screen all triples with a small sample set (parallel).
/// Returns the top `keep` triples sorted by score.
fn screen(bits: u32, triples: &[[u32; 3]], samples: &[u64], keep: usize) -> Vec<([u32; 3], f64)> {
    let mut scored: Vec<([u32; 3], f64)> = triples
        .par_iter()
        .map(|&[s1, s2, s3]| {
            let score = avalanche_score(bits, s1, s2, s3, samples);
            ([s1, s2, s3], score)
        })
        .collect();

    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    scored.truncate(keep);
    scored
}

/// Phase 2: re-evaluate finalists with a large sample set (parallel).
/// Returns the single best triple.
fn refine(bits: u32, finalists: &[([u32; 3], f64)], samples: &[u64]) -> ([u32; 3], f64) {
    finalists
        .par_iter()
        .map(|&([s1, s2, s3], _)| {
            let score = avalanche_score(bits, s1, s2, s3, samples);
            ([s1, s2, s3], score)
        })
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap()
}

fn find_optimal(bits: u32) -> ([u32; 3], f64) {
    if bits <= 2 {
        return ([1, 1, 1], 0.0);
    }

    let triples = candidate_triples(bits);

    // For small domains where exhaustive evaluation is cheap,
    // skip the two-phase approach and just evaluate everything.
    if bits <= 16 {
        let samples = make_samples(bits, 1 << bits);
        let result = triples
            .par_iter()
            .map(|&[s1, s2, s3]| {
                let score = avalanche_score(bits, s1, s2, s3, &samples);
                ([s1, s2, s3], score)
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        return result;
    }

    // Phase 1: screen with small sample set
    let screen_samples = make_samples(bits, 2_000);
    let finalists = screen(bits, &triples, &screen_samples, 50);

    // Phase 2: refine with large sample set
    let refine_samples = make_samples(bits, 200_000);
    refine(bits, &finalists, &refine_samples)
}

// ── main ────────────────────────────────────────────────────────────

fn main() {
    eprintln!("Searching for optimal shift constants (1–64 bits)...");
    eprintln!("Using {} threads.\n", rayon::current_num_threads());
    eprintln!(
        "{:>4}  {:>3} {:>3} {:>3}   {:<14}  time",
        "bits", "s1", "s2", "s3", "score"
    );
    eprintln!("{}", "-".repeat(48));

    let total_start = Instant::now();
    let mut table = [[0u32; 3]; 64];

    for bits in 1..=64u32 {
        let start = Instant::now();
        let (shifts, score) = find_optimal(bits);
        let elapsed = start.elapsed();

        table[bits as usize - 1] = shifts;

        eprintln!(
            "{bits:>4}  {:>3} {:>3} {:>3}   {score:<14.8}  {:.1}s",
            shifts[0],
            shifts[1],
            shifts[2],
            elapsed.as_secs_f64()
        );
    }

    let total = total_start.elapsed();
    eprintln!("\nDone in {:.1}s.\n", total.as_secs_f64());

    // ── output the const table ──

    println!("/// Optimal xor-shift constants per bit width, found by exhaustive");
    println!("/// avalanche search (two-phase: screen + refine).");
    println!("/// Indexed by `bits - 1`.");
    println!("///");
    println!("/// Each entry is `[s1, s2, s3]` for the mix function:");
    println!("/// ```");
    println!("/// x = (x ^ (x >> s1)) * C0  &  mask;");
    println!("/// x = (x ^ (x >> s2)) * C1  &  mask;");
    println!("/// x = (x ^ (x >> s3)) * C2  &  mask;");
    println!("/// x ^ (x >> s3)");
    println!("/// ```");
    println!("const OPTIMAL_SHIFTS: [[u32; 3]; 64] = [");
    for (i, s) in table.iter().enumerate() {
        let bits = i + 1;
        println!("    [{:2}, {:2}, {:2}], // {bits:2} bits", s[0], s[1], s[2]);
    }
    println!("];");
}
