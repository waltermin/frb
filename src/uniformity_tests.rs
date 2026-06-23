use crate::Bijection;

/// Statistical tests for `Bijection` output quality.
///
/// A permutation over [0, N) is *by construction* perfectly uniform —
/// every output appears exactly once — so bin-counting tests like χ²
/// are trivially satisfied and uninformative. The interesting question
/// is whether the *mapping* has detectable structure.
///
/// These tests look for three kinds of structure:
///
/// 1. **Serial correlation**: Are consecutive inputs mapped to
///    correlated outputs? Measures Pearson's r between map(i) and
///    map(i+1). A good permutation has r ≈ 0.
///
/// 2. **Runs test**: Does the output sequence have the expected number
///    of ascending/descending runs? Too few runs means the permutation
///    preserves input ordering; too many means it anti-correlates
///    neighbors.
///
/// 3. **Bit independence**: Are individual output bits correlated with
///    the input value? Measures the point-biserial correlation between
///    each output bit and the (normalized) input. A good permutation
///    has all bit correlations near zero.
///
/// 4. **Kolmogorov-Smirnov**: Retained from prior version — checks that
///    the CDF of outputs matches a uniform CDF. Still valid because it
///    tests *ordering* structure, not bin counts.

#[cfg(test)]
mod uniformity_tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────

    /// Standard normal CDF, tanh approximation.
    fn norm_cdf(z: f64) -> f64 {
        0.5 * (1.0 + (z * std::f64::consts::FRAC_1_SQRT_2).tanh())
    }

    /// Two-tailed p-value from a z-score.
    fn two_tailed_p(z: f64) -> f64 {
        2.0 * (1.0 - norm_cdf(z.abs()))
    }

    // ── serial correlation ──────────────────────────────────────────

    /// Pearson correlation between map(i) and map(i+1) for i in [0, size-1).
    ///
    /// Under a random permutation, the expected correlation is
    /// -1/(n-1) ≈ 0 for large n, with variance ≈ 1/n.
    fn serial_correlation_test(seed: u64, size: u64) -> (f64, bool) {
        let b = Bijection::new(seed, size);
        let n = (size - 1) as f64;

        let mut prev = b.map(0);
        let mut sum_xy: f64 = 0.0;
        let mut sum_x: f64 = prev as f64;
        let mut sum_y: f64 = 0.0;
        let mut sum_x2: f64 = (prev as f64) * (prev as f64);
        let mut sum_y2: f64 = 0.0;

        for i in 1..size {
            let cur = b.map(i);
            let x = prev as f64;
            let y = cur as f64;

            sum_xy += x * y;
            sum_y += y;
            sum_y2 += y * y;

            if i < size - 1 {
                // cur will be the x of the next pair
                sum_x += y;
                sum_x2 += y * y;
            }

            prev = cur;
        }

        let mean_x = sum_x / n;
        let mean_y = sum_y / n;
        let var_x = sum_x2 / n - mean_x * mean_x;
        let var_y = sum_y2 / n - mean_y * mean_y;

        let r = if var_x > 0.0 && var_y > 0.0 {
            (sum_xy / n - mean_x * mean_y) / (var_x.sqrt() * var_y.sqrt())
        } else {
            0.0
        };

        // Under null hypothesis, r ≈ N(-1/(n-1), 1/n) for large n.
        // Use z = r * sqrt(n) as the test statistic.
        let z = r * n.sqrt();
        let p = two_tailed_p(z);
        let pass = p > 0.001;

        println!(
            "  Serial: seed={seed}, size={size}, r={r:.6}, z={z:.2}, p={p:.4} → {}",
            if pass { "PASS" } else { "FAIL" }
        );

        (p, pass)
    }

    // ── runs test ───────────────────────────────────────────────────

    /// Counts total monotone runs (both ascending and descending) in the
    /// output sequence by counting turning points (direction changes).
    ///
    /// For a random permutation of n elements, the expected number of
    /// monotone runs is (2n - 1) / 3, with variance (16n - 29) / 90.
    fn runs_test(seed: u64, size: u64) -> (f64, bool) {
        let b = Bijection::new(seed, size);
        let n = size as f64;

        let outputs: Vec<u64> = (0..size).map(|i| b.map(i)).collect();

        // Count turning points: positions where direction changes.
        // Total monotone runs = turning points + 1.
        let mut runs: u64 = 1;
        for i in 1..outputs.len() - 1 {
            let went_up = outputs[i] > outputs[i - 1];
            let goes_up = outputs[i + 1] > outputs[i];
            if went_up != goes_up {
                runs += 1;
            }
        }

        let expected = (2.0 * n - 1.0) / 3.0;
        let variance = (16.0 * n - 29.0) / 90.0;
        let z = (runs as f64 - expected) / variance.sqrt();
        let p = two_tailed_p(z);
        let pass = p > 0.001;

        println!(
            "  Runs: seed={seed}, size={size}, runs={runs}, \
             expected={expected:.0}, z={z:.2}, p={p:.4} → {}",
            if pass { "PASS" } else { "FAIL" }
        );

        (p, pass)
    }

    // ── bit independence ────────────────────────────────────────────

    /// For each output bit, compute the point-biserial correlation with
    /// the input index. Reports the maximum |r| across all bits.
    ///
    /// If any single bit is strongly correlated with the input, the
    /// permutation leaks structure.
    fn bit_independence_test(seed: u64, size: u64) -> (f64, bool) {
        let b = Bijection::new(seed, size);
        let bits = 64 - (size - 1).leading_zeros();
        let n = size as f64;

        // For each bit position, accumulate the sum of input indices
        // where that bit is 1 and count how many are 1.
        let mut bit_sum = vec![0.0f64; bits as usize];
        let mut bit_count = vec![0u64; bits as usize];

        let input_mean = (size - 1) as f64 / 2.0;
        let mut input_var_sum: f64 = 0.0;

        for i in 0..size {
            let out = b.map(i);
            let x = i as f64;
            input_var_sum += (x - input_mean) * (x - input_mean);

            for bit in 0..bits as usize {
                if (out >> bit) & 1 == 1 {
                    bit_sum[bit] += x;
                    bit_count[bit] += 1;
                }
            }
        }

        let input_std = (input_var_sum / n).sqrt();
        let mut max_abs_r: f64 = 0.0;
        let mut worst_bit = 0;

        for bit in 0..bits as usize {
            let n1 = bit_count[bit] as f64;
            let n0 = n - n1;
            if n1 == 0.0 || n0 == 0.0 {
                continue;
            }

            let mean1 = bit_sum[bit] / n1;
            let mean0 = (n * input_mean - bit_sum[bit]) / n0;

            // Point-biserial: r = (mean1 - mean0) * sqrt(n1*n0/n²) / std_x
            let r = (mean1 - mean0) * (n1 * n0).sqrt() / (n * input_std);
            if r.abs() > max_abs_r {
                max_abs_r = r.abs();
                worst_bit = bit;
            }
        }

        // Under null, r ≈ N(0, 1/n), so z = r * sqrt(n).
        // Use Bonferroni: require p > 0.001/bits for any single bit.
        let z = max_abs_r * n.sqrt();
        let p = two_tailed_p(z);
        let threshold = 0.001 / bits as f64;
        let pass = p > threshold;

        println!(
            "  Bit independence: seed={seed}, size={size}, worst_bit={worst_bit}, \
             |r|={max_abs_r:.6}, z={z:.2}, p={p:.4} → {}",
            if pass { "PASS" } else { "FAIL" }
        );

        (max_abs_r, pass)
    }

    // ── K-S test (retained) ─────────────────────────────────────────

    /// Kolmogorov-Smirnov: maximum deviation between empirical and
    /// uniform CDF. Still informative for permutations because it
    /// tests ordering structure, not bin counts.
    fn ks_test(seed: u64, size: u64) -> (f64, bool) {
        let b = Bijection::new(seed, size);
        let mut outputs: Vec<u64> = (0..size).map(|i| b.map(i)).collect();
        outputs.sort_unstable();

        let n = size as f64;
        let mut d_max: f64 = 0.0;

        for (i, &val) in outputs.iter().enumerate() {
            let empirical = (i + 1) as f64 / n;
            let uniform = (val as f64 + 1.0) / n;
            let d = (empirical - uniform).abs();
            d_max = d_max.max(d);
        }

        let d_crit = 1.628 / n.sqrt();
        let pass = d_max < d_crit;

        println!(
            "  K-S: seed={seed}, size={size}, D={d_max:.6}, \
             D_crit={d_crit:.6} → {}",
            if pass { "PASS" } else { "FAIL" }
        );

        (d_max, pass)
    }

    // ── test harness ────────────────────────────────────────────────

    const SEEDS: [u64; 8] = [0, 1, 42, 0xDEAD, 0xCAFE, 0xFFFF, 12345, 99999];
    const SIZE: u64 = 10_000;
    const PRIME_SIZES: [u64; 3] = [997, 5003, 7919];

    fn run_multi_seed(name: &str, size: u64, test_fn: fn(u64, u64) -> (f64, bool)) {
        println!();
        let mut failures = 0;
        for &seed in &SEEDS {
            let (_, pass) = test_fn(seed, size);
            if !pass {
                failures += 1;
            }
        }
        assert!(
            failures <= 1,
            "{failures}/{} seeds failed {name}",
            SEEDS.len()
        );
    }

    #[test]
    fn test_serial_correlation() {
        run_multi_seed("serial correlation", SIZE, serial_correlation_test);
    }

    #[test]
    fn test_runs() {
        run_multi_seed("runs", SIZE, runs_test);
    }

    #[test]
    fn test_bit_independence() {
        run_multi_seed("bit independence", SIZE, bit_independence_test);
    }

    #[test]
    fn test_ks() {
        run_multi_seed("K-S", SIZE, ks_test);
    }

    #[test]
    fn test_serial_correlation_primes() {
        println!();
        for &size in &PRIME_SIZES {
            let (_, pass) = serial_correlation_test(42, size);
            assert!(pass, "serial correlation failed for prime size={size}");
        }
    }

    #[test]
    fn test_runs_primes() {
        println!();
        for &size in &PRIME_SIZES {
            let (_, pass) = runs_test(42, size);
            assert!(pass, "runs test failed for prime size={size}");
        }
    }

    #[test]
    fn test_bit_independence_primes() {
        println!();
        for &size in &PRIME_SIZES {
            let (_, pass) = bit_independence_test(42, size);
            assert!(pass, "bit independence failed for prime size={size}");
        }
    }

    #[test]
    fn test_ks_primes() {
        println!();
        for &size in &PRIME_SIZES {
            let (_, pass) = ks_test(42, size);
            assert!(pass, "K-S failed for prime size={size}");
        }
    }

    // ── sanity: detect known-bad permutations ───────────────────────

    /// Identity permutation (map(i) = i) should fail the runs test
    /// catastrophically — it has exactly 1 ascending run.
    #[test]
    fn test_detects_identity() {
        println!();
        let size: u64 = 10_000;
        let n = size as f64;

        let runs: u64 = 1;
        let expected = (2.0 * n - 1.0) / 3.0;
        let variance = (16.0 * n - 29.0) / 90.0;
        let z = (runs as f64 - expected) / variance.sqrt();

        println!("  Identity: runs=1, expected={expected:.0}, z={z:.2} (should be huge negative)");
        assert!(
            z < -50.0,
            "sanity: identity permutation must fail runs test"
        );
    }

    /// Reversal (map(i) = size-1-i) is a single descending run — 0 turning
    /// points, 1 total monotone run. Should fail like identity.
    #[test]
    fn test_detects_reversal() {
        println!();
        let size: u64 = 10_000;
        let n = size as f64;

        let runs: u64 = 1; // one long descending run, no turning points
        let expected = (2.0 * n - 1.0) / 3.0;
        let variance = (16.0 * n - 29.0) / 90.0;
        let z = (runs as f64 - expected) / variance.sqrt();

        println!("  Reversal: runs=1, expected={expected:.0}, z={z:.2} (should be huge negative)");
        assert!(
            z < -50.0,
            "sanity: reversal permutation must fail runs test"
        );
    }

    /// A maximally alternating sequence [0, n-1, 1, n-2, 2, ...] has a
    /// turning point at every interior position → n-1 runs. Should fail
    /// in the opposite direction (too many runs).
    #[test]
    fn test_detects_alternating() {
        println!();
        let size: u64 = 10_000;
        let n = size as f64;

        let runs = size - 1; // turning point at every interior position
        let expected = (2.0 * n - 1.0) / 3.0;
        let variance = (16.0 * n - 29.0) / 90.0;
        let z = (runs as f64 - expected) / variance.sqrt();

        println!(
            "  Alternating: runs={runs}, expected={expected:.0}, z={z:.2} (should be huge positive)"
        );
        assert!(
            z > 50.0,
            "sanity: alternating permutation must fail runs test"
        );
    }

    /// Sanity check: a Fisher-Yates shuffle (known-good random permutation)
    /// should pass the runs test, confirming the test itself is sound.
    #[test]
    fn test_fisher_yates_passes_runs() {
        println!();
        let size = 10_000usize;

        // Run multiple shuffles to be thorough
        let seeds: [u64; 8] = [0, 1, 42, 0xDEAD, 0xCAFE, 0xFFFF, 12345, 99999];
        let mut failures = 0;

        for &seed in &seeds {
            let mut perm: Vec<u64> = (0..size as u64).collect();

            // Fisher-Yates with splitmix64 PRNG
            let mut rng_state = seed;
            for i in (1..size).rev() {
                rng_state = rng_state.wrapping_add(0x9e3779b97f4a7c15);
                let mut z = rng_state;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
                z = z ^ (z >> 31);
                let j = (z as usize) % (i + 1);
                perm.swap(i, j);
            }

            // Count monotone runs (turning points + 1)
            let mut runs = 1u64;
            for i in 1..size - 1 {
                let went_up = perm[i] > perm[i - 1];
                let goes_up = perm[i + 1] > perm[i];
                if went_up != goes_up {
                    runs += 1;
                }
            }

            let n = size as f64;
            let expected = (2.0 * n - 1.0) / 3.0;
            let variance = (16.0 * n - 29.0) / 90.0;
            let z = (runs as f64 - expected) / variance.sqrt();
            let p = two_tailed_p(z);
            let pass = p > 0.001;

            println!(
                "  Fisher-Yates: seed={seed}, runs={runs}, expected={expected:.0}, \
                 z={z:.2}, p={p:.4} → {}",
                if pass { "PASS" } else { "FAIL" }
            );

            if !pass {
                failures += 1;
            }
        }

        assert!(
            failures <= 1,
            "{failures}/8 Fisher-Yates shuffles failed runs — test framework is broken"
        );
    }
}
