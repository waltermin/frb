use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use frb::Bijection;

/// Benchmark `Bijection::new()` across bit widths.
fn bench_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("construction");

    for &bits in &[8, 16, 32, 64] {
        let size: u64 = if bits == 64 { 1u64 << 63 } else { 1u64 << bits };
        group.bench_with_input(BenchmarkId::new("new", bits), &size, |b, &size| {
            b.iter(|| Bijection::new(black_box(42), black_box(size)));
        });
    }

    group.finish();
}

/// Benchmark `map()` for power-of-two sizes (no cycle walking).
fn bench_map_power_of_two(c: &mut Criterion) {
    let mut group = c.benchmark_group("map_pow2");
    group.throughput(Throughput::Elements(1));

    for &bits in &[8, 16, 24, 32, 48, 63] {
        let size = 1u64 << bits;
        let bij = Bijection::new(42, size);

        group.bench_with_input(BenchmarkId::new("bits", bits), &bij, |b, bij| {
            let mut i = 0u64;
            b.iter(|| {
                let out = bij.map(black_box(i % size));
                i = i.wrapping_add(1);
                out
            });
        });
    }

    group.finish();
}

/// Benchmark `map()` for non-power-of-two sizes (with cycle walking).
/// Uses prime sizes to exercise worst-case rejection ratios.
fn bench_map_non_power_of_two(c: &mut Criterion) {
    let mut group = c.benchmark_group("map_non_pow2");
    group.throughput(Throughput::Elements(1));

    // Primes just above powers of two → nearly 50% rejection rate (worst case)
    let cases: &[(&str, u64)] = &[
        ("257", 257),
        ("65537", 65537),
        ("1000003", 1_000_003),
        ("1073741827", 1_073_741_827), // just above 2^30
    ];

    for &(label, size) in cases {
        let bij = Bijection::new(42, size);

        group.bench_with_input(BenchmarkId::new("size", label), &bij, |b, bij| {
            let mut i = 0u64;
            b.iter(|| {
                let out = bij.map(black_box(i % size));
                i = i.wrapping_add(1);
                out
            });
        });
    }

    group.finish();
}

/// Compare `map()` vs `map_unchecked()` to measure bounds-check overhead.
fn bench_checked_vs_unchecked(c: &mut Criterion) {
    let mut group = c.benchmark_group("checked_vs_unchecked");
    group.throughput(Throughput::Elements(1));

    let size = 1u64 << 32;
    let bij = Bijection::new(42, size);

    group.bench_function("map", |b| {
        let mut i = 0u64;
        b.iter(|| {
            let out = bij.map(black_box(i % size));
            i = i.wrapping_add(1);
            out
        });
    });

    group.bench_function("map_unchecked", |b| {
        let mut i = 0u64;
        b.iter(|| {
            let out = unsafe { bij.map_unchecked(black_box(i % size)) };
            i = i.wrapping_add(1);
            out
        });
    });

    group.bench_function("try_map", |b| {
        let mut i = 0u64;
        b.iter(|| {
            let out = bij.try_map(black_box(i % size));
            i = i.wrapping_add(1);
            out
        });
    });

    group.finish();
}

/// Bulk throughput: map a contiguous range of inputs.
fn bench_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput");

    let counts: &[u64] = &[1_000, 10_000, 100_000];

    for &count in counts {
        let size = 1u64 << 32;
        let bij = Bijection::new(42, size);
        group.throughput(Throughput::Elements(count));

        group.bench_with_input(
            BenchmarkId::new("sequential", count),
            &count,
            |b, &count| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for i in 0..count {
                        sum = sum.wrapping_add(bij.map(i));
                    }
                    black_box(sum)
                });
            },
        );
    }

    group.finish();
}

/// Measure cycle walking cost by comparing a worst-case non-power-of-two
/// size (just above 2^n → ~50% rejection) against exact power of two.
fn bench_cycle_walking_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("cycle_walking_overhead");
    group.throughput(Throughput::Elements(1));

    // 2^20 = 1048576 (no rejection)
    let bij_pow2 = Bijection::new(42, 1 << 20);
    // 2^20 + 1 (worst-case rejection ratio: just over 50%)
    let bij_worst = Bijection::new(42, (1 << 20) + 1);
    // 3/4 of 2^20 (mild rejection: ~25%)
    let bij_mild = Bijection::new(42, (1 << 20) * 3 / 4);

    group.bench_function("pow2_exact", |b| {
        let mut i = 0u64;
        b.iter(|| {
            let out = bij_pow2.map(black_box(i % (1 << 20)));
            i = i.wrapping_add(1);
            out
        });
    });

    group.bench_function("pow2_plus_1", |b| {
        let size = (1u64 << 20) + 1;
        let mut i = 0u64;
        b.iter(|| {
            let out = bij_worst.map(black_box(i % size));
            i = i.wrapping_add(1);
            out
        });
    });

    group.bench_function("three_quarters", |b| {
        let size = (1u64 << 20) * 3 / 4;
        let mut i = 0u64;
        b.iter(|| {
            let out = bij_mild.map(black_box(i % size));
            i = i.wrapping_add(1);
            out
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_construction,
    bench_map_power_of_two,
    bench_map_non_power_of_two,
    bench_checked_vs_unchecked,
    bench_throughput,
    bench_cycle_walking_overhead,
);
criterion_main!(benches);
