# frb

`frb` (fast random bijection) is a high-performance library that generates a psuedorandom bijection over an arbitrarily sized set. In more human terms, you get a function that randomly and uniquely maps every input to an output within a range. Useful for a number of things, such as shuffling a massive set of items without actually storing the shuffled set.

## Why use this?
- It's constant-memory. A `Bijection` struct is a constant 48 bytes, regardless of the size.
- It's stateless and deterministic. A given seed will *always* produce the same bijection.
- It's *fast*. Mapping an input to an output takes on the order of single-digit nanoseconds on my laptop (Framework 13, Ryzen 7840U). See `benchmarks.txt` for a full set of numbers.

## Why not use this?
- It's *not* cryptographically secure. With just a couple input-output pairs, an attacker can reconstruct the seed for a given bijection, and predict all other mappings. If that matters for your application, you should consider using a different approach, such as cycle-walking a Feistel cipher. If you're just shuffling a really big playlist, you're probably fine.

## Example

```rust
use frb::Bijection;

let b = Bijection::new(0xC0FFEE, 1_000_000);

// Walk 0..N and emit a shuffled permutation of the same range.
let shuffled: Vec<u64> = (0..10).map(|i| b.map(i)).collect();

// Every output is distinct and lies in [0, 1_000_000).
let mut sorted = shuffled.clone();
sorted.sort();
sorted.dedup();
assert_eq!(sorted.len(), shuffled.len());
assert!(shuffled.iter().all(|&x| x < 1_000_000));
```

## License

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
