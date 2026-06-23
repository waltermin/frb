use crate::Bijection;

mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_bijection_power_of_two() {
        let size = 256;
        let b = Bijection::new(42, size);
        let outputs: HashSet<u64> = (0..size).map(|i| b.map(i)).collect();
        assert_eq!(
            outputs.len(),
            size as usize,
            "must hit every value exactly once"
        );
    }

    #[test]
    fn test_bijection_non_power_of_two() {
        let size = 1000;
        let b = Bijection::new(123, size);
        let outputs: HashSet<u64> = (0..size).map(|i| b.map(i)).collect();
        assert_eq!(outputs.len(), size as usize);
    }

    #[test]
    fn test_different_seeds_differ() {
        let size = 1024;
        let b1 = Bijection::new(0, size);
        let b2 = Bijection::new(1, size);
        let same = (0..size).filter(|&i| b1.map(i) == b2.map(i)).count();
        assert!(
            same < size as usize / 2,
            "different seeds should produce different permutations"
        );
    }

    #[test]
    fn test_small_sizes() {
        for size in 1..=16 {
            let b = Bijection::new(7, size);
            let outputs: HashSet<u64> = (0..size).map(|i| b.map(i)).collect();
            assert_eq!(outputs.len(), size as usize, "failed for size={size}");
        }
    }

    #[test]
    fn test_output_in_range() {
        let size = 777;
        let b = Bijection::new(0xDEAD, size);
        for i in 0..size {
            let out = b.map(i);
            assert!(out < size, "output {out} out of range for size {size}");
        }
    }

    #[test]
    #[should_panic]
    fn test_panics_on_out_of_range() {
        let b = Bijection::new(0, 100);
        b.map(100);
    }

    #[test]
    fn test_try_map() {
        let b = Bijection::new(42, 500);
        assert!(b.try_map(0).is_some());
        assert!(b.try_map(499).is_some());
        assert!(b.try_map(500).is_none());
        assert!(b.try_map(u64::MAX).is_none());
    }

    #[test]
    fn test_map_unchecked() {
        let size = 256;
        let b = Bijection::new(42, size);
        let outputs: HashSet<u64> = (0..size).map(|i| unsafe { b.map_unchecked(i) }).collect();
        assert_eq!(outputs.len(), size as usize);
    }
}
