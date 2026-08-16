use hyperopt_core::{Distribution, Sampler, StudyState, Trial, Value};
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::random::sample_value;

/// Exhaustive search over a caller-provided discrete grid.
///
/// Unlike [`crate::RandomSampler`], which can sample continuous distributions
/// directly, `GridSampler` only makes sense for parameters with a **finite,
/// pre-specified set of values** — you must register each parameter's grid up
/// front with [`GridSampler::add_grid`] (or the typed helpers). The trials are
/// enumerated as a mixed-radix odometer over the registered grids: trial `n`
/// maps deterministically to one point in the Cartesian product, in the order
/// the grids were added.
///
/// Notes / limitations:
/// - Because this is define-by-run, a given trial may not request every
///   registered parameter (conditional spaces). Enumeration still ranges over
///   the full product; unused coordinates simply don't get read that trial.
/// - If more trials are run than there are grid points, the odometer wraps
///   (trial `n` uses combination `n % total`).
/// - A parameter that is suggested but was never registered has no grid to draw
///   from, so it falls back to a random draw (seeded, for reproducibility) and
///   is documented as such rather than silently returning a constant.
pub struct GridSampler {
    grids: Vec<(String, Vec<Value>)>,
    fallback_rng: StdRng,
}

impl GridSampler {
    /// A grid sampler with no parameters registered yet.
    pub fn new() -> Self {
        GridSampler {
            grids: Vec::new(),
            fallback_rng: StdRng::seed_from_u64(0),
        }
    }

    /// Register a parameter's grid of explicit values. Order of registration
    /// defines the odometer axis order.
    pub fn add_grid(mut self, name: &str, values: Vec<Value>) -> Self {
        self.grids.push((name.to_string(), values));
        self
    }

    /// Register a float-valued grid.
    pub fn add_float_grid(self, name: &str, values: &[f64]) -> Self {
        self.add_grid(name, values.iter().map(|v| Value::Float(*v)).collect())
    }

    /// Register an integer-valued grid.
    pub fn add_int_grid(self, name: &str, values: &[i64]) -> Self {
        self.add_grid(name, values.iter().map(|v| Value::Int(*v)).collect())
    }

    /// Register a categorical grid.
    pub fn add_categorical_grid(self, name: &str, values: &[&str]) -> Self {
        self.add_grid(
            name,
            values.iter().map(|v| Value::Categorical(v.to_string())).collect(),
        )
    }

    /// Total number of grid points (the product of all registered grid sizes).
    /// Running this many trials covers the whole grid exactly once.
    pub fn grid_size(&self) -> usize {
        self.grids
            .iter()
            .map(|(_, v)| v.len().max(1))
            .product::<usize>()
            .max(1)
    }

    /// The index into `name`'s grid for the given trial number, via mixed-radix
    /// decomposition over the registered grids.
    fn index_for(&self, name: &str, trial_number: usize) -> Option<usize> {
        // Product of the radices of the axes *after* the target axis.
        let mut suffix_product = 1usize;
        let mut target_len = None;
        let mut divisor = 1usize;
        for (grid_name, values) in self.grids.iter().rev() {
            let radix = values.len().max(1);
            if grid_name == name {
                target_len = Some(values.len());
                divisor = suffix_product;
            }
            suffix_product = suffix_product.saturating_mul(radix);
        }
        let len = target_len?;
        if len == 0 {
            return None;
        }
        let total = self.grid_size();
        let combo = trial_number % total;
        Some((combo / divisor) % len)
    }
}

impl Default for GridSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl Sampler for GridSampler {
    fn suggest(
        &mut self,
        _study_state: &StudyState,
        trial: &Trial,
        param_name: &str,
        distribution: &Distribution,
    ) -> Value {
        if let Some(idx) = self.index_for(param_name, trial.number) {
            if let Some((_, values)) = self.grids.iter().find(|(n, _)| n == param_name) {
                if let Some(v) = values.get(idx) {
                    return v.clone();
                }
            }
        }
        // Unregistered parameter — fall back to a (seeded) random draw.
        sample_value(&mut self.fallback_rng, distribution)
    }
}
