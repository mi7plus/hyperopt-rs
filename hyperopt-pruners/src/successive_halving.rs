use hyperopt_core::{Pruner, StudyState, Trial};

/// ASHA-style (Asynchronous Successive Halving) pruner — the "advanced" option.
///
/// Trials are compared at a ladder of resource **rungs**
/// `min_resource * reduction_factor^k` (starting at exponent
/// `min_early_stopping_rate`). When a trial crosses a rung, only the top
/// `1/reduction_factor` fraction of the trials that reached that rung are
/// promoted to continue; the rest are pruned. Compared to [`MedianPruner`],
/// this allocates a shrinking budget across rungs rather than a single
/// median cut, and is asynchronous (each trial is judged against whichever
/// peers have reached its rung so far), which suits parallel execution.
///
/// Implemented second and treated as the advanced choice; [`MedianPruner`] is
/// the simpler default.
#[derive(Debug, Clone)]
pub struct SuccessiveHalvingPruner {
    min_resource: usize,
    reduction_factor: usize,
    min_early_stopping_rate: u32,
}

impl Default for SuccessiveHalvingPruner {
    fn default() -> Self {
        SuccessiveHalvingPruner {
            min_resource: 1,
            reduction_factor: 4,
            min_early_stopping_rate: 0,
        }
    }
}

impl SuccessiveHalvingPruner {
    /// Defaults: `min_resource = 1`, `reduction_factor = 4`,
    /// `min_early_stopping_rate = 0`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Resource (step count) of the first rung.
    pub fn min_resource(mut self, r: usize) -> Self {
        self.min_resource = r.max(1);
        self
    }

    /// `eta`: the fraction `1/eta` of trials promoted at each rung, and the
    /// factor by which rung resources grow. Must be `>= 2`.
    pub fn reduction_factor(mut self, eta: usize) -> Self {
        self.reduction_factor = eta.max(2);
        self
    }

    /// Starting rung exponent — skip the earliest, cheapest rungs.
    pub fn min_early_stopping_rate(mut self, s: u32) -> Self {
        self.min_early_stopping_rate = s;
        self
    }

    /// The highest rung resource `<= step`, or `None` if `step` hasn't reached
    /// the first rung yet.
    fn rung_resource_for(&self, step: usize) -> Option<usize> {
        let eta = self.reduction_factor as u64;
        let first = (self.min_resource as u64) * eta.pow(self.min_early_stopping_rate);
        if (step as u64) < first {
            return None;
        }
        let mut rung = first;
        loop {
            let next = rung.saturating_mul(eta);
            if next <= step as u64 {
                rung = next;
            } else {
                break;
            }
        }
        Some(rung as usize)
    }
}

impl Pruner for SuccessiveHalvingPruner {
    fn should_prune(&self, study_state: &StudyState, trial: &Trial) -> bool {
        let Some((step, value)) = trial.last_intermediate() else {
            return false;
        };
        let Some(rung) = self.rung_resource_for(step) else {
            return false;
        };

        // Peers that have reached this rung (the current trial is not in the
        // snapshot, so it is counted separately below).
        let peers = study_state.values_at_or_after(rung);
        let eta = self.reduction_factor;

        // A rung must be "full enough" before anyone is promoted or cut.
        if peers.len() + 1 < eta {
            return false;
        }

        let direction = study_state.direction();
        let better_than_current = peers
            .iter()
            .filter(|&&p| direction.is_better(p, value))
            .count();

        // Total trials at this rung including the current one.
        let total = peers.len() + 1;
        let top_k = (total / eta).max(1);

        // Promote (keep) if the current trial ranks within the top 1/eta.
        let rank = better_than_current; // number strictly better than current
        rank >= top_k
    }
}
