use hyperopt_core::{Trial, TrialState, Value};
use std::collections::BTreeMap;

/// A parameter's estimated importance: the fraction of objective variance a
/// single split on that parameter explains, plus its share once all parameters
/// are normalized to sum to 1.
#[derive(Debug, Clone, PartialEq)]
pub struct Importance {
    pub name: String,
    /// Variance explained by the best single split on this parameter, in
    /// `[0, 1]` (a decision-stump R²).
    pub variance_explained: f64,
    /// `variance_explained` normalized so all reported importances sum to 1.
    pub normalized: f64,
}

/// A simple, documented **proxy** for parameter importance.
///
/// Full fANOVA (Optuna's method) is out of scope for v1. Instead, for each
/// parameter this fits the best single decision stump (one threshold split for
/// numeric parameters, a group-by for categoricals) on `parameter -> objective`
/// across completed trials and reports the fraction of objective variance that
/// split explains. It is **not** the same statistical method as fANOVA and
/// ignores interaction effects, but it reliably ranks a genuinely influential
/// parameter above an irrelevant one and needs no extra statistical machinery.
///
/// Returns importances sorted most-important first. Parameters that never vary
/// (or never appear) contribute zero.
pub fn parameter_importance(trials: &[Trial]) -> Vec<Importance> {
    // Completed trials with a final value.
    let completed: Vec<&Trial> = trials
        .iter()
        .filter(|t| t.state == TrialState::Complete && t.value.is_some())
        .collect();

    // Gather the set of parameter names in first-seen order.
    let mut names: Vec<String> = Vec::new();
    for t in &completed {
        for p in &t.params {
            if !names.contains(&p.name) {
                names.push(p.name.clone());
            }
        }
    }

    let mut raw: Vec<Importance> = names
        .into_iter()
        .map(|name| {
            let mut xs: Vec<&Value> = Vec::new();
            let mut ys: Vec<f64> = Vec::new();
            for t in &completed {
                if let Some(v) = t.param_value(&name) {
                    xs.push(v);
                    ys.push(t.value.unwrap());
                }
            }
            let variance_explained = stump_variance_explained(&xs, &ys);
            Importance {
                name,
                variance_explained,
                normalized: 0.0,
            }
        })
        .collect();

    let total: f64 = raw.iter().map(|i| i.variance_explained).sum();
    if total > 0.0 {
        for imp in &mut raw {
            imp.normalized = imp.variance_explained / total;
        }
    }

    raw.sort_by(|a, b| b.variance_explained.total_cmp(&a.variance_explained));
    raw
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

fn variance(v: &[f64]) -> f64 {
    if v.len() < 2 {
        return 0.0;
    }
    let m = mean(v);
    v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64
}

/// Fraction of variance in `ys` explained by the best single split on `xs`.
fn stump_variance_explained(xs: &[&Value], ys: &[f64]) -> f64 {
    let n = ys.len();
    if n < 2 {
        return 0.0;
    }
    let total_var = variance(ys);
    if total_var <= 0.0 {
        return 0.0;
    }

    // Categorical: group-by label.
    if xs.iter().all(|v| v.as_categorical().is_some()) {
        let mut groups: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
        for (x, y) in xs.iter().zip(ys) {
            groups.entry(x.as_categorical().unwrap()).or_default().push(*y);
        }
        if groups.len() < 2 {
            return 0.0;
        }
        let weighted: f64 = groups
            .values()
            .map(|g| g.len() as f64 * variance(g))
            .sum::<f64>()
            / n as f64;
        return ((total_var - weighted) / total_var).clamp(0.0, 1.0);
    }

    // Numeric: best threshold split.
    let mut pairs: Vec<(f64, f64)> = xs
        .iter()
        .zip(ys)
        .filter_map(|(x, y)| x.to_f64().map(|xf| (xf, *y)))
        .collect();
    if pairs.len() < 2 {
        return 0.0;
    }
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut best_weighted = total_var; // no worse than not splitting
    for i in 1..pairs.len() {
        // Only split between distinct x values.
        if pairs[i].0 == pairs[i - 1].0 {
            continue;
        }
        let left: Vec<f64> = pairs[..i].iter().map(|p| p.1).collect();
        let right: Vec<f64> = pairs[i..].iter().map(|p| p.1).collect();
        let weighted =
            (left.len() as f64 * variance(&left) + right.len() as f64 * variance(&right))
                / n as f64;
        if weighted < best_weighted {
            best_weighted = weighted;
        }
    }
    ((total_var - best_weighted) / total_var).clamp(0.0, 1.0)
}
