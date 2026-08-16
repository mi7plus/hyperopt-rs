use hyperopt_core::{Direction, Trial, TrialState};
use plotters::prelude::*;
use std::path::Path;

/// Error rendering a plot.
#[derive(Debug)]
pub enum VizError {
    /// Nothing to plot (no completed trials).
    NoData,
    /// The plotting backend failed.
    Backend(String),
}

impl std::fmt::Display for VizError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VizError::NoData => write!(f, "no completed trials to plot"),
            VizError::Backend(m) => write!(f, "plotting backend error: {m}"),
        }
    }
}

impl std::error::Error for VizError {}

/// The best-value-so-far series: for each completed trial (in number order),
/// the best objective seen up to and including it, under `direction`.
///
/// Exposed on its own so callers can reuse the exact data the optimization
/// history plot draws (e.g. to render it in some other medium).
pub fn best_so_far(trials: &[Trial], direction: Direction) -> Vec<(usize, f64)> {
    let mut series = Vec::new();
    let mut best: Option<f64> = None;
    let mut ordered: Vec<&Trial> = trials
        .iter()
        .filter(|t| t.state == TrialState::Complete && t.value.is_some())
        .collect();
    ordered.sort_by_key(|t| t.number);
    for t in ordered {
        let v = t.value.unwrap();
        best = Some(match best {
            None => v,
            Some(b) => {
                if direction.is_better(v, b) {
                    v
                } else {
                    b
                }
            }
        });
        series.push((t.number, best.unwrap()));
    }
    series
}

/// Render an **optimization-history** plot to an SVG file: best-value-so-far on
/// the y-axis versus trial number on the x-axis. This is the one-call version
/// of the best-score-vs-evaluations chart, so projects don't re-hand-roll it.
pub fn optimization_history_svg(
    trials: &[Trial],
    direction: Direction,
    path: impl AsRef<Path>,
    title: &str,
) -> Result<(), VizError> {
    let series = best_so_far(trials, direction);
    if series.is_empty() {
        return Err(VizError::NoData);
    }

    let x_min = series.first().unwrap().0 as f64;
    let x_max = series.last().unwrap().0 as f64;
    let x_max = if x_max > x_min { x_max } else { x_min + 1.0 };
    let (y_lo, y_hi) = series.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), (_, v)| {
        (lo.min(*v), hi.max(*v))
    });
    let pad = if (y_hi - y_lo).abs() < f64::EPSILON {
        y_lo.abs().max(1.0) * 0.1
    } else {
        (y_hi - y_lo) * 0.05
    };

    let root = SVGBackend::new(path.as_ref(), (800, 500)).into_drawing_area();
    root.fill(&WHITE).map_err(|e| VizError::Backend(e.to_string()))?;

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 26))
        .margin(16)
        .x_label_area_size(44)
        .y_label_area_size(64)
        .build_cartesian_2d(x_min..x_max, (y_lo - pad)..(y_hi + pad))
        .map_err(|e| VizError::Backend(e.to_string()))?;

    chart
        .configure_mesh()
        .x_desc("Trial")
        .y_desc("Best value so far")
        .draw()
        .map_err(|e| VizError::Backend(e.to_string()))?;

    chart
        .draw_series(LineSeries::new(
            series.iter().map(|(n, v)| (*n as f64, *v)),
            RED.stroke_width(2),
        ))
        .map_err(|e| VizError::Backend(e.to_string()))?;

    chart
        .draw_series(
            series
                .iter()
                .map(|(n, v)| Circle::new((*n as f64, *v), 3, RED.filled())),
        )
        .map_err(|e| VizError::Backend(e.to_string()))?;

    root.present().map_err(|e| VizError::Backend(e.to_string()))?;
    Ok(())
}
