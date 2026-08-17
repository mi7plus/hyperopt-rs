# hyperopt-distributed

Multi-machine distributed execution for
[`hyperopt-rs`](https://crates.io/crates/hyperopt-rs): a **coordinator** server
owns one authoritative study (sampler + pruner + storage) and hands out
parameter suggestions, while **workers** on any number of machines run the
objective and round-trip each `suggest_*` / `report` / `should_prune` call back.

This is the true-distributed counterpart to `SqliteStorage`'s shared-file model.
The coordinator assigns every trial number atomically and runs the one sampler
against the one history, so N machines behave like one big
`Study::optimize_parallel` — with the same intentional slightly-stale-snapshot
semantics under concurrency.

- **Transport** — newline-delimited JSON over `std::net` TCP, one thread per
  worker connection. No async runtime.
- **Auth** (optional, `std`-only) — a shared-secret token via
  `Coordinator::require_token` / `Worker::authenticate`, compared in constant
  time.
- **TLS** (optional, `tls` feature) — `rustls` with the `ring` provider (no
  OpenSSL): `Coordinator::listen_tls` / `Worker::connect_tls`, taking DER
  certificate/key bytes.

Because the remote trial implements the same `Suggest` trait as `TrialContext`,
one objective closure runs unchanged locally or distributed. See the
[repository](https://github.com/mi7plus/hyperopt-rs) for the full guide.

## License

MIT © mi7plus
