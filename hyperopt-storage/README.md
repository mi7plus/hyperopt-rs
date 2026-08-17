# hyperopt-storage

`Storage` backends for [`hyperopt-rs`](https://crates.io/crates/hyperopt-rs) —
where a study's trial history lives:

- `InMemoryStorage` — fast and dependency-free, but lost when the process exits.
- `SqliteStorage` — schema-versioned, resumable studies (feature `sqlite`): a
  study can be optimized partway, dropped, and resumed in a fresh process, with
  adaptive samplers picking up the loaded history. Several local processes can
  also share one study file.

Both implement the same `Storage` trait (interior-mutable, `Send + Sync`), so a
study can be optimized in parallel over either.

Most users want the [`hyperopt-rs`](https://crates.io/crates/hyperopt-rs) facade,
which re-exports these. See the
[repository](https://github.com/mi7plus/hyperopt-rs) for the full guide.

## License

MIT © mi7plus
