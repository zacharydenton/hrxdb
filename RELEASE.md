# Releasing hrxdb

hrxdb `0.3.0` accepts compatible `hrx-rs 0.7` releases; its lockfile qualifies
`0.7.0`. Earlier `0.1.0` release artifacts:

- [Source and release notes](https://github.com/zacharydenton/hrxdb/releases/tag/v0.1.0)
- [crates.io package](https://crates.io/crates/hrxdb/0.1.0)
- [API reference](https://docs.rs/hrxdb/0.1.0/hrxdb/)

There is no automatic publishing workflow. Repository visibility, package
publication, and release creation are managed explicitly.

## 0.3 qualification — 2026-09-17

All 36 hardware correctness tests pass against published HRX 0.7.0, including
prepared inference, retained output leases and the large shard-boundary cases.
CPU tests, doctests, all-target Clippy and unpacked-crate compilation also pass.
This release migrates the graph API; it does not claim improved search speed.

## 0.2 candidate qualification — 2026-09-16

All 36 hardware correctness tests pass on gfx1151 with the HRX 0.6 candidate,
including shared-context resident queries, retained-output backpressure,
budgeted workspace growth/refusal, corpus pins, and the 9.4 GB shard-boundary
case. Performance comparisons are excluded from that correctness command.
CPU tests, Clippy and the registry-dependency package are checked separately
before publication. RSS, corpus bytes and workspace bytes are different scopes;
these checks do not establish a throughput or whole-process memory improvement.

## Earlier candidate qualification — 2026-09-13

Local checks passed with the registry dependency `hrx-rs 0.5.0`:

- Rust 1.88 and stable: all-target CPU tests (8 tests) and warning-free Clippy.
- Formatting, 6 doctests, and Rustdoc with warnings denied.
- All 33 hardware correctness tests on gfx1151, including the large allocation
  and shard-boundary cases; opt-in performance comparisons excluded.
- All four examples and single-query/batch benchmark CLI smoke runs.
- Publication dry run, including compilation of the unpacked crate.
- Local documentation links and versioned dependency setup/license links.
- Gitleaks 8.30.1 found no leaks across all 11 pre-release commits. Cargo-audit
  0.22.2 reported no known vulnerabilities or warnings with RustSec database
  commit `b50980aad8b8f14f77e25a97b32dd94bf008b0af` (updated September 9).

The final release commit must also pass CI. Qualification includes a clean-tree
publication dry run and running the README quick start against the unpacked
crate. Verify public endpoints after publishing.

## Validate the release candidate

Use Rust 1.91 and stable. From the repository root:

```sh
cargo fmt --all -- --check
HRX_OFFLINE=1 cargo clippy --locked --all-targets -- -D warnings
HRX_OFFLINE=1 cargo test --locked --all-targets
HRX_OFFLINE=1 cargo test --locked --doc
HRX_OFFLINE=1 RUSTDOCFLAGS='-D warnings' cargo doc --locked --no-deps
cargo package --locked
cargo publish --dry-run --locked
```

Builds and documentation require no GPU or native runtime download. Packaging
builds the unpacked archive using registry dependencies. During preparation,
`--allow-dirty` permits checking uncommitted edits; repeat on the clean release
commit before publishing.

Inspect `cargo package --list`: source, Loom kernels, examples, tests, the API
guide, contributor guide, release notes, and MIT license belong in the crate.
CI configuration and bulky benchmark artifacts stay in the repository. Confirm
README and guide links work in their rendered destinations.

On a prepared gfx1151 host, run the hardware checks and all four examples from
[CONTRIBUTING.md](CONTRIBUTING.md). The hardware correctness command deliberately
excludes opt-in benchmark comparisons:

```sh
HRX_OFFLINE=1 cargo test --locked --release -- --ignored --test-threads=1 --skip batch::bench
HRX_OFFLINE=1 cargo run --locked --release --bin hrxdb-bench -- --rows 100000 --samples 3
HRX_OFFLINE=1 cargo run --locked --release --bin hrxdb-bench -- --rows 100000 --batch 9 --k 33 --samples 3
```

The largest correctness test allocates about 9.4 GB. Run hardware work
sequentially and keep benchmarks separate from tests and other GPU activity.
Runtime/compiler changes require hardware qualification. Preserve raw JSON and
methodology for any new public performance claim; a small smoke run does not
replace large-corpus performance qualification.

Before exposing history, scan all commits for credentials and review tracked
artifacts for private data. Audit the lockfile against current RustSec advisories.
Record tool versions and findings; automated scans cannot establish that every
secret or unknown vulnerability is absent. The MIT license covers this crate;
the separately distributed runtime/compiler carries its own
[third-party notices](https://github.com/zacharydenton/hrx-rs/blob/v0.7.0/THIRD-PARTY.md).

## Publication procedure

For the already public repository and package:

1. Confirm the candidate version is not published and the publishing account
   has access. Verify the repository license, description, topics,
   and dependency source links.
2. Update installation instructions, versioned links, and the dated changelog
   entry. Commit and push the final release contents.
3. Confirm CI passes for that exact commit. Repeat packaging and the publication
   dry run from the clean tree; retain the hardware qualification results.
4. Verify anonymous access to the README, examples, API guide, and benchmark
   evidence before publishing the crate. Do not change repository visibility.
5. Run `cargo publish --locked`. Create and push `v0.3.0` on the qualified commit,
   then create a GitHub release using the `0.3.0` changelog entry.
6. Verify the crates.io README, docs.rs build, and a fresh consuming project.
   Check `cargo install hrxdb --version 0.3.0 --locked`; the executable is
   `hrxdb-bench`. Check execution separately on supported hardware.

For subsequent releases, update the manifest, lockfile, changelog, and versioned
commands together. Treat ownership, ID semantics, ordering, precision, and
supported platforms as public API contracts.
