# Releasing hrxdb

The first release is `0.1.0`, with the published `hrx-rs =0.4.0` dependency.
The repository is `https://github.com/zacharydenton/hrxdb`, currently private.
Making it public, tagging, and crates.io publication are separate release actions.
There is no workflow that automatically publishes packages.

## 0.1.0 qualification — 2026-09-11

Local release checks passed against the registry version of `hrx-rs 0.4.0`:

- Rust 1.88 and stable: all-target CPU tests, doctests, and warning-free Clippy.
- Rustdoc with warnings denied, formatting, and archive contents inspection.
- Five opt-in GPU tests, including cross-thread query/drop, all 16 scan
  schedules, exact selection, and the 7.68 GB allocation boundary test.
- The threaded external-ID example and benchmark CLI smoke test, including
  cache-relative artifact identifiers in its saved JSON.
- `cargo publish --dry-run --allow-dirty --locked`, including a successful build
  of the unpacked release archive using registry dependencies.

These are local qualification results; GitHub-hosted CI reports its status on
each pushed commit. No release tag or crates.io publication has been created.
Generate `target/package/hrxdb-0.1.0.crate` using the clean-commit commands below
for the eventual publication.

## Validate the release candidate

From the repository root:

```sh
cargo fmt --all -- --check
cargo clippy --locked --all-targets -- -D warnings
cargo +1.88.0 test --locked --all-targets
cargo test --locked --doc
RUSTDOCFLAGS='-D warnings' cargo doc --locked --no-deps
cargo package --locked
cargo publish --dry-run --locked
```

CPU tests and docs work without a GPU and require no native runtime download.
On a prepared gfx1151 machine, also run these separately from benchmarks:

```sh
HRX_OFFLINE=1 cargo test --locked --release -- --ignored --test-threads=1
HRX_OFFLINE=1 cargo run --locked --release --example search
HRX_OFFLINE=1 cargo run --locked --release --bin hrxdb-bench -- --output default.json
```

The full hardware suite allocates about 7.8 GB. Preserve the JSON from any new
performance qualification, including percentiles, schedule, and compiler
metadata. Runtime/compiler changes require hardware qualification; the initial
dependency is pinned because the kernels were qualified against that bundle.

`cargo package` builds the unpacked archive, exercising registry dependencies
instead of local path overrides. Check `cargo package --list` before publishing:
source, Loom kernels, examples, tests, README, changelog, and license belong in
the crate. Repository CI files and bulky benchmark artifacts do not. The
benchmark reports and disassembly remain available in the Git repository.

## Publish an approved release

After the repository and package are ready for public distribution:

1. Commit the reviewed release contents and push the repository to the intended
   public remote. Confirm CI passes for the same commit.
2. Confirm the `hrxdb` crates.io name is still available and authenticate the
   publishing account with Cargo.
3. Run `cargo publish --locked` from that clean commit.
4. Create and push the `v0.1.0` tag on that commit, then create a GitHub release
   using the `0.1.0` changelog entry.
5. Verify installation with `cargo install hrxdb --version 0.1.0 --locked`, the
   docs.rs build, and the package README links. The executable is `hrxdb-bench`.

For later releases, update the manifest, lockfile, changelog, and these versioned
commands together. Changes to `Send`, ID semantics, ranking order, precision,
or supported platforms are part of the public API contract.
