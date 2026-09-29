//! GPU kernel microbenchmarks; compilation, upload and checks precede timing.
//! Use SCAN_SOURCE / THRESHOLD_SOURCE for controlled source comparisons and
//! REPORT_DIR to retain detailed compiler reports outside the source tree.
//! Run: HRX_GPU_BENCH=1 HRX_OFFLINE=1 cargo bench --bench batch_kernels.
//! REFERENCE_SCAN_SOURCE / REFERENCE_THRESHOLD_SOURCE enable alternating
//! reference/candidate replays within each Criterion sample under shared load.
//! Query packing is timed on every replay (production packs once per search).
//! Custom SCAN_SOURCE files use FP32 queries unless SCAN_PACK_QUERIES=1;
//! set REFERENCE_PACK_QUERIES=1 for a reference that also uses packed queries.
use criterion::{Criterion, Throughput, criterion_group};
use half::f16;
use hrx::{Buffer, Constants, Device, GraphExec, Kernel, Stream, View, loom};
use std::{
    path::Path,
    time::{Duration, Instant},
};

const ROWS: usize = 65_536;
const DIM: usize = 768;
const BATCH: usize = 256;
const K: usize = 50;
const CAPACITY: usize = 4096;

fn source(option: &str, default: &str) -> String {
    std::env::var(option).map_or_else(
        |_| default.to_owned(),
        |p| std::fs::read_to_string(p).unwrap(),
    )
}

fn compile(stream: &Stream, source: &str, symbol: &str) -> Kernel {
    let compiler = loom::Compiler::for_stream(None, stream).unwrap();
    let mut spec = loom::Specialization::new(symbol).with_report(loom::ReportMode::Details);
    if symbol.starts_with("batch_scan") || symbol == "batch_query_pack" {
        spec.set_config("db.batch", BATCH.to_string());
        spec.set_config("db.scan.dimensions", DIM.to_string());
    }
    let artifact = compiler.module(source).compile(&spec).unwrap();
    if let Ok(dir) = std::env::var("REPORT_DIR") {
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            Path::new(&dir).join(format!(
                "{symbol}-{}.json",
                &hrx::bundle::digest(source.as_bytes())[..12]
            )),
            serde_json::to_vec_pretty(artifact.report().unwrap().json()).unwrap(),
        )
        .unwrap();
    }
    // SAFETY: the benchmark supplies the authored kernels' complete bindings.
    unsafe { stream.load_artifact(&artifact).unwrap() }
}

fn upload(stream: &mut Stream, bytes: &[u8]) -> Buffer {
    let buffer = stream.allocate(bytes.len()).unwrap();
    stream.upload_blocking(buffer.binding(), bytes).unwrap();
    buffer
}

fn floats(stream: &mut Stream, values: impl IntoIterator<Item = f32>) -> Buffer {
    upload(
        stream,
        &values
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>(),
    )
}

fn constants(values: &[usize]) -> Constants {
    let mut constants = Constants::new();
    for &value in values {
        constants.push(value as u32).unwrap();
    }
    constants
}

fn graph(
    stream: &Stream,
    kernel: &Kernel,
    grid: [u32; 3],
    args: &[usize],
    bindings: &[View<'_>],
) -> GraphExec {
    packed_graph(stream, kernel, grid, args, bindings, None)
}

fn packed_graph(
    stream: &Stream,
    kernel: &Kernel,
    grid: [u32; 3],
    args: &[usize],
    bindings: &[View<'_>],
    pack: Option<(&Kernel, [View<'_>; 2])>,
) -> GraphExec {
    let mut graph = stream.graph().unwrap();
    let packed = pack.map(|(pack, inputs)| {
        // SAFETY: both buffers contain DIM * BATCH elements of the declared type.
        unsafe {
            graph
                .dispatch(
                    &[],
                    pack,
                    [(DIM * BATCH / 1024) as u32, 1, 1],
                    pack.info().workgroup_size,
                    &Constants::new(),
                    &inputs,
                )
                .unwrap()
        }
    });
    // SAFETY: callers allocate and initialize every kernel's declared extent.
    unsafe {
        graph
            .dispatch(
                &packed.into_iter().collect::<Vec<_>>(),
                kernel,
                grid,
                kernel.info().workgroup_size,
                &constants(args),
                bindings,
            )
            .unwrap();
    }
    graph.finish().unwrap()
}

fn replay(stream: &mut Stream, graph: &mut GraphExec) {
    stream.launch(graph).unwrap();
    stream.synchronize().unwrap();
}

fn bench_pair(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    case: &str,
    names: [&str; 2],
    stream: &mut Stream,
    graphs: &mut [GraphExec; 2],
) {
    for (arm, name) in names.into_iter().enumerate() {
        let mut pairs = Vec::new();
        group.bench_function(format!("{case}/{name}"), |b| {
            b.iter_custom(|iterations| {
                let mut elapsed = Duration::ZERO;
                for iteration in 0..iterations {
                    let mut times = [Duration::ZERO; 2];
                    for index in if iteration % 2 == 0 { [0, 1] } else { [1, 0] } {
                        let start = Instant::now();
                        replay(stream, &mut graphs[index]);
                        times[index] = start.elapsed();
                    }
                    elapsed += times[arm];
                    pairs.push(times.map(|t| t.as_secs_f64()));
                }
                elapsed
            })
        });
        if let Ok(dir) = std::env::var("REPORT_DIR")
            && !pairs.is_empty()
        {
            let name = format!("{}-{name}-pairs.json", case.replace('/', "-"));
            std::fs::write(
                Path::new(&dir).join(name),
                serde_json::to_vec(&pairs).unwrap(),
            )
            .unwrap();
        }
    }
}

fn random(mut seed: u32) -> impl Iterator<Item = f32> {
    std::iter::from_fn(move || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        Some(((seed >> 8) as f32 / 8388608.0 - 1.0) / (DIM as f32 / 3.0).sqrt())
    })
}

fn scan(c: &mut Criterion) {
    let device = Device::open(0).unwrap();
    let mut stream = device.stream().unwrap();
    let source = source("SCAN_SOURCE", include_str!("../kernels/batch_scan.loom"));
    let kernel = compile(&stream, &source, "batch_scan");
    let corpus: Vec<_> = random(42).take(ROWS * DIM).map(f16::from_f32).collect();
    let data = upload(
        &mut stream,
        &corpus
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    let queries: Vec<_> = random(123).take(DIM * BATCH).collect();
    let query = floats(&mut stream, queries.iter().copied());
    let half_query = stream.allocate_zeroed(DIM * BATCH * 2).unwrap();
    let candidate_packed = std::env::var("SCAN_PACK_QUERIES").is_ok_and(|v| v == "1")
        || (std::env::var_os("SCAN_PACK_QUERIES").is_none()
            && source == include_str!("../kernels/batch_scan.loom"));
    let reference_packed = std::env::var("REFERENCE_PACK_QUERIES").is_ok_and(|v| v == "1");
    let pack = compile(
        &stream,
        include_str!("../kernels/batch_query_pack.loom"),
        "batch_query_pack",
    );
    let mut pack_graph = graph(
        &stream,
        &pack,
        [(DIM * BATCH / 1024) as u32, 1, 1],
        &[],
        &[query.binding(), half_query.binding()],
    );
    c.bench_function("query_pack/256x768", |b| {
        b.iter(|| replay(&mut stream, &mut pack_graph))
    });
    let scan_query = if candidate_packed {
        half_query.binding()
    } else {
        query.binding()
    };
    let norms = floats(&mut stream, std::iter::repeat_n(1.0, ROWS));
    let scores = stream.allocate(ROWS * BATCH * 4).unwrap();
    let mut graph = packed_graph(
        &stream,
        &kernel,
        [(ROWS.div_ceil(64) * BATCH.div_ceil(64)) as u32, 1, 1],
        &[ROWS, 0, 0, ROWS],
        &[
            data.binding(),
            scan_query,
            norms.binding(),
            scores.binding(),
            scores.binding(),
        ],
        candidate_packed.then_some((&pack, [query.binding(), half_query.binding()])),
    );
    replay(&mut stream, &mut graph);
    // Independent dots sample every query and both row boundaries.
    for q in 0..BATCH {
        for row in [0, 31, ROWS - 1] {
            let mut bytes = [0; 4];
            stream
                .read_blocking(
                    scores.try_slice((q * ROWS + row) * 4, 4).unwrap(),
                    &mut bytes,
                )
                .unwrap();
            let expected: f64 = (0..DIM)
                .map(|d| {
                    corpus[row * DIM + d].to_f64() * f16::from_f32(queries[d * BATCH + q]).to_f64()
                })
                .sum();
            assert!((f32::from_le_bytes(bytes) as f64 - expected).abs() < 3e-6);
        }
    }
    let mut group = c.benchmark_group("batch_scan");
    group.throughput(Throughput::Elements((ROWS * BATCH) as u64));
    if let Ok(path) = std::env::var("REFERENCE_SCAN_SOURCE") {
        let reference = compile(
            &stream,
            &std::fs::read_to_string(path).unwrap(),
            "batch_scan",
        );
        let mut reference_graph = packed_graph(
            &stream,
            &reference,
            [(ROWS.div_ceil(64) * BATCH.div_ceil(64)) as u32, 1, 1],
            &[ROWS, 0, 0, ROWS],
            &[
                data.binding(),
                if reference_packed {
                    half_query.binding()
                } else {
                    query.binding()
                },
                norms.binding(),
                scores.binding(),
                scores.binding(),
            ],
            reference_packed.then_some((&pack, [query.binding(), half_query.binding()])),
        );
        let mut candidate_bytes = vec![0; ROWS * BATCH * 4];
        stream
            .read_blocking(scores.binding(), &mut candidate_bytes)
            .unwrap();
        replay(&mut stream, &mut reference_graph);
        let mut reference_bytes = vec![0; ROWS * BATCH * 4];
        stream
            .read_blocking(scores.binding(), &mut reference_bytes)
            .unwrap();
        assert!(
            candidate_bytes == reference_bytes,
            "scan scores differ from reference"
        );
        bench_pair(
            &mut group,
            "256x65536x768",
            ["reference", "candidate"],
            &mut stream,
            &mut [reference_graph, graph],
        );
    } else {
        group.bench_function("256x65536x768", |b| {
            b.iter(|| replay(&mut stream, &mut graph))
        });
    }
    group.finish();
    if source.contains("export(\"batch_scan_pruned\")") {
        let fused = compile(&stream, &source, "batch_scan_pruned");
        let reference = std::env::var("REFERENCE_SCAN_SOURCE").ok().map(|path| {
            compile(
                &stream,
                &std::fs::read_to_string(path).unwrap(),
                "batch_scan_pruned",
            )
        });
        let compact = compile(
            &stream,
            include_str!("../kernels/threshold_select.loom"),
            "threshold_compact",
        );
        let counts = stream.allocate_zeroed(BATCH * 4).unwrap();
        let zero_counts = stream.allocate_zeroed(BATCH * 4).unwrap();
        let candidate_scores = stream.allocate(BATCH * CAPACITY * 4).unwrap();
        let candidate_ids = stream.allocate(BATCH * CAPACITY * 4).unwrap();
        let mut group = c.benchmark_group("scan_filter");
        for bound in [0.04f32, 0.12, 0.16] {
            let running = floats(&mut stream, std::iter::repeat_n(bound, BATCH * K));
            let mut graphs = [false, true].map(|candidate| {
                let prune = candidate || reference.is_some();
                let scan_kernel = if candidate {
                    &fused
                } else {
                    reference.as_ref().unwrap_or(&kernel)
                };
                let mut graph = stream.graph().unwrap();
                let reset = graph
                    .copy(&[], counts.binding(), zero_counts.binding())
                    .unwrap();
                let packed = if !candidate && reference.is_some() {
                    reference_packed
                } else {
                    candidate_packed
                };
                let reset = if packed {
                    // SAFETY: the scan waits for the packed query buffer.
                    unsafe {
                        graph
                            .dispatch(
                                &[reset],
                                &pack,
                                [(DIM * BATCH / 1024) as u32, 1, 1],
                                pack.info().workgroup_size,
                                &Constants::new(),
                                &[query.binding(), half_query.binding()],
                            )
                            .unwrap()
                    }
                } else {
                    reset
                };
                let mut bindings = vec![
                    data.binding(),
                    if packed {
                        half_query.binding()
                    } else {
                        query.binding()
                    },
                    norms.binding(),
                    scores.binding(),
                    scores.binding(),
                ];
                let mut args = vec![ROWS, 0, ROWS, ROWS * 2];
                if prune {
                    args.extend([BATCH, K, CAPACITY]);
                    bindings.extend([
                        running.binding(),
                        counts.binding(),
                        candidate_scores.binding(),
                        candidate_ids.binding(),
                    ]);
                }
                // SAFETY: the scan writes a full score tile; fused candidates
                // use bounded lists, reset counters and immutable thresholds.
                let scan = unsafe {
                    graph
                        .dispatch(
                            &[reset],
                            scan_kernel,
                            [(ROWS.div_ceil(64) * BATCH.div_ceil(64)) as u32, 1, 1],
                            scan_kernel.info().workgroup_size,
                            &constants(&args),
                            &bindings,
                        )
                        .unwrap()
                };
                if !prune {
                    // SAFETY: scan completion precedes reading scores; all
                    // output extents match the fused arm's allocations.
                    unsafe {
                        graph
                            .dispatch(
                                &[scan],
                                &compact,
                                [(ROWS / 1024) as u32, BATCH as u32, 1],
                                [256, 1, 1],
                                &constants(&[ROWS, BATCH, K, ROWS, CAPACITY]),
                                &[
                                    scores.binding(),
                                    running.binding(),
                                    counts.binding(),
                                    candidate_scores.binding(),
                                    candidate_ids.binding(),
                                ],
                            )
                            .unwrap();
                    }
                }
                graph.finish().unwrap()
            });
            let outputs = graphs.each_mut().map(|graph| {
                replay(&mut stream, graph);
                let mut lengths = vec![0; BATCH * 4];
                let mut ids = vec![0; BATCH * CAPACITY * 4];
                let mut values = vec![0; BATCH * CAPACITY * 4];
                stream
                    .read_blocking(counts.binding(), &mut lengths)
                    .unwrap();
                stream
                    .read_blocking(candidate_ids.binding(), &mut ids)
                    .unwrap();
                stream
                    .read_blocking(candidate_scores.binding(), &mut values)
                    .unwrap();
                (0..BATCH)
                    .map(|q| {
                        let n = u32::from_le_bytes(lengths[q * 4..q * 4 + 4].try_into().unwrap())
                            as usize;
                        // Overflow lists retain a scheduling-dependent subset;
                        // exact selection recovers from the dense scores.
                        let mut list: Vec<_> = (0..n.min(CAPACITY))
                            .map(|i| {
                                let at = (q * CAPACITY + i) * 4;
                                (
                                    u32::from_le_bytes(ids[at..at + 4].try_into().unwrap()),
                                    u32::from_le_bytes(values[at..at + 4].try_into().unwrap()),
                                )
                            })
                            .collect();
                        list.sort_unstable();
                        (n, (n <= CAPACITY).then_some(list))
                    })
                    .collect::<Vec<_>>()
            });
            assert_eq!(outputs[0], outputs[1], "fused candidate lists differ");
            bench_pair(
                &mut group,
                &format!("threshold{bound}"),
                if reference.is_some() {
                    ["reference", "candidate"]
                } else {
                    ["separate", "fused"]
                },
                &mut stream,
                &mut graphs,
            );
        }
        group.finish();
    }
}

fn selection(c: &mut Criterion) {
    let device = Device::open(0).unwrap();
    let mut stream = device.stream().unwrap();
    let source = source(
        "THRESHOLD_SOURCE",
        include_str!("../kernels/threshold_select.loom"),
    );
    let compact = compile(&stream, &source, "threshold_compact");
    let merge = compile(&stream, &source, "candidate_merge");
    let reference = std::env::var("REFERENCE_THRESHOLD_SOURCE")
        .ok()
        .map(|path| {
            compile(
                &stream,
                &std::fs::read_to_string(path).unwrap(),
                "candidate_merge",
            )
        });
    let counts = stream.allocate_zeroed(BATCH * 4).unwrap();
    let zero_counts = stream.allocate_zeroed(BATCH * 4).unwrap();
    let candidate_scores = floats(
        &mut stream,
        (0..BATCH * CAPACITY).map(|i| 2.0 + (i % CAPACITY) as f32),
    );
    let candidate_ids = upload(
        &mut stream,
        &(0..BATCH * CAPACITY)
            .flat_map(|i| ((i % CAPACITY + ROWS) as u32).to_le_bytes())
            .collect::<Vec<_>>(),
    );
    let seed_scores = floats(
        &mut stream,
        (0..BATCH * K).map(|i| 1.0 - (i % K) as f32 * 0.001),
    );
    let seed_ids = upload(
        &mut stream,
        &(0..BATCH * K)
            .flat_map(|i| ((i % K) as u32).to_le_bytes())
            .collect::<Vec<_>>(),
    );
    let running_scores = stream.allocate(BATCH * K * 4).unwrap();
    let running_ids = stream.allocate(BATCH * K * 4).unwrap();
    let mut group = c.benchmark_group("candidate_merge");
    for survivors in [0usize, 1, 8, 64, 256, 4096] {
        let seed_counts = upload(
            &mut stream,
            &(0..BATCH)
                .flat_map(|_| (survivors as u32).to_le_bytes())
                .collect::<Vec<_>>(),
        );
        let build = |kernel: &Kernel| {
            let mut graph = stream.graph().unwrap();
            let a = graph
                .copy(&[], counts.binding(), seed_counts.binding())
                .unwrap();
            let b = graph
                .copy(&[a], running_scores.binding(), seed_scores.binding())
                .unwrap();
            let d = graph
                .copy(&[b], running_ids.binding(), seed_ids.binding())
                .unwrap();
            // SAFETY: all 256 independent lists hold k=50 entries; the initialized
            // candidate prefix fits capacity. These cases never read overflow data.
            unsafe {
                graph
                    .dispatch(
                        &[d],
                        kernel,
                        [BATCH as u32, 1, 1],
                        [256, 1, 1],
                        &constants(&[BATCH, K, CAPACITY, 1]),
                        &[
                            counts.binding(),
                            candidate_scores.binding(),
                            candidate_ids.binding(),
                            running_scores.binding(),
                            running_ids.binding(),
                            candidate_scores.binding(),
                            candidate_ids.binding(),
                        ],
                    )
                    .unwrap();
            }
            graph.finish().unwrap()
        };
        let reference_graph = reference.as_ref().map(build);
        let mut graph = build(&merge);
        replay(&mut stream, &mut graph);
        let mut bytes = vec![0; BATCH * K * 4];
        stream
            .read_blocking(running_scores.binding(), &mut bytes)
            .unwrap();
        for list in bytes.chunks_exact(K * 4) {
            for (rank, bytes) in list.chunks_exact(4).enumerate() {
                let expected = if rank < survivors {
                    2.0 + (survivors - 1 - rank) as f32
                } else {
                    1.0 - (rank - survivors) as f32 * 0.001
                };
                assert_eq!(f32::from_le_bytes(bytes.try_into().unwrap()), expected);
            }
        }
        if let Some(reference_graph) = reference_graph {
            bench_pair(
                &mut group,
                &format!("k50/survivors{survivors}"),
                ["reference", "candidate"],
                &mut stream,
                &mut [reference_graph, graph],
            );
        } else {
            group.bench_function(format!("k50/survivors{survivors}"), |b| {
                b.iter(|| replay(&mut stream, &mut graph))
            });
        }
    }
    group.finish();
    let mut group = c.benchmark_group("threshold_compact");
    group.throughput(Throughput::Bytes((ROWS * BATCH * 4) as u64));
    for survivors in [1usize, 64, 4096] {
        let scores = floats(
            &mut stream,
            (0..BATCH * ROWS).map(|i| if i % ROWS < survivors { 2.0 } else { 0.0 }),
        );
        let mut graph = stream.graph().unwrap();
        let a = graph
            .copy(&[], counts.binding(), zero_counts.binding())
            .unwrap();
        // SAFETY: inputs are query-major and all survivor lists fit capacity.
        unsafe {
            graph
                .dispatch(
                    &[a],
                    &compact,
                    [(ROWS / 1024) as u32, BATCH as u32, 1],
                    [256, 1, 1],
                    &constants(&[ROWS, BATCH, K, ROWS, CAPACITY]),
                    &[
                        scores.binding(),
                        seed_scores.binding(),
                        counts.binding(),
                        candidate_scores.binding(),
                        candidate_ids.binding(),
                    ],
                )
                .unwrap();
        }
        let mut graph = graph.finish().unwrap();
        replay(&mut stream, &mut graph);
        let mut bytes = vec![0; BATCH * 4];
        stream.read_blocking(counts.binding(), &mut bytes).unwrap();
        assert!(
            bytes
                .chunks_exact(4)
                .all(|b| u32::from_le_bytes(b.try_into().unwrap()) == survivors as u32)
        );
        group.bench_function(format!("survivors{survivors}"), |b| {
            b.iter(|| replay(&mut stream, &mut graph))
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(30).warm_up_time(Duration::from_secs(1)).measurement_time(Duration::from_secs(3));
    targets = scan, selection
}
fn main() {
    // `cargo test --all-targets` also executes harness-free bench targets.
    if std::env::var("HRX_GPU_BENCH").as_deref() != Ok("1") {
        eprintln!("Set HRX_GPU_BENCH=1 to run the gfx1151 kernel benchmarks.");
        return;
    }
    benches();
    Criterion::default().configure_from_args().final_summary();
}
