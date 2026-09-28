use super::*;

/// Exercise the selector itself, including score patterns absent from random
/// cosine benchmarks. Compare raw bits with its predecessor and IDs with a CPU
/// oracle; signed zeros compare numerically equal for the documented tie rule.
#[test]
#[ignore = "requires gfx1151"]
fn selection_matches_predecessor_for_special_values_and_tails() -> Result<()> {
    let device = Device::open(0)?;
    let mut stream = device.stream()?;
    let compiler = hrx::loom::Compiler::for_stream(None, &stream)?;
    let source = std::env::var("CANDIDATE_SOURCE")
        .ok()
        .map(std::fs::read_to_string)
        .transpose()?
        .unwrap_or_else(|| kernels::SELECT.to_owned());
    let batch = 3;
    let limit = 32777;
    let mut plans = Vec::new();
    for source in [
        include_str!("../tests/fixtures/select_two_reductions.loom"),
        &source,
    ] {
        let (mut plan, _) = SelectionPlan::new(&compiler, &stream, batch, limit)?;
        for first in [true, false] {
            let mut spec = kernels::select_spec(first);
            spec.set_config("db.batch", batch.to_string());
            spec.set_config("db.select.limit", limit.to_string());
            let (kernel, _) = compile(&compiler, &stream, source, spec)?;
            if first {
                plan.first = kernel;
            } else {
                plan.merge = kernel;
            }
        }
        plans.push(plan);
    }
    let bytes = batch * limit.div_ceil(1024) * 32 * 4;
    let pair = || Ok::<_, Error>((stream.allocate(bytes)?, stream.allocate(bytes)?));
    let candidates = [pair()?, pair()?];
    let origin = 536_870_912;
    for rows in [1, 31, 1025, limit] {
        for pattern in 0..4 {
            let values: Vec<f32> = (0..batch * rows)
                .map(|i| match pattern {
                    0 => match (i * 7 + i / rows) % 13 {
                        0 => f32::NAN,
                        1 => f32::NEG_INFINITY,
                        2 => f32::INFINITY,
                        3 => -0.0,
                        4 => 0.0,
                        5 => f32::from_bits(1),
                        6 => -f32::from_bits(1),
                        7 => f32::MIN_POSITIVE,
                        8 => -f32::MIN_POSITIVE,
                        9 => f32::MAX,
                        10 => -f32::MAX,
                        _ => ((i * 17) % 7) as f32 - 3.0,
                    },
                    1 => {
                        if i % 3 == 0 {
                            -0.0
                        } else {
                            0.0
                        }
                    }
                    2 => {
                        if i % 2 == 0 {
                            f32::NAN
                        } else {
                            f32::NEG_INFINITY
                        }
                    }
                    _ => ((i * 17) % 11) as f32 - 5.0,
                })
                .collect();
            let input = stream.allocate(values.len() * 4)?;
            stream.upload(
                input.binding(),
                &values
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            for k in [1, 2, 5, 10, 31, 32] {
                let mut outputs = Vec::new();
                for plan in &plans {
                    let selected = select_scores(
                        &stream,
                        &candidates,
                        plan,
                        SelectionInput {
                            scores: input.binding(),
                            rows,
                            start: origin,
                            k,
                            batch,
                        },
                    )?;
                    let (scores, ids) = &candidates[selected];
                    let mut score_bytes = vec![0; batch * k * 4];
                    let mut id_bytes = score_bytes.clone();
                    stream
                        .read_blocking(scores.try_slice(0, score_bytes.len())?, &mut score_bytes)?;
                    stream.read_blocking(ids.try_slice(0, id_bytes.len())?, &mut id_bytes)?;
                    outputs.push((score_bytes, id_bytes));
                }
                assert_eq!(
                    outputs[0], outputs[1],
                    "rows={rows} pattern={pattern} k={k}"
                );
                for q in 0..batch {
                    let mut expected: Vec<_> = (0..rows)
                        .filter(|&i| {
                            !values[q * rows + i].is_nan()
                                && values[q * rows + i] != f32::NEG_INFINITY
                        })
                        .collect();
                    expected.sort_by(|&a, &b| {
                        values[q * rows + b]
                            .partial_cmp(&values[q * rows + a])
                            .unwrap()
                            .then(a.cmp(&b))
                    });
                    expected.truncate(k);
                    for (rank, &id) in expected.iter().enumerate() {
                        let at = (q * k + rank) * 4;
                        let got = u32::from_le_bytes(outputs[1].1[at..at + 4].try_into().unwrap());
                        assert_eq!(
                            got as usize,
                            id + origin,
                            "rows={rows} pattern={pattern} k={k} q={q}"
                        );
                    }
                }
            }
        }
    }
    Ok(())
}
