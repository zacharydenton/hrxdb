use super::*;

#[test]
#[ignore = "requires gfx1151"]
fn direct_merge_matches_staged_and_cpu_for_every_k() -> Result<()> {
    let device = Device::open(0)?;
    let mut stream = device.stream()?;
    let compiler = hrx::loom::Compiler::for_stream(None, &stream)?;
    let mut old_spec = kernels::named_spec("sorted_merge");
    old_spec.set_config("db.merge.planar", "1");
    let (old, _) = compile(
        &compiler,
        &stream,
        include_str!("../tests/fixtures/batch_merge_staged.loom"),
        old_spec,
    )?;
    let mut spec = kernels::named_spec("batch_merge");
    spec.set_config("db.batch", "1");
    spec.set_config("db.select.limit", TILE_ROWS.to_string());
    let (direct, _) = compile(&compiler, &stream, kernels::BATCH_MERGE, spec)?;
    let batch = 3;
    let bytes = batch * MAX_K * 4;
    let pair = |bytes| Ok::<_, Error>((stream.allocate(bytes)?, stream.allocate(bytes)?));
    let left = pair(bytes)?;
    let right = pair(bytes)?;
    let joined = pair(bytes * 2)?;
    let reference = pair(bytes)?;
    let output = pair(bytes)?;
    for k in 1..=MAX_K {
        let mut lists = [Vec::new(), Vec::new()];
        for (side, list) in lists.iter_mut().enumerate() {
            for q in 0..batch {
                let mut row: Vec<_> = (0..k)
                    .map(|rank| {
                        let id = (2 * rank + side) as u32 + 536_870_912;
                        let score = match q {
                            0 => ((k - rank) / 3) as f32 - (k / 6) as f32,
                            1 => match (rank + side) % 9 {
                                0 => f32::INFINITY,
                                1 => -0.0,
                                2 => 0.0,
                                3 => f32::from_bits(1),
                                4 => -f32::from_bits(1),
                                5 => f32::NEG_INFINITY,
                                _ => rank as f32 - k as f32 / 2.0,
                            },
                            _ => f32::NEG_INFINITY,
                        };
                        let id = if score == f32::NEG_INFINITY {
                            i32::MAX as u32
                        } else {
                            id
                        };
                        (score, id)
                    })
                    .collect();
                row.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
                list.extend(row);
            }
        }
        for (buffers, values) in [(&left, &lists[0]), (&right, &lists[1])] {
            stream.upload(
                buffers.0.binding(),
                &values
                    .iter()
                    .flat_map(|x| x.0.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            stream.upload(
                buffers.1.binding(),
                &values
                    .iter()
                    .flat_map(|x| x.1.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
        }
        let extent = batch * k * 4;
        for (joined, a, b) in [
            (&joined.0, &left.0, &right.0),
            (&joined.1, &left.1, &right.1),
        ] {
            stream.copy(joined.try_slice(0, extent)?, a.try_slice(0, extent)?)?;
            stream.copy(joined.try_slice(extent, extent)?, b.try_slice(0, extent)?)?;
        }
        let mut constants = Constants::new();
        constants.push((batch * 2) as u32)?;
        constants.push(k as u32)?;
        // SAFETY: joined contains two planar batches; the output is disjoint.
        unsafe {
            stream.dispatch(
                &old,
                [batch as u32, 1, 1],
                [256, 1, 1],
                &constants,
                &[
                    joined.0.binding(),
                    joined.1.binding(),
                    reference.0.binding(),
                    reference.1.binding(),
                ],
            )?;
        }
        let mut constants = Constants::new();
        constants.push(batch as u32)?;
        constants.push(k as u32)?;
        constants.push(k.ilog2() + 1)?;
        stream.fill(output.0.try_slice(0, extent)?, 0xff)?;
        stream.fill(output.1.try_slice(0, extent)?, 0xff)?;
        // SAFETY: each input has batch*k sorted pairs and neither aliases output.
        unsafe {
            stream.dispatch(
                &direct,
                [batch as u32, 1, 1],
                [256, 1, 1],
                &constants,
                &[
                    left.0.try_slice(0, extent)?,
                    left.1.try_slice(0, extent)?,
                    right.0.try_slice(0, extent)?,
                    right.1.try_slice(0, extent)?,
                    output.0.try_slice(0, extent)?,
                    output.1.try_slice(0, extent)?,
                ],
            )?;
        }
        let mut actual = (vec![0; extent], vec![0; extent]);
        let mut expected = actual.clone();
        for (buffers, bytes) in [(&reference, &mut expected), (&output, &mut actual)] {
            stream.read_blocking(buffers.0.try_slice(0, extent)?, &mut bytes.0)?;
            stream.read_blocking(buffers.1.try_slice(0, extent)?, &mut bytes.1)?;
        }
        assert_eq!(actual, expected, "k={k}");
        for q in 0..batch {
            let mut expected: Vec<_> = lists[0][q * k..(q + 1) * k]
                .iter()
                .chain(&lists[1][q * k..(q + 1) * k])
                .copied()
                .collect();
            // Stable sorting also preserves the left-list tie rule for sentinels.
            expected.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
            for (rank, &(score, id)) in expected[..k].iter().enumerate() {
                let at = (q * k + rank) * 4;
                assert_eq!(
                    actual.0[at..at + 4],
                    score.to_le_bytes(),
                    "k={k} q={q} rank={rank}"
                );
                assert_eq!(
                    actual.1[at..at + 4],
                    id.to_le_bytes(),
                    "k={k} q={q} rank={rank}"
                );
            }
        }
    }
    Ok(())
}
