use super::*;

fn axis(n: usize) -> [f32; 3] {
    let mut row = [0.0; 3];
    row[n % 3] = 1.0;
    row
}

#[test]
#[ignore = "requires gfx1151"]
fn wide_workspace_growth_snapshots_and_replay() -> Result<()> {
    let device = Device::open(0)?;
    let base = Corpus::build(&device, 3, (0..131_139).map(axis))?;
    let updated = base
        .clone()
        .update(&[0, 65_536, 131_138], [[0.0, 0.0, -1.0]; 3])?;
    let appended = updated.append((0..257).map(axis).collect::<Vec<_>>())?;
    let small = Corpus::build(&device, 3, (0..2057).map(axis))?;
    let mut db = base.searcher()?;
    let mut reserved_width = 0;
    for corpus in [base.clone(), updated, appended, small, base] {
        db.set_corpus(corpus.clone())?;
        assert!(db.batch.as_ref().is_none_or(|s| s.search_graph.is_none()));
        let mut reference = corpus.searcher()?;
        for (iteration, (batch, k)) in [
            (64usize, 5),
            (65, 5),
            (128, 33),
            (129, 33),
            (256, 33),
            (3, 1024),
            (65, 5),
            (256, 32),
            (256, 32),
        ]
        .into_iter()
        .enumerate()
        {
            let queries: Vec<_> = (0..batch).flat_map(|q| axis(q + iteration)).collect();
            let excluded = [iteration as u32, 1024];
            let actual = db.search_batch_excluding(&queries, k, &excluded)?;
            let expected: Vec<_> = (0..3)
                .map(|q| reference.search_excluding(&axis(q), k, &excluded))
                .collect::<Result<_>>()?;
            for (q, actual) in actual.iter().enumerate() {
                assert_eq!(*actual, expected[(q + iteration) % 3]);
            }
            reserved_width = reserved_width.max(batch.next_power_of_two().max(8));
            let scratch = db.batch.as_ref().unwrap();
            assert_eq!(scratch.width, reserved_width);
            let max_rows = match reserved_width {
                128 => 131_072,
                256 => 65_536,
                _ => 262_144,
            };
            assert_eq!(scratch.tile_rows, corpus.capacity_range().end.min(max_rows));
            assert!(scratch.scores.bytes() <= 64 * 1024 * 1024);
            assert_eq!(
                scratch.scores.bytes(),
                reserved_width * scratch.tile_rows * 4
            );
        }
        let queries: Vec<_> = (0..256).flat_map(axis).collect();
        let expected = db.search_batch(&queries, 33)?;
        let profiled = db.profile_search(&queries, 33, &[])?;
        assert_eq!(profiled.neighbors, expected);
        let profile = profiled.execution.unwrap();
        assert_eq!(profile.stages["copy"].commands, 4);
        let tile_rows = db.batch.as_ref().unwrap().tile_rows;
        for command in profile
            .commands
            .iter()
            .filter(|c| c.operation.starts_with("batch_scan"))
        {
            let rows =
                u32::from_le_bytes(command.constants.as_ref().unwrap()[..4].try_into().unwrap())
                    as usize;
            assert!(rows <= tile_rows);
            assert_eq!(
                command.grid.unwrap(),
                [(rows.div_ceil(64) * 4) as u32, 1, 1]
            );
            assert!(command.binding_bytes[4] <= 64 * 1024 * 1024);
        }
        // Device search reuses the same bounded tile policy after host/profile
        // replays and after both shrinking and growing corpus capacity.
        let input = db.stream().allocate(queries.len() * 4)?;
        db.stream().upload(
            input.binding(),
            &queries
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let mut output = DeviceNeighbors::new(db.stream(), 256, 33)?;
        db.search_device(
            DeviceQueries::new(input.binding(), 256, 3, 3)?,
            None,
            &mut output,
        )?;
        assert_eq!(output.read(db.stream())?, expected);
        assert_eq!(db.batch.as_ref().unwrap().tile_rows, tile_rows);
    }
    db.set_corpus(Corpus::build(&device, 3, std::iter::empty::<[f32; 3]>())?)?;
    let queries: Vec<_> = (0..256).flat_map(axis).collect();
    assert_eq!(db.search_batch(&queries, 33)?, vec![vec![]; 256]);
    assert!(db.profile_search(&queries, 33, &[])?.execution.is_none());
    assert_eq!(db.batch_workspace_bytes(), 0);
    let input = db.stream().allocate(queries.len() * 4)?;
    db.stream().upload(
        input.binding(),
        &queries
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    let mut output = DeviceNeighbors::new(db.stream(), 256, 33)?;
    db.search_device(
        DeviceQueries::new(input.binding(), 256, 3, 3)?,
        None,
        &mut output,
    )?;
    assert_eq!(output.read(db.stream())?, vec![vec![]; 256]);
    Ok(())
}

#[test]
#[ignore = "requires gfx1151"]
fn invalid_wide_host_queries_leave_output_and_workspace_untouched() -> Result<()> {
    let device = Device::open(0)?;
    let mut db = Searcher::build(&device, 3, (0..17).map(axis))?;
    let mut output = vec![vec![Neighbor {
        id: 42,
        similarity: 0.5,
    }]];
    let expected = output.clone();
    for bad in [0.0, f32::NAN, f32::INFINITY] {
        let mut queries: Vec<_> = (0..256).flat_map(axis).collect();
        queries[255 * 3..].fill(bad);
        assert!(db.search_batch_into(&queries, 5, &mut output).is_err());
        assert_eq!(output, expected);
        assert_eq!(db.batch_workspace_bytes(), 0);
    }
    assert!(db.search_batch(&vec![1.0; 257 * 3], 5).is_err());
    assert!(db.reserve_batch(257, 5).is_err());
    assert!(db.search_batch(&[], 5)?.is_empty());
    Ok(())
}
