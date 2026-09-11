//! Example application code, deliberately separate from hrxdb's search API.
use hrx::loom::{Compiler, CompilerOptions, Specialization};
use hrxdb::{Corpus, Error, Result};

/// Best cosine per (query row, album), with negative infinity for absent albums.
///
/// This example accepts up to 64 queries and 1,024 albums. It reads full 256-row
/// tiles once per output cell and masks slack before reduction. It illustrates
/// interoperability, not a tuned album search algorithm. Queries are row-major
/// FP32. Ordinals follow insertion IDs. Output stays on the supplied stream;
/// callers can select from it directly without reading the matrix back.
pub fn score_by_album(
    stream: &mut hrx::Stream,
    corpus: &Corpus,
    ordinals: &[u32],
    album_count: usize,
    queries: &[f32],
) -> Result<hrx::Buffer> {
    let invalid = || Error::Message("invalid album example input".into());
    let d = corpus.dimensions();
    let padded = corpus.padded_dimensions();
    if ordinals.len() != corpus.len()
        || !(1..=1024).contains(&album_count)
        || ordinals.iter().any(|&id| id as usize >= album_count)
        || !queries.len().is_multiple_of(d)
        || queries.len() / d > 64
    {
        return Err(invalid());
    }
    let count = queries.len() / d;
    let mut query_bytes = Vec::with_capacity(count * padded * 4);
    for query in queries.chunks_exact(d) {
        let norm = query
            .iter()
            .map(|&v| (v as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        if !norm.is_finite() || norm == 0.0 {
            return Err(invalid());
        }
        query_bytes.extend(
            query
                .iter()
                .flat_map(|&v| ((v as f64 / norm) as f32).to_le_bytes()),
        );
        query_bytes.resize(query_bytes.len() + (padded - d) * 4, 0);
    }
    let absent = vec![vec![f32::NEG_INFINITY; album_count]; count];
    let output = stream.allocate(count * album_count * 4)?;
    let output_bytes: Vec<_> = absent
        .iter()
        .flatten()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    stream.upload(output.binding(), &output_bytes)?;
    if count == 0 || corpus.is_empty() {
        return Ok(output);
    }

    // The stream, compiler, side arrays, query storage, and output belong to
    // the application. The corpus stays in its original allocations.
    let compiler = Compiler::with_options(
        None,
        CompilerOptions {
            target: stream.target().clone(),
            ..Default::default()
        },
    )?;
    // The global side array must also cover the last tile of every binding.
    // Interior slack may overlap later logical rows; the kernel masks by its
    // shard-local row count, independently of the ordinal's value.
    let ordinal_capacity = corpus
        .shards()
        .map(|shard| shard.row_range().start + shard.capacity_rows())
        .max()
        .unwrap();
    let album_buffer = stream.allocate(ordinal_capacity * 4)?;
    let query_buffer = stream.allocate(query_bytes.len())?;
    let mut album_bytes: Vec<_> = ordinals.iter().flat_map(|id| id.to_le_bytes()).collect();
    album_bytes.resize(ordinal_capacity * 4, 0xff); // Sentinel for trailing slack.
    stream.upload(album_buffer.binding(), &album_bytes)?;
    stream.upload(query_buffer.binding(), &query_bytes)?;

    for shard in corpus.shards() {
        let rows = shard.row_range();
        let mut spec = Specialization::new("album_scores");
        for (key, value) in [
            ("album.rows", rows.len()),
            ("album.capacity", shard.capacity_rows()),
            ("album.dimensions", padded),
            ("album.queries", count),
            ("album.count", album_count),
        ] {
            spec.config.insert(key.into(), value.to_string());
        }
        let artifact = compiler
            .module(include_str!("../album_scores.loom"))
            .compile(&spec)?;
        // SAFETY: trusted example source, specialized for this device.
        let kernel = unsafe { stream.load_artifact(&artifact)? };
        // SAFETY: all bindings have the specialized capacity/query/album extents.
        // The kernel only reads the borrowed vectors/norms. One workgroup owns
        // each output cell; shard dispatches on this stream execute in order.
        unsafe {
            stream.dispatch(
                &kernel,
                [album_count as u32, count as u32, 1],
                [256, 1, 1],
                &hrx::Constants::new(),
                &[
                    shard.vectors(),
                    shard.inverse_norms(),
                    album_buffer.try_slice(rows.start * 4, shard.capacity_rows() * 4)?,
                    query_buffer.binding(),
                    output.binding(),
                ],
            )?;
        }
    }
    Ok(output)
}

/// Host readback adapter used by the corpus interoperability tests.
#[cfg(test)]
#[allow(dead_code)]
pub fn best_by_album(
    corpus: &Corpus,
    ordinals: &[u32],
    album_count: usize,
    queries: &[f32],
) -> Result<Vec<Vec<f32>>> {
    let mut stream = corpus.stream()?;
    let output = score_by_album(&mut stream, corpus, ordinals, album_count, queries)?;
    let mut bytes = vec![0; queries.len() / corpus.dimensions() * album_count * 4];
    stream.read_blocking(output.binding(), &mut bytes)?;
    Ok(bytes
        .chunks_exact(album_count * 4)
        .map(|row| {
            row.as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect()
        })
        .collect())
}
