//! Borrowed access to the resident corpus for application-owned GPU kernels.
use crate::FlatIndex;
use hrx::View;
use std::ops::Range;

/// A borrowed, zero-copy view of the vectors owned by a [`FlatIndex`].
///
/// Rows retain their insertion order across contiguous, nonoverlapping shards.
/// The stored representation is row-major little-endian FP16, with zero padding
/// to [`padded_dimensions`](Self::padded_dimensions). Each row has an FP32
/// inverse norm computed from its stored FP16 values. [`FlatIndex::build`]
/// normalizes before quantization; [`FlatIndex::build_fp16`] preserves the
/// supplied FP16 values. Neither accessor reconstructs the original FP32 input.
///
/// # Execution and access contract
///
/// Corpus uploads complete before construction returns. Bindings may be read on
/// a caller-owned HRX stream created from the same device used to build the
/// index. The caller owns kernel compilation, side arrays, output buffers, and
/// execution/completion; acquiring a view performs no allocation or GPU work.
/// This view does not expose or synchronize the index's search workspace.
///
/// Treat vector and inverse-norm bindings as **read-only**. HRX's [`View`] type
/// does not enforce this restriction: uploading, filling, or dispatching writes
/// to them is unsupported and invalidates the index's data invariants. The
/// ordinary safety requirements of [`hrx::Stream::dispatch`] still apply.
///
/// Bindings borrow the index. Rust lifetimes do not wait for asynchronous GPU
/// work; callers must manage completion using HRX. The custom-kernel example
/// completes its stream before returning its results.
///
/// A binding cannot outlive the index that owns it:
///
/// ```compile_fail
/// fn escape(index: hrxdb::FlatIndex) -> hrx::View<'static> {
///     index.corpus().shards().next().unwrap().vectors()
/// }
/// ```
#[derive(Clone, Copy)]
pub struct CorpusView<'a> {
    index: &'a FlatIndex,
}

/// Borrowed bindings for one contiguous range of corpus insertion IDs.
///
/// Both bindings start at local row zero. Use [`row_range`](Self::row_range)
/// to align application side arrays or convert local rows to global IDs.
/// Shard boundaries are storage details and may split application groups.
/// See [`CorpusView`] for the read-only and execution contract.
#[derive(Clone, Copy, Debug)]
pub struct CorpusShardView<'a> {
    vectors: View<'a>,
    inverse_norms: View<'a>,
    start: usize,
    count: usize,
}

impl FlatIndex {
    /// Borrow the resident FP16 corpus for custom GPU computations.
    ///
    /// This allocates no storage and performs no copy, compilation, or GPU work.
    /// See [`CorpusView`] for layout, normalization, and execution requirements.
    ///
    /// ```no_run
    /// use hrxdb::{Device, FlatIndex};
    /// # fn main() -> hrxdb::Result<()> {
    /// let device = Device::open(0)?;
    /// let index = FlatIndex::build(&device, 3, [[1.0, 2.0, 3.0]])?;
    /// let corpus = index.corpus();
    /// for shard in corpus.shards() {
    ///     let rows = shard.row_range();
    ///     assert_eq!(shard.vectors().len(), rows.len() * corpus.padded_dimensions() * 2);
    ///     assert_eq!(shard.inverse_norms().len(), rows.len() * 4);
    ///     // Bind these views to a custom kernel on a stream from `device`.
    ///     // Side arrays use the global insertion IDs in `rows`.
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn corpus(&self) -> CorpusView<'_> {
        CorpusView { index: self }
    }
}

impl<'a> CorpusView<'a> {
    /// Total number of indexed vectors across all shards.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Whether the corpus has no rows (and no exposed shards).
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Logical vector length supplied at construction.
    pub fn dimensions(&self) -> usize {
        self.index.dimensions()
    }

    /// Stored row stride in FP16 elements, including zero padding.
    ///
    /// Multiply by two for the row stride in bytes. Padding rounds the logical
    /// dimension up to a multiple of 128.
    pub fn padded_dimensions(&self) -> usize {
        self.index.padded_dimensions()
    }

    /// Iterate over nonempty shards in increasing insertion-ID order.
    ///
    /// The ranges partition `0..self.len()` without gaps. An empty corpus yields
    /// no shards. Iteration creates borrowed descriptors without allocating.
    /// Bindings borrow the index, so they can outlive this descriptor/iterator.
    pub fn shards(&self) -> impl ExactSizeIterator<Item = CorpusShardView<'a>> + 'a {
        let index = self.index;
        index.shards[..index.shard_count()]
            .iter()
            .map(move |shard| CorpusShardView {
                vectors: shard.data.binding(),
                inverse_norms: index
                    .norms
                    .try_slice(shard.start * 4, shard.count * 4)
                    .expect("corpus shard range lies within the inverse-norm allocation"),
                start: shard.start,
                count: shard.count,
            })
    }
}

impl<'a> CorpusShardView<'a> {
    /// Global insertion IDs covered by this shard, with an exclusive end.
    pub fn row_range(&self) -> Range<usize> {
        self.start..self.start + self.count
    }

    /// Read-only row-major FP16 vector binding, starting at local row zero.
    ///
    /// Its byte length is `row_range().len() * corpus.padded_dimensions() * 2`.
    /// Values are little-endian IEEE binary16; padding components are zero.
    /// HRX does not enforce read-only access; see [`CorpusView`]'s contract.
    pub fn vectors(&self) -> View<'a> {
        self.vectors
    }

    /// Read-only FP32 inverse-norm binding, starting at local row zero.
    ///
    /// Contains exactly `row_range().len()` little-endian FP32 values. For local
    /// row `r`, multiply its FP32 dot product with a unit query by element `r`
    /// to obtain cosine similarity over the stored vector. For an unnormalized
    /// query, also divide by the query's norm. Accumulation order can affect
    /// rounding relative to built-in search; scores are not clamped.
    /// HRX does not enforce read-only access; see [`CorpusView`]'s contract.
    pub fn inverse_norms(&self) -> View<'a> {
        self.inverse_norms
    }
}

#[cfg(test)]
#[path = "../examples/support/album_scores.rs"]
mod album_example;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Device, Result, ScanConfig, encode, encode_fp16};
    use half::f16;

    #[test]
    #[ignore = "requires gfx1151"]
    fn custom_kernel_reads_shards_and_merges_album_scores() -> Result<()> {
        let device = Device::open(0)?;
        for d in [3usize, 129, 769] {
            let padded = d.div_ceil(128) * 128;
            let rows: Vec<Vec<f32>> = (0..263)
                .map(|i| {
                    (0..d)
                        .map(|j| ((i * 37 + j * 13) % 103) as f32 - 49.25)
                        .collect()
                })
                .collect();
            let ordinals: Vec<u32> = (0..rows.len())
                .map(|i| ((i * 7 + i / 129) % 5) as u32)
                .collect();
            let queries: Vec<f32> = (0..3).flat_map(|q| rows[q * 131].iter().copied()).collect();
            for direct_fp16 in [false, true] {
                let mut index = if direct_fp16 {
                    let bytes: Vec<Vec<u8>> = rows
                        .iter()
                        .map(|row| {
                            row.iter()
                                .flat_map(|&v| f16::from_f32(v).to_le_bytes())
                                .collect()
                        })
                        .collect();
                    FlatIndex::build_encoded(
                        &device,
                        d,
                        &bytes,
                        ScanConfig::default(),
                        129 * padded,
                        |r, d, p, bytes| encode_fp16(r, d, p, bytes),
                    )?
                } else {
                    FlatIndex::build_encoded(
                        &device,
                        d,
                        &rows,
                        ScanConfig::default(),
                        129 * padded,
                        |r, d, p, bytes| encode(r, d, p, bytes),
                    )?
                };
                let before = index.search(&queries[..d], 5)?;
                let corpus = index.corpus();
                assert_eq!(corpus.len(), rows.len());
                assert!(!corpus.is_empty());
                assert_eq!(corpus.dimensions(), d);
                assert_eq!(corpus.padded_dimensions(), padded);
                assert_eq!(corpus.shards().len(), 3);
                assert_eq!(
                    corpus.shards().map(|s| s.row_range()).collect::<Vec<_>>(),
                    [0..129, 129..258, 258..263]
                );

                // Bindings remain usable after the temporary corpus/shard
                // descriptors disappear, and work on a separate caller stream.
                let binding = index.corpus().shards().next().unwrap().vectors();
                let mut reader = device.stream()?;
                reader.read_blocking(binding.slice(0, 2)?, &mut [0; 2])?;
                for shard in corpus.shards() {
                    let range = shard.row_range();
                    assert_eq!(shard.vectors().len(), range.len() * padded * 2);
                    assert_eq!(shard.inverse_norms().len(), range.len() * 4);
                    let mut vectors = vec![0; shard.vectors().len()];
                    let mut norms = vec![0; shard.inverse_norms().len()];
                    reader.read_blocking(shard.vectors(), &mut vectors)?;
                    reader.read_blocking(shard.inverse_norms(), &mut norms)?;
                    for (local, global) in range.enumerate() {
                        let source = &rows[global];
                        let length = source
                            .iter()
                            .map(|&v| (v as f64).powi(2))
                            .sum::<f64>()
                            .sqrt();
                        let stored = &vectors[local * padded * 2..(local + 1) * padded * 2];
                        let mut sum = 0.0;
                        for (j, bytes) in stored.as_chunks::<2>().0.iter().enumerate() {
                            let actual = f16::from_le_bytes(*bytes);
                            let expected = if j >= d {
                                f16::ZERO
                            } else if direct_fp16 {
                                f16::from_f32(source[j])
                            } else {
                                f16::from_f64(source[j] as f64 / length)
                            };
                            assert_eq!(actual.to_bits(), expected.to_bits());
                            sum += actual.to_f64().powi(2);
                        }
                        let inverse =
                            f32::from_le_bytes(norms[local * 4..local * 4 + 4].try_into().unwrap());
                        assert_eq!(inverse, (1.0 / sum.sqrt()) as f32);
                    }
                }

                // Album 5 is absent; all other groups cross shard boundaries.
                // Maxima must persist when later shards contain worse matches.
                let actual = album_example::best_by_album(&device, corpus, &ordinals, 6, &queries)?;
                for (q, query) in queries.chunks_exact(d).enumerate() {
                    let mut expected = [f32::NEG_INFINITY; 6];
                    for (score, &album) in index.scores(query)?.iter().zip(&ordinals) {
                        expected[album as usize] = expected[album as usize].max(*score);
                    }
                    for (&got, &want) in actual[q].iter().zip(&expected) {
                        assert!(
                            got == want || (got - want).abs() < 3e-6,
                            "d={d} direct_fp16={direct_fp16} query={q}: {got} vs {want}"
                        );
                    }
                }
                assert_eq!(index.search(&queries[..d], 5)?, before);
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires gfx1151"]
    fn empty_corpus_exposes_no_shards() -> Result<()> {
        let device = Device::open(0)?;
        let index = FlatIndex::build(&device, 3, std::iter::empty::<[f32; 3]>())?;
        let corpus = index.corpus();
        assert_eq!(corpus.len(), 0);
        assert!(corpus.is_empty());
        assert_eq!(corpus.dimensions(), 3);
        assert_eq!(corpus.padded_dimensions(), 128);
        assert_eq!(corpus.shards().len(), 0);
        assert!(corpus.shards().next().is_none());
        assert_eq!(
            album_example::best_by_album(&device, corpus, &[], 2, &[1.0, 0.0, 0.0])?,
            vec![vec![f32::NEG_INFINITY; 2]]
        );
        Ok(())
    }
}
