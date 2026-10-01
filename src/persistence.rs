//! Versioned, little-endian native corpus storage. Payloads are copied verbatim.
use crate::*;
use hrx::inference::ModelContext;
use std::{
    fs::File,
    io::{Read, Write},
    path::Path,
};

const MAGIC: &[u8; 8] = b"HRXDB\0\0\x01";
const HEADER_BYTES: u64 = 40;

fn io(error: std::io::Error) -> Error {
    Error::Message(format!("corpus file: {error}"))
}

struct Header {
    dimensions: usize,
    padded: usize,
    count: usize,
    shards: Vec<(usize, usize)>,
}

impl Header {
    fn read(file: &mut File) -> Result<Self> {
        let length = file.metadata().map_err(io)?.len();
        let mut magic = [0; 8];
        file.read_exact(&mut magic).map_err(io)?;
        if &magic != MAGIC {
            return Err(invalid("unsupported corpus file format"));
        }
        let mut word = || -> Result<usize> {
            let mut bytes = [0; 8];
            file.read_exact(&mut bytes).map_err(io)?;
            usize::try_from(u64::from_le_bytes(bytes))
                .map_err(|_| invalid("corpus file size overflow"))
        };
        let dimensions = word()?;
        let padded = word()?;
        let count = word()?;
        let shard_count = word()?;
        if padded != layout(dimensions, count)?.0
            || shard_count > count
            || (count == 0) != (shard_count == 0)
        {
            return Err(invalid("invalid corpus file layout"));
        }
        let expected =
            HEADER_BYTES + shard_count as u64 * 16 + count as u64 * (padded as u64 * 2 + 4);
        if length != expected {
            return Err(invalid("corpus file length does not match its layout"));
        }
        let mut shards = Vec::new();
        let mut total = 0usize;
        for _ in 0..shard_count {
            let rows = word()?;
            let capacity = word()?;
            if rows == 0
                || rows > capacity
                || !capacity.is_multiple_of(SHARD_ROW_ALIGNMENT)
                || capacity > shard_rows(padded, MAX_ELEMENTS)
            {
                return Err(invalid("invalid corpus file shard"));
            }
            total = total
                .checked_add(rows)
                .filter(|&n| n <= count)
                .ok_or_else(|| invalid("corpus file shard count overflow"))?;
            shards.push((rows, capacity));
        }
        if total != count {
            return Err(invalid("corpus file shard rows do not match its count"));
        }
        Ok(Self {
            dimensions,
            padded,
            count,
            shards,
        })
    }
}

impl Corpus {
    /// Write padded FP16 rows, FP32 inverse norms and shard sizes to a file.
    /// Preserves row order and exact stored values, including after updates.
    /// Unused row capacity is recorded but its contents are not saved.
    ///
    /// Replaces the contents of `path`. For atomic publication, save to a
    /// temporary path and rename after success. The completed file is synced.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut file = File::create(path).map_err(io)?;
        file.write_all(MAGIC).map_err(io)?;
        for value in [
            self.dimensions(),
            self.padded_dimensions(),
            self.len(),
            self.inner.shards.len(),
        ] {
            file.write_all(&(value as u64).to_le_bytes()).map_err(io)?;
        }
        for shard in &self.inner.shards {
            for value in [shard.count, shard.capacity] {
                file.write_all(&(value as u64).to_le_bytes()).map_err(io)?;
            }
        }
        let mut stream = self.stream()?;
        let bytes = CHUNK_BYTES.min(self.len() * self.padded_dimensions() * 2);
        let _staging = self
            .workspace_budget
            .as_ref()
            .map(|b| b.reserve(bytes))
            .transpose()?;
        let mut chunk = vec![0; bytes];
        for shard in &self.inner.shards {
            for view in [
                shard.vectors(self.padded_dimensions(), 0, shard.count)?,
                shard.inverse_norms(0, shard.count)?,
            ] {
                for at in (0..view.len()).step_by(CHUNK_BYTES) {
                    let n = chunk.len().min(view.len() - at);
                    stream.read_blocking(view.slice(at, n)?, &mut chunk[..n])?;
                    file.write_all(&chunk[..n]).map_err(io)?;
                }
            }
        }
        file.sync_all().map_err(io)
    }

    /// Load a file written by [`Self::save`] in the caller's retained context.
    /// Native buffers and bounded upload staging use the context's budget.
    /// Reads 4 MiB chunks while the preceding upload runs, with no per-element
    /// conversion, padding or norm computation. Completes uploads before return.
    ///
    /// Validates the version, lengths and shard layout. Payload values are
    /// trusted as saved; this does not validate individual floats or norms.
    pub fn load_in(context: &ModelContext, path: impl AsRef<Path>) -> Result<Self> {
        let mut file = File::open(path).map_err(io)?;
        let header = Header::read(&mut file)?;
        let device = Device::open(context.runtime().gpu()?.index())?;
        if device.target().as_str() != "gfx1151" {
            return Err(invalid("hrxdb currently targets gfx1151"));
        }
        let budget = context.runtime().memory_budget();
        // Keep destination allocations alive until the stream drains, including
        // early returns from I/O errors after a transfer was submitted.
        let mut shards = Vec::new();
        let mut stream = device.stream()?;
        if let Some(budget) = budget {
            stream = stream.with_memory_budget(budget.clone());
        }
        let bytes = CHUNK_BYTES.min(header.count * header.padded * 2);
        let _staging = budget.map(|b| b.reserve(bytes)).transpose()?;
        let mut chunk = vec![0; bytes];
        let mut start = 0;
        let loaded = (|| -> Result<()> {
            for (count, capacity) in header.shards {
                let storage = Allocation::new(&stream, header.padded, capacity, count, false)?;
                shards.push(Shard {
                    storage,
                    offset: 0,
                    start,
                    count,
                    capacity,
                });
                let shard = shards.last().unwrap();
                for view in [
                    shard.vectors(header.padded, 0, count)?,
                    shard.inverse_norms(0, count)?,
                ] {
                    for at in (0..view.len()).step_by(CHUNK_BYTES) {
                        let n = chunk.len().min(view.len() - at);
                        // upload owns its staging copy, so the disk read can
                        // overwrite this host buffer while that copy runs.
                        file.read_exact(&mut chunk[..n]).map_err(io)?;
                        stream.synchronize()?;
                        stream.upload(view.slice(at, n)?, &chunk[..n])?;
                    }
                }
                start += count;
            }
            Ok(())
        })();
        if let Err(error) = stream.synchronize() {
            // Match mutation's lifetime policy when completion is uncertain.
            std::mem::forget(shards);
            return Err(error);
        }
        loaded?;
        Ok(Self {
            workspace_budget: budget.cloned(),
            inner: Arc::new(CorpusStorage {
                context: Some(crate::corpus::CorpusContext::shared(context)),
                device,
                device_id: stream.device_id(),
                shards,
                dimensions: header.dimensions,
                padded: header.padded,
                count: header.count,
                gather: Default::default(),
                writer: Default::default(),
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_malformed_layouts_before_upload() {
        let path = std::env::temp_dir().join(format!("hrxdb-header-{}", std::process::id()));
        let mut valid = MAGIC.to_vec();
        for value in [3u64, 128, 1, 1, 1, 256] {
            valid.extend_from_slice(&value.to_le_bytes());
        }
        valid.resize(56 + 260, 0);
        let parse = |bytes: &[u8]| {
            std::fs::write(&path, bytes).unwrap();
            Header::read(&mut File::open(&path).unwrap())
        };
        assert!(parse(&valid).is_ok());
        for (offset, value) in [
            (0, 0),
            (8, 0),
            (8, 16_385),
            (16, 256),
            (24, (MAX_ROWS + 1) as u64),
            (32, 0),
            (32, 2),
            (40, 0),
            (40, 2),
            (48, 0),
            (48, 1),
            (48, MAX_ELEMENTS as u64),
        ] {
            let mut bad = valid.clone();
            bad[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
            assert!(
                parse(&bad).is_err(),
                "accepted offset {offset}, value {value}"
            );
        }
        for len in [0, 7, 39, 55, valid.len() - 1] {
            assert!(parse(&valid[..len]).is_err());
        }
        valid.push(0);
        assert!(parse(&valid).is_err());
        let mut empty = MAGIC.to_vec();
        for value in [3u64, 128, 0, 0] {
            empty.extend_from_slice(&value.to_le_bytes());
        }
        assert!(parse(&empty).is_ok());
        std::fs::remove_file(path).unwrap();
    }
}
