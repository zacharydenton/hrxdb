//! Owned, coordinated device queries and bounded result slots.
use crate::{Corpus, DeviceNeighbors, DeviceQueries, Neighbor, Result, invalid};
use hrx::{
    Access,
    execution::GpuAccess,
    inference::{Inference, InferenceGraph, ModelContext, PreparedModel},
    tensor::{DType, DeviceTensor, Layout, TensorDesc},
};

/// A fixed query shape with shared corpus storage and a bounded worker count.
pub struct PreparedSearch {
    plan: PreparedModel,
}
impl Corpus {
    /// Prepare shared-context query normalization, exhaustive scan and top-k.
    /// Each slot owns private search workspace. `workers=1` is the conservative
    /// bandwidth-bound default; increasing it never duplicates the corpus.
    /// A context-built corpus requires the same runtime (context clones work).
    /// Device-built corpora require only the same native GPU.
    pub fn prepare_search(
        &self,
        context: &ModelContext,
        batch: usize,
        k: usize,
        workers: usize,
    ) -> Result<PreparedSearch> {
        if self
            .context()
            .is_some_and(|owner| !owner.runtime().same_domain(context.runtime()))
        {
            return Err(invalid("corpus belongs to another runtime"));
        }
        if batch == 0 || batch > crate::MAX_BATCH || !(1..=crate::MAX_K).contains(&k) {
            return Err(invalid("invalid coordinated search shape"));
        }
        if self.inner.device_id
            != hrx::Device::open(context.runtime().gpu()?.index())?
                .stream()?
                .device_id()
        {
            return Err(invalid("corpus belongs to another device"));
        }
        let dimensions = self.dimensions();
        let input =
            TensorDesc::new(DType::F32, vec![batch, dimensions])?.with_layout(Layout::Rows)?;
        let output = [
            TensorDesc::new(DType::U32, vec![batch])?,
            TensorDesc::new(DType::U32, vec![batch])?,
            TensorDesc::new(DType::F32, vec![batch, k])?.with_layout(Layout::Rows)?,
            TensorDesc::new(DType::U32, vec![batch, k])?.with_layout(Layout::Rows)?,
        ];
        let plan = PreparedModel::prepare(context, workers, |context| {
            let inputs = vec![context.allocate(input.clone())?];
            let outputs = output
                .iter()
                .cloned()
                .map(|desc| context.allocate(desc))
                .collect::<hrx::Result<Vec<_>>>()?;
            let mut corpus = self.clone();
            if let Some(budget) = context.runtime().memory_budget() {
                corpus = corpus.with_workspace_budget(budget.clone());
            }
            let mut searcher = corpus.searcher()?;
            searcher.reserve_device(batch, k)?;
            let mut results = DeviceNeighbors::new(&searcher.stream, batch, k)?;
            searcher.stream.synchronize()?;
            let bindings = std::iter::once(GpuAccess {
                view: inputs[0].binding().unwrap(),
                access: Access::Read,
            })
            .chain(outputs.iter().map(|tensor| GpuAccess {
                view: tensor.binding().unwrap(),
                access: Access::Write,
            }))
            .collect::<Vec<_>>();
            let mut graph = context.runtime().graph();
            // SAFETY: The closure owns its private searcher, result allocation
            // and corpus clone. Only input is read; all four declared outputs
            // are fully copied. Every stream operation completes before return.
            unsafe {
                graph.gpu_scoped(&bindings, move |views| {
                    let queries = DeviceQueries::new(views[0], batch, dimensions, dimensions)?;
                    searcher.search_device(queries, None, &mut results)?;
                    for (destination, source) in views[1..].iter().zip([
                        results.status(),
                        results.counts(),
                        results.scores(),
                        results.ids(),
                    ]) {
                        searcher.stream.copy(*destination, source)?;
                    }
                    searcher.stream.synchronize()
                })?;
            }
            Ok(InferenceGraph {
                inputs,
                outputs,
                graph: graph.prepare()?,
            })
        })?;
        Ok(PreparedSearch { plan })
    }
}
impl PreparedSearch {
    /// Submit resident contiguous f32 rows after their producer, without host waits.
    pub fn submit(&self, queries: &DeviceTensor) -> Result<SearchInference> {
        Ok(SearchInference {
            inference: self.plan.submit(std::slice::from_ref(queries))?,
        })
    }
    /// Whether every worker and exported output is idle, for budgeted eviction.
    pub fn is_idle(&self) -> bool {
        self.plan.is_idle()
    }
}
/// Resident search results; only each row's `counts` entries are valid.
pub struct SearchInference {
    inference: Inference,
}
impl SearchInference {
    /// Row status: zero is success; one means a non-finite or zero-norm query.
    pub fn status(&self) -> &DeviceTensor {
        &self.inference.outputs()[0]
    }
    /// Valid neighbor counts, one u32 per query.
    pub fn counts(&self) -> &DeviceTensor {
        &self.inference.outputs()[1]
    }
    /// Resident row-major f32 cosine scores.
    pub fn scores(&self) -> &DeviceTensor {
        &self.inference.outputs()[2]
    }
    /// Resident row-major u32 insertion IDs.
    pub fn ids(&self) -> &DeviceTensor {
        &self.inference.outputs()[3]
    }
    /// Read results at an explicit host boundary. Invalid rows are errors.
    pub fn read(self) -> Result<Vec<Vec<Neighbor>>> {
        let batch = self.status().desc().elements();
        let k = self.ids().desc().shape()[1];
        let values = self.inference.download()?.wait()?;
        let word = |output: usize, i: usize| -> [u8; 4] {
            values[output][i * 4..i * 4 + 4].try_into().unwrap()
        };
        let mut rows = Vec::with_capacity(batch);
        for row in 0..batch {
            if u32::from_le_bytes(word(0, row)) != 0 {
                return Err(invalid("device query is nonfinite or has zero norm"));
            }
            let count = u32::from_le_bytes(word(1, row)) as usize;
            if count > k {
                return Err(invalid("invalid device result count"));
            }
            rows.push(
                (0..count)
                    .map(|column| Neighbor {
                        id: u32::from_le_bytes(word(3, row * k + column)),
                        similarity: f32::from_le_bytes(word(2, row * k + column)),
                    })
                    .collect(),
            );
        }
        Ok(rows)
    }
}
