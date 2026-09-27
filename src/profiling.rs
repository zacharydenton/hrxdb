//! Explicit diagnostic replays using the same commands as ordinary search.
use crate::*;
use std::collections::BTreeMap;

/// One command, in the same order as the device profile's intervals.
#[derive(Clone, Debug, Serialize)]
pub struct CommandProfile {
    /// Unique label joining this command to its timestamp interval.
    pub label: String,
    /// Kernel entry symbol, or `copy` for a buffer transfer.
    pub operation: String,
    /// Dispatch workgroup counts; absent for transfers.
    pub grid: Option<[u32; 3]>,
    /// Dispatch threads per workgroup; absent for transfers.
    pub block: Option<[u32; 3]>,
    /// Binding extents in argument order, without process-specific addresses.
    pub binding_bytes: Vec<usize>,
    /// Packed scalar arguments in declaration order; absent for transfers.
    /// Decode using the authored kernel's argument types, not inferred widths.
    pub constants: Option<Vec<u8>>,
    /// Transfer size; absent for dispatches.
    pub copy_bytes: Option<usize>,
}

/// Instrumented intervals grouped by kernel symbol or transfer operation.
#[derive(Clone, Debug, Default, Serialize)]
pub struct StageProfile {
    /// Number of instrumented commands.
    pub commands: usize,
    /// Sum of the command intervals, including profiling perturbation.
    pub device_ms: f64,
}

/// Completed, instrumented GPU replay. This is separate from ordinary timing.
#[derive(Clone, Debug, Serialize)]
pub struct ExecutionProfile {
    /// Host time building the diagnostic graph, including timestamp storage.
    pub preparation_ms: f64,
    /// Host interval enclosing this replay, synchronization and timestamp harvest.
    /// Excludes query encoding, graph preparation and neighbor decoding.
    pub replay_host_ms: f64,
    /// Raw ticks, frequency, device/queue identity, interval union, span and gaps.
    pub device: hrx::fabric::DeviceProfile,
    /// Commands joined to the device intervals by their unique labels.
    pub commands: Vec<CommandProfile>,
    /// Per-operation sums from this replay; these are not hardware utilization.
    pub stages: BTreeMap<String, StageProfile>,
}

/// Search results and an optional diagnostic replay, retaining query order.
#[derive(Clone, Debug, Serialize)]
pub struct ProfiledSearch {
    /// Results of the instrumented invocation.
    pub neighbors: Vec<Vec<Neighbor>>,
    /// Absent when the empty batch/corpus or exclusions require no GPU work.
    pub execution: Option<ExecutionProfile>,
}

impl Searcher {
    /// Execute a diagnostic search with GPU timestamps around every command.
    ///
    /// Accepts 0–64 row-major queries and shared exclusions, with the same
    /// validation and results as [`Self::search_batch_excluding`]. One query
    /// uses the single-query path. Scoring, masking, selection, merging and
    /// transfers all use the ordinary search implementation.
    ///
    /// Markers introduce completion barriers and perturb execution. Collect
    /// this separately from ordinary latency samples; do not subtract its GPU
    /// intervals from an ordinary search's host time. No profiling resources
    /// are retained after the call. Normal searches remain uninstrumented.
    ///
    /// # Errors
    /// Invalid inputs, allocation/compilation failures and unavailable native
    /// timestamp support return errors. Profiling never silently falls back to
    /// host timers. Timestamp storage is native diagnostic overhead outside the
    /// searcher's workspace budget.
    pub fn profile_search(
        &mut self,
        queries: &[f32],
        k: usize,
        excluded: &[u32],
    ) -> Result<ProfiledSearch> {
        let mut neighbors = Vec::new();
        let mut execution = None;
        self.search_batch_impl(queries, k, excluded, &mut neighbors, Some(&mut execution))?;
        Ok(ProfiledSearch {
            neighbors,
            execution,
        })
    }
}

pub(crate) fn replay(
    stream: &mut Stream,
    mut graph: hrx::GraphExec,
    commands: Vec<CommandProfile>,
    preparation_ms: f64,
) -> Result<ExecutionProfile> {
    let start = Instant::now();
    let device = stream.launch_profiled(&mut graph)?;
    let replay_host_ms = start.elapsed().as_secs_f64() * 1000.0;
    if device.intervals.len() != commands.len() {
        return Err(invalid("GPU profile command count mismatch"));
    }
    let mut stages = BTreeMap::<String, StageProfile>::new();
    for (command, interval) in commands.iter().zip(&device.intervals) {
        if command.label != interval.label {
            return Err(invalid("GPU profile label mismatch"));
        }
        let stage = stages.entry(command.operation.clone()).or_default();
        stage.commands += 1;
        stage.device_ms +=
            (interval.end_tick - interval.start_tick) as f64 * 1000.0 / device.frequency_hz as f64;
    }
    Ok(ExecutionProfile {
        preparation_ms,
        replay_host_ms,
        device,
        commands,
        stages,
    })
}
