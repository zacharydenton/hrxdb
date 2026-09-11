//! Shared selection dispatch for built-in scans and application score matrices.
use crate::*;
use hrx::View;
pub(crate) struct SelectionPlan {
    pub first: Kernel,
    pub merge: Kernel,
    pub sort: Kernel,
    pub sorted_merge: Kernel,
}
impl SelectionPlan {
    pub fn new(
        compiler: &hrx::loom::Compiler,
        stream: &Stream,
        batch: usize,
        limit: usize,
    ) -> Result<(Self, Vec<Compilation>)> {
        let mut reports = Vec::new();
        let mut build = |source, mut spec: hrx::loom::Specialization| {
            spec.config.insert("db.batch".into(), batch.to_string());
            spec.config
                .insert("db.select.limit".into(), limit.to_string());
            let (kernel, report) = compile(compiler, stream, source, spec)?;
            reports.push(report);
            Ok::<_, Error>(kernel)
        };
        let plan = Self {
            first: build(kernels::SELECT, kernels::select_spec(true))?,
            merge: build(kernels::SELECT, kernels::select_spec(false))?,
            sort: build(kernels::SORT, kernels::named_spec("sort_select"))?,
            sorted_merge: build(kernels::SORT, kernels::named_spec("sorted_merge"))?,
        };
        Ok((plan, reports))
    }
}
pub(crate) struct SelectionInput<'a> {
    pub scores: View<'a>,
    pub rows: usize,
    pub start: usize,
    pub k: usize,
    pub batch: usize,
}
pub(crate) fn select_scores(
    stream: &Stream,
    candidates: &[(Buffer, Buffer); 2],
    plan: &SelectionPlan,
    input: SelectionInput<'_>,
) -> Result<usize> {
    let SelectionInput {
        scores,
        rows,
        start,
        k,
        batch,
    } = input;
    let mut count = rows;
    let mut output = 0;
    let mut first = true;
    loop {
        let groups = count.div_ceil(1024);
        let mut constants = Constants::new();
        constants.push(count as u32)?;
        constants.push(k as u32)?;
        constants.push(if first { start as u32 } else { 0 })?;
        let (input_scores, input_ids) = if first {
            (scores, scores)
        } else {
            (
                candidates[1 - output].0.binding(),
                candidates[1 - output].1.binding(),
            )
        };
        let (out_scores, out_ids) = &candidates[output];
        // SAFETY: per-query selection uses packed count/ceil(count/1024)*k
        // strides, all within the reserved width * tile capacity. Input and
        // output never alias. Only the first pass assigns global insertion IDs.
        unsafe {
            if k > 32 {
                stream.dispatch(
                    &plan.sort,
                    [groups as u32, batch as u32, 1],
                    [256, 1, 1],
                    &constants,
                    &[input_scores, out_scores.binding(), out_ids.binding()],
                )?;
            } else {
                stream.dispatch(
                    if first { &plan.first } else { &plan.merge },
                    [groups as u32, batch as u32, 1],
                    [256, 1, 1],
                    &constants,
                    &[
                        input_scores,
                        input_ids,
                        out_scores.binding(),
                        out_ids.binding(),
                    ],
                )?;
            }
        }
        if k > 32 {
            let mut lists = groups;
            while lists > 1 {
                let next = 1 - output;
                let mut constants = Constants::new();
                constants.push(lists as u32)?;
                constants.push(k as u32)?;
                // SAFETY: each query merges its own adjacent sorted k-lists;
                // the number of lists halves, and outputs use the other pair.
                unsafe {
                    stream.dispatch(
                        &plan.sorted_merge,
                        [lists.div_ceil(2) as u32, batch as u32, 1],
                        [256, 1, 1],
                        &constants,
                        &[
                            candidates[output].0.binding(),
                            candidates[output].1.binding(),
                            candidates[next].0.binding(),
                            candidates[next].1.binding(),
                        ],
                    )?;
                }
                lists = lists.div_ceil(2);
                output = next;
            }
            return Ok(output);
        }
        if groups == 1 {
            return Ok(output);
        }
        count = groups * k;
        output = 1 - output;
        first = false;
    }
}
