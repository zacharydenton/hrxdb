//! Shared dispatch recording for immediate device queries and reusable host searches.
use crate::CommandProfile;
use hrx::{Constants, Graph, GraphExec, Kernel, Node, Result, Stream, View};

pub(crate) enum Commands<'a> {
    Immediate(&'a Stream),
    Recorded {
        graph: Graph<'a>,
        last: Option<Node>,
        profile: Option<Vec<CommandProfile>>,
    },
}
impl<'a> Commands<'a> {
    pub fn immediate(stream: &'a Stream) -> Self {
        Self::Immediate(stream)
    }
    pub fn record(stream: &'a Stream) -> Result<Self> {
        Ok(Self::Recorded {
            graph: stream.graph()?,
            last: None,
            profile: None,
        })
    }
    pub fn profile(stream: &'a Stream) -> Result<Self> {
        Ok(Self::Recorded {
            graph: stream.graph()?,
            last: None,
            profile: Some(Vec::new()),
        })
    }
    // Safety: the caller validates binding sizes and kernel accesses. Recorded
    // commands form a serial dependency chain, preserving immediate ordering.
    pub unsafe fn dispatch(
        &mut self,
        kernel: &'a Kernel,
        grid: [u32; 3],
        block: [u32; 3],
        constants: &Constants,
        bindings: &[View<'a>],
    ) -> Result<()> {
        match self {
            // SAFETY: the caller supplies valid kernel accesses and bindings.
            Self::Immediate(stream) => unsafe {
                stream.dispatch(kernel, grid, block, constants, bindings)
            },
            Self::Recorded {
                graph,
                last,
                profile,
            } => {
                // SAFETY: the caller validates accesses; last orders all prior work.
                *last = Some(unsafe {
                    graph.dispatch(last.as_slice(), kernel, grid, block, constants, bindings)?
                });
                if let Some(rows) = profile {
                    rows.push(CommandProfile {
                        label: format!("{}.{}", rows.len(), kernel.symbol()),
                        operation: kernel.symbol().to_owned(),
                        grid: Some(grid),
                        block: Some(block),
                        binding_bytes: bindings.iter().map(|b| b.len()).collect(),
                        constants: Some(constants.as_bytes().to_vec()),
                        copy_bytes: None,
                    });
                }
                Ok(())
            }
        }
    }
    pub fn copy(&mut self, dst: View<'a>, src: View<'a>) -> Result<()> {
        match self {
            Self::Immediate(stream) => stream.copy(dst, src),
            Self::Recorded {
                graph,
                last,
                profile,
            } => {
                *last = Some(graph.copy(last.as_slice(), dst, src)?);
                if let Some(rows) = profile {
                    rows.push(CommandProfile {
                        label: format!("{}.copy", rows.len()),
                        operation: "copy".into(),
                        grid: None,
                        block: None,
                        binding_bytes: vec![dst.len(), src.len()],
                        constants: None,
                        copy_bytes: Some(src.len()),
                    });
                }
                Ok(())
            }
        }
    }
    pub fn finish(self) -> Result<GraphExec> {
        match self {
            Self::Recorded { graph, .. } => graph.finish(),
            Self::Immediate(_) => unreachable!("only recorded commands are finished"),
        }
    }
    pub fn finish_profiled(self) -> Result<(GraphExec, Vec<CommandProfile>)> {
        match self {
            Self::Recorded {
                graph,
                profile: Some(rows),
                ..
            } => {
                let labels = rows.iter().map(|r| r.label.clone()).collect::<Vec<_>>();
                Ok((graph.finish_profiled(&labels)?, rows))
            }
            _ => unreachable!("only profiling recordings are finished with markers"),
        }
    }
}
