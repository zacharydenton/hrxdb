//! Shared dispatch recording for immediate device queries and reusable host searches.
use hrx::{Constants, Graph, GraphExec, Kernel, Node, Result, Stream, View};

pub(crate) enum Commands<'a> {
    Immediate(&'a Stream),
    Recorded {
        graph: Graph<'a>,
        last: Option<Node>,
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
            Self::Recorded { graph, last } => {
                // SAFETY: the caller validates accesses; last orders all prior work.
                *last = Some(unsafe {
                    graph.dispatch(last.as_slice(), kernel, grid, block, constants, bindings)?
                });
                Ok(())
            }
        }
    }
    pub fn copy(&mut self, dst: View<'a>, src: View<'a>) -> Result<()> {
        match self {
            Self::Immediate(stream) => stream.copy(dst, src),
            Self::Recorded { graph, last } => {
                *last = Some(graph.copy(last.as_slice(), dst, src)?);
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
}
