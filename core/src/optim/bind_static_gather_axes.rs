use crate::internal::*;
use crate::ops::array::Gather;
use crate::optim::OptimizerSession;
use std::cell::RefCell;

/// Binds symbolic Gather output axes made concrete by constant folding its indices.
#[derive(Debug, Clone, Default)]
pub struct BindStaticGatherAxes;

impl super::TypedPass for BindStaticGatherAxes {
    fn reset(&mut self) -> TractResult<()> {
        Ok(())
    }

    fn next(
        &mut self,
        _session: &mut OptimizerSession,
        _model: &TypedModel,
    ) -> TractResult<Option<TypedModelPatch>> {
        Ok(None)
    }

    fn run_direct(&mut self, model: &mut TypedModel) -> TractResult<bool> {
        let bindings: RefCell<HashMap<Symbol, TDim>> = Default::default();
        Rewriter::default()
            .with_rule_for("bind-static-gather-axes", collect_bindings)
            .rewrite(&bindings, model)?;
        let bindings = bindings.into_inner();
        if bindings.is_empty() {
            return Ok(false);
        }
        *model = model.set_symbols(&bindings)?;
        Ok(true)
    }
}

fn collect_bindings(
    bindings: &RefCell<HashMap<Symbol, TDim>>,
    model: &TypedModel,
    node: &TypedNode,
    _name: &str,
    op: &Gather,
) -> TractResult<Option<TypedModelPatch>> {
    let facts = model.node_input_facts(node.id)?;
    let inferred = op.output_facts(&facts)?;
    for (old, new) in node.outputs[0].fact.shape.iter().zip(inferred[0].shape.iter()) {
        if let TDim::Sym(symbol) = old
            && new.as_usize().is_some()
            && let Some(previous) = bindings.borrow_mut().insert(symbol.clone(), new.clone())
        {
            ensure!(
                previous == *new,
                "Conflicting fixed dimension for {symbol}: {previous} and {new}"
            );
        }
    }
    Ok(None)
}
