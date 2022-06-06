from pydgn.training.callback.engine_callback import EngineCallback


class iCGMMEngineCallback(EngineCallback):

    def on_forward(self, state):
        outputs = state.model.forward(state.batch_input, state.id_batch)
        state.update(batch_outputs=outputs)
