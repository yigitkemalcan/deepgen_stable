"""Transformer adapter for swapping attention processors at runtime."""

from diffusers.models.attention_processor import (
    Attention,
    FluxAttnProcessor2_0,
    FluxSingleAttnProcessor2_0,
)

from .processors.adaptive_flux import (
    AdaptiveFluxAttnProcessor2_0,
    AdaptiveFluxSingleAttnProcessor2_0,
)


class TransformerAdapter:
    """Swaps original Flux attention processors for adaptive (soft-blend) versions.

    Usage:
        adapter = TransformerAdapter(pipe.transformer)
        adapter.install()        # swap processors
        adapter.set_alpha(0.8)   # set injection strength
        ...                      # run pipeline
        adapter.uninstall()      # restore originals
    """

    def __init__(self, transformer):
        self.transformer = transformer
        self._original_processors: dict = {}

    def install(self):
        """Replace original processors with adaptive versions."""
        for name, module in self.transformer.named_modules():
            if isinstance(module, Attention):
                proc = module.get_processor()
                if isinstance(proc, FluxAttnProcessor2_0):
                    self._original_processors[name] = proc
                    module.set_processor(AdaptiveFluxAttnProcessor2_0())
                elif isinstance(proc, FluxSingleAttnProcessor2_0):
                    self._original_processors[name] = proc
                    module.set_processor(AdaptiveFluxSingleAttnProcessor2_0())

    def set_alpha(self, alpha: float):
        """Set injection_alpha on all adapted Attention modules."""
        for name, module in self.transformer.named_modules():
            if isinstance(module, Attention) and name in self._original_processors:
                module._injection_alpha = alpha

    def uninstall(self):
        """Restore original processors and clean up attributes."""
        for name, module in self.transformer.named_modules():
            if name in self._original_processors:
                module.set_processor(self._original_processors[name])
                if hasattr(module, '_injection_alpha'):
                    delattr(module, '_injection_alpha')
        self._original_processors.clear()
