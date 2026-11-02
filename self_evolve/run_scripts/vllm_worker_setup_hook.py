"""Ray worker hook: disable broken flash-attn ViT path in vLLM on this cluster."""


def setup_worker() -> None:
    try:
        import vllm.attention.layer as layer

        layer.check_upstream_fa_availability = lambda dtype: False
    except Exception:
        pass
