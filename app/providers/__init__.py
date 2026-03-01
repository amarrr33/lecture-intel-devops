import os
import logging
from app.providers.base_provider import BaseProvider

logger = logging.getLogger(__name__)


def get_provider() -> BaseProvider:
    """Factory to get the configured provider.

    Selection logic:
    - GEN_PROVIDER=gemini or auto (default): Use Gemini if API key is set.
    - GEN_PROVIDER=t5: Use T5 (offline, limited quality).
    - If Gemini is selected but API key is missing, raises an error
      instead of silently falling back to a broken T5 provider.
    """
    provider_name = os.getenv("GEN_PROVIDER", "auto").lower()
    has_key = bool(os.getenv("GEMINI_API_KEY"))

    if provider_name == "t5":
        logger.info("Using T5 provider (offline mode, limited quality).")
        from app.providers.t5_provider import T5Provider
        return T5Provider()

    # Gemini or auto
    if has_key:
        from app.providers.gemini_provider import GeminiProvider
        logger.info("Using Gemini provider.")
        return GeminiProvider()

    # No API key and not explicitly T5
    if provider_name == "gemini":
        raise RuntimeError(
            "GEN_PROVIDER=gemini but GEMINI_API_KEY is not set. "
            "Set the API key or use GEN_PROVIDER=t5 for offline mode."
        )

    # auto mode without key: warn clearly, fall back to T5
    logger.warning(
        "GEMINI_API_KEY not set. Falling back to T5 provider. "
        "Output quality will be very limited. Set GEMINI_API_KEY for production use."
    )
    from app.providers.t5_provider import T5Provider
    return T5Provider()
