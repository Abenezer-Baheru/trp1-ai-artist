"""
Google Imagen image provider - UPDATED 2026
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from ai_content.core.registry import ProviderRegistry
from ai_content.core.result import GenerationResult
from ai_content.core.exceptions import ProviderError, AuthenticationError
from ai_content.config import get_settings

logger = logging.getLogger(__name__)

@ProviderRegistry.register_image("imagen")
class GoogleImagenProvider:
    name = "imagen"

    def __init__(self):
        self.settings = get_settings().google
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from google import genai
                api_key = self.settings.api_key
                if not api_key:
                    raise AuthenticationError("imagen")
                self._client = genai.Client(api_key=api_key)
            except ImportError:
                raise ProviderError("imagen", "google-genai package not installed.")
        return self._client

    async def generate(
        self,
        prompt: str,
        *,
        aspect_ratio: str = "16:9",
        num_images: int = 1,
        output_path: str | None = None,
        use_gemini: bool = False,
    ) -> GenerationResult:
        from google.genai import types
        client = self._get_client()

        # FIX: Migration to 2026 Stable Models
        # 'gemini-3-pro-image-preview' is deprecated. Using stable Imagen 3.
        model = "imagen-3.0-generate-002" 

        logger.info(f"🖼️ Imagen: Generating image ({aspect_ratio}) via {model}")

        try:
            # FIX: Using the correct GenerateImagesConfig for 2026 SDK
            response = await client.aio.models.generate_images(
                model=model,
                prompt=prompt,
                config=types.GenerateImagesConfig(
                    number_of_images=num_images,
                    aspect_ratio=aspect_ratio,
                ),
            )

            if not response.generated_images:
                return GenerationResult(success=False, provider=self.name, content_type="image", error="No images generated")

            image_data = response.generated_images[0].image.image_bytes

            # File Handling
            if output_path:
                file_path = Path(output_path)
            else:
                output_dir = get_settings().output_dir
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                file_path = output_dir / f"imagen_{timestamp}.png"

            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(image_data)

            return GenerationResult(
                success=True,
                provider=self.name,
                content_type="image",
                file_path=file_path,
                metadata={"aspect_ratio": aspect_ratio, "model": model},
            )

        except Exception as e:
            logger.error(f"Imagen generation failed: {e}")
            return GenerationResult(success=False, provider=self.name, content_type="image", error=str(e))