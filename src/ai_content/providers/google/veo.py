"""
Google Veo video provider.
Updated for 2026 SDK compatibility.
"""

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

from ai_content.core.registry import ProviderRegistry
from ai_content.core.result import GenerationResult
from ai_content.core.exceptions import (
    ProviderError,
    AuthenticationError,
)
from ai_content.config import get_settings

logger = logging.getLogger(__name__)

@ProviderRegistry.register_video("veo")
class GoogleVeoProvider:
    name = "veo"
    supports_image_to_video = True
    max_duration_seconds = 8

    def __init__(self):
        self.settings = get_settings().google
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from google import genai
                api_key = self.settings.api_key
                if not api_key:
                    raise AuthenticationError("veo")
                self._client = genai.Client(api_key=api_key)
            except ImportError:
                raise ProviderError("veo", "google-genai package not installed.")
        return self._client

    async def generate(
        self,
        prompt: str,
        *,
        aspect_ratio: str = "16:9",
        duration_seconds: int = 5,
        first_frame_url: str | None = None,
        output_path: str | None = None,
        use_fast_model: bool = False,
        person_generation: str = "allow_adult",
    ) -> GenerationResult:
        from google.genai import types

        client = self._get_client()

        # FIX: Ensure we use a stable 2026 model ID
        model = "veo-3.1-generate-001" 

        logger.info(f"🎬 Veo: Generating video ({aspect_ratio})")

        try:
            # FIX: Updated to GenerateVideosConfig (plural) for 2026 SDK
            config = types.GenerateVideosConfig(
                aspect_ratio=aspect_ratio,
                person_generation=person_generation,
            )

            # Generate
            if first_frame_url:
                image_data = await self._fetch_image(first_frame_url)
                image = types.Image(image_bytes=image_data)
                # FIX: Updated to generate_videos (plural)
                operation = await client.aio.models.generate_videos(
                    model=model,
                    prompt=prompt,
                    image=image,
                    config=config,
                )
            else:
                # FIX: Updated to generate_videos (plural)
                operation = await client.aio.models.generate_videos(
                    model=model,
                    prompt=prompt,
                    config=config,
                )

            # Poll until complete
            logger.info("   Waiting for generation...")
            while not operation.done:
                await asyncio.sleep(10) # Increased sleep to be kind to the API
                operation = await client.aio.operations.get(operation.name)

            if not operation.response or not operation.response.generated_videos:
                return GenerationResult(
                    success=False, provider=self.name, content_type="video", error="No video generated"
                )

            # Get video data
            video = operation.response.generated_videos[0]
            video_data = video.video.video_bytes

            # Save
            if output_path:
                file_path = Path(output_path)
            else:
                output_dir = get_settings().output_dir
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                file_path = output_dir / f"veo_{timestamp}.mp4"

            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(video_data)

            logger.info(f"✅ Veo: Saved to {file_path}")

            return GenerationResult(
                success=True,
                provider=self.name,
                content_type="video",
                file_path=file_path,
                data=video_data,
                metadata={
                    "aspect_ratio": aspect_ratio,
                    "model": model,
                    "prompt": prompt,
                },
            )

        except Exception as e:
            logger.error(f"Veo generation failed: {e}")
            return GenerationResult(
                success=False, provider=self.name, content_type="video", error=str(e)
            )

    async def _fetch_image(self, url: str) -> bytes:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.content