import asyncio
import shutil
import tempfile
from pathlib import Path

import numpy as np

from mixhound.services.audio import SAMPLE_RATE
from mixhound.services.ports import Samples

CLIP_SECONDS = 30
MIN_SECONDS = 5
DECODE_TIMEOUT = 30


def decode_args(ffmpeg: str, path: Path, sample_rate: int, seconds: int) -> list[str]:
    """Return the ffmpeg command that writes the first seconds of a file as mono f32 samples."""
    return [
        ffmpeg, "-nostdin", "-v", "error",
        "-i", str(path),
        "-t", str(seconds),
        "-ac", "1",
        "-ar", str(sample_rate),
        "-f", "f32le",
        "-",
    ]  # fmt: skip


class FfmpegDecoder:
    """Decodes preview clips with the ffmpeg binary."""

    def __init__(
        self,
        *,
        ffmpeg: str | None = None,
        sample_rate: int = SAMPLE_RATE,
        seconds: int = CLIP_SECONDS,
        timeout: float = DECODE_TIMEOUT,
    ) -> None:
        self.ffmpeg = ffmpeg or shutil.which("ffmpeg")
        self.sample_rate = sample_rate
        self.seconds = seconds
        self.timeout = timeout

    def available(self) -> bool:
        """Whether ffmpeg is installed."""
        return self.ffmpeg is not None

    async def decode(self, data: bytes) -> Samples | None:
        """Return mono samples of the clip, or None if it is shorter than MIN_SECONDS."""
        if self.ffmpeg is None:
            raise OSError("ffmpeg is not installed")
        # A file, not a pipe: MP4 previews keep their index at the end.
        with tempfile.TemporaryDirectory(prefix="mixhound-audio-") as directory:
            path = Path(directory) / "clip"
            await asyncio.to_thread(path.write_bytes, data)
            process = await asyncio.create_subprocess_exec(
                *decode_args(self.ffmpeg, path, self.sample_rate, self.seconds),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                async with asyncio.timeout(self.timeout):
                    stdout, stderr = await process.communicate()
            except TimeoutError:
                process.kill()
                await process.wait()
                raise
        if process.returncode != 0:
            raise OSError(f"ffmpeg failed: {stderr.decode(errors='replace')[:200]}")
        samples = np.frombuffer(stdout, dtype=np.float32)
        if samples.size < self.sample_rate * MIN_SECONDS:
            return None
        return samples.astype(np.float64)
