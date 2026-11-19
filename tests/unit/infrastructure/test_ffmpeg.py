import shutil
import subprocess
from pathlib import Path

import pytest

from mixhound.infrastructure.ffmpeg import FfmpegDecoder, decode_args

FFMPEG = shutil.which("ffmpeg")
needs_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg is not installed")


def tone(path: Path, seconds: int) -> bytes:
    assert FFMPEG is not None
    subprocess.run(  # noqa: S603 - fixed arguments
        [
            FFMPEG,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={seconds}",
            str(path),
        ],
        check=True,
    )
    return path.read_bytes()


def test_decode_args_ask_for_mono_float_samples(tmp_path: Path) -> None:
    args = decode_args("ffmpeg", tmp_path / "clip", 22_050, 30)

    assert args[args.index("-t") + 1] == "30"
    assert args[args.index("-ac") + 1] == "1"
    assert args[args.index("-ar") + 1] == "22050"
    assert args[-3:] == ["-f", "f32le", "-"]


async def test_a_missing_binary_is_reported() -> None:
    decoder = FfmpegDecoder(ffmpeg=None)
    decoder.ffmpeg = None

    assert not decoder.available()
    with pytest.raises(OSError, match="not installed"):
        await decoder.decode(b"")


@needs_ffmpeg
async def test_a_clip_is_decoded_to_mono_samples(tmp_path: Path) -> None:
    samples = await FfmpegDecoder(seconds=6).decode(tone(tmp_path / "tone.wav", 8))

    assert samples is not None
    assert samples.size == 6 * 22_050


@needs_ffmpeg
async def test_a_short_clip_is_none(tmp_path: Path) -> None:
    assert await FfmpegDecoder().decode(tone(tmp_path / "tone.wav", 2)) is None


@needs_ffmpeg
async def test_undecodable_data_fails() -> None:
    with pytest.raises(OSError, match="ffmpeg failed"):
        await FfmpegDecoder().decode(b"not audio")


@needs_ffmpeg
async def test_a_slow_decode_times_out(tmp_path: Path) -> None:
    with pytest.raises(TimeoutError):
        await FfmpegDecoder(timeout=0).decode(tone(tmp_path / "tone.wav", 8))
