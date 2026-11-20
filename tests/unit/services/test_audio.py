from dataclasses import asdict

import numpy as np
import pytest

from mixhound.infrastructure.cache import MemoryCache
from mixhound.models.domain import (
    DiscoveryReference,
    DiscoveryResearchResult,
    DiscoveryRun,
    DiscoveryRunInput,
    DiscoveryTrack,
)
from mixhound.services import audio
from mixhound.services.audio import FEATURES, AudioAnalyzer, AudioMatcher, features, marked_audio
from mixhound.services.audio_features import AudioFeatures, similarity
from mixhound.services.ports import AUDIO, CatalogRecording, FetchError, Preview, Samples
from mixhound.services.preparation import Preparation, ReferenceFacts
from mixhound.services.run_state import RunState
from tests.fakes import FakeDecoder, FakeFetch
from tests.support.factories import clicks

# The first librosa call in a fresh environment compiles numba code for about half a minute.
LIBROSA_TIMEOUT = 180


def sound(**changes: float) -> AudioFeatures:
    values = {
        "tempo": 100.0,
        "pulse": 0.6,
        "loudness_db": -20.0,
        "dynamics_db": 5.0,
        "centroid_hz": 1500.0,
        "rolloff_hz": 3000.0,
        "flatness": 0.1,
        "low": 0.3,
        "mid": 0.5,
        "high": 0.05,
        "onset_rate": 2.0,
    }
    values.update(changes)
    return AudioFeatures(**values)


@pytest.mark.timeout(LIBROSA_TIMEOUT)
def test_audio_features_measure_tempo_and_brightness() -> None:
    slow, fast = features(clicks(90)), features(clicks(140))
    assert abs(slow.tempo - 90) < 6 or abs(slow.tempo - 180) < 10
    assert abs(fast.tempo - 140) < 8 or abs(fast.tempo - 70) < 5
    assert slow.pulse > 0.4
    bright = features(clicks(90, tone=5000))
    assert bright.centroid_hz > slow.centroid_hz
    assert similarity(slow, slow) == pytest.approx(100)
    assert similarity(slow, features(clicks(91))) > similarity(slow, bright)


@pytest.mark.timeout(LIBROSA_TIMEOUT)
def test_silence_has_no_tempo_or_onsets() -> None:
    quiet = features(np.zeros(audio.SAMPLE_RATE * 6))

    assert (quiet.tempo, quiet.pulse, quiet.onset_rate) == (0.0, 0.0, 0.0)


def test_too_short_clip_is_rejected() -> None:
    with pytest.raises(ValueError, match="too short"):
        features(np.zeros(audio.FRAME))


def measured(monkeypatch: pytest.MonkeyPatch, result: AudioFeatures) -> list[int]:
    """Replace the librosa analysis; return the sizes of the analysed sample arrays."""
    sizes: list[int] = []

    def fake(samples: Samples) -> AudioFeatures:
        sizes.append(samples.size)
        return result

    monkeypatch.setattr(audio, "features", fake)
    return sizes


async def test_url_features_are_cached_under_the_recording_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sizes = measured(monkeypatch, sound())
    fetch = FakeFetch({"cdn.example/clip.m4a": b"clip"})
    cache = MemoryCache()
    analyzer = AudioAnalyzer(fetch, FakeDecoder({b"clip": np.ones(10)}), cache)

    first = await analyzer.url("https://cdn.example/clip.m4a?token=1", key="itunes:page")
    second = await analyzer.url("https://cdn.example/clip.m4a?token=2", key="itunes:page")

    assert first == second == sound()
    assert sizes == [10]
    assert fetch.requests == [("https://cdn.example/clip.m4a?token=1", AUDIO)]
    assert await cache.get(FEATURES.name, "itunes:page") == (True, asdict(sound()))
    assert analyzer.available()


@pytest.mark.parametrize(
    "decoded",
    [None, OSError("ffmpeg missing"), TimeoutError(), ValueError("bad clip")],
)
async def test_audio_failure_gives_none_and_is_not_cached(
    monkeypatch: pytest.MonkeyPatch, decoded: Samples | Exception | None
) -> None:
    measured(monkeypatch, sound())
    cache = MemoryCache()
    analyzer = AudioAnalyzer(
        FakeFetch({"cdn.example/clip.mp3": b"clip"}), FakeDecoder({b"clip": decoded}), cache
    )

    assert await analyzer.url("https://cdn.example/clip.mp3") is None
    assert cache.entries == {}


async def test_unmeasurable_samples_give_none(monkeypatch: pytest.MonkeyPatch) -> None:
    def too_short(samples: Samples) -> AudioFeatures:
        raise ValueError("Audio clip is too short")

    monkeypatch.setattr(audio, "features", too_short)
    analyzer = AudioAnalyzer(
        FakeFetch({"cdn.example/a.mp3": b"a"}), FakeDecoder({b"a": np.ones(2)}), MemoryCache()
    )

    assert await analyzer.url("https://cdn.example/a.mp3") is None


async def test_unreachable_clip_gives_none() -> None:
    analyzer = AudioAnalyzer(
        FakeFetch({"cdn.example/clip.mp3": FetchError("unreachable", 403)}),
        FakeDecoder(),
        MemoryCache(),
    )

    assert await analyzer.url("https://cdn.example/clip.mp3") is None


async def test_unreadable_cache_entry_is_measured_again(monkeypatch: pytest.MonkeyPatch) -> None:
    sizes = measured(monkeypatch, sound())
    cache = MemoryCache()
    await cache.set(FEATURES.name, "key", "not an object", 60)
    analyzer = AudioAnalyzer(
        FakeFetch({"cdn.example/a.mp3": b"a"}), FakeDecoder({b"a": np.ones(3)}), cache
    )

    assert await analyzer.url("https://cdn.example/a.mp3", key="key") == sound()
    assert sizes == [3]


def test_marked_audio_keeps_only_valid_marks() -> None:
    feedback: list[dict[str, object]] = [
        {"audio": asdict(sound()), "polarity": "positive", "weight": 3},
        {"audio": asdict(sound()), "polarity": "neutral", "weight": 3},
        {"audio": {"tempo": 1.0}, "polarity": "negative", "weight": 2},
        {"audio": asdict(sound()), "polarity": "negative", "weight": "2"},
        {"polarity": "positive", "weight": 1},
    ]

    [mark] = marked_audio(feedback)

    assert (mark.features, mark.polarity, mark.priority) == (sound(), "positive", 3)


def matcher_state(
    *tracks: DiscoveryTrack, feedback: list[dict[str, object]] | None = None
) -> tuple[RunState, DiscoveryReference, DiscoveryResearchResult]:
    reference = DiscoveryReference(kind="manual", artist="Low", title="Words")
    run = DiscoveryRun(
        session_id=reference.id,
        input=DiscoveryRunInput(references=[reference], feedback=feedback or []),
    )
    return RunState(run), reference, DiscoveryResearchResult(results=list(tracks))


async def test_matcher_measures_the_first_preview_that_decodes() -> None:
    track = DiscoveryTrack(artist="Duster", title="Inside Out")
    state, reference, result = matcher_state(track)
    state.recordings[track.id] = CatalogRecording(
        "Duster",
        "Inside Out",
        previews=(
            Preview("itunes", "https://music.apple.com/a", "https://cdn.example/broken.m4a"),
            Preview("bandcamp", "https://duster.bandcamp.com/track/x", "https://cdn.example/b.mp3"),
        ),
    )
    clips: dict[bytes, Samples | Exception | None] = {b"broken": None, b"stream": np.ones(4)}
    analyzer = AudioAnalyzer(
        FakeFetch({"cdn.example/broken.m4a": b"broken", "cdn.example/b.mp3": b"stream"}),
        FakeDecoder(clips),
        MemoryCache(),
    )
    preparation = Preparation(facts={reference.id: ReferenceFacts(audio=sound())})

    with pytest.MonkeyPatch.context() as patch:
        measured(patch, sound())
        await AudioMatcher(analyzer).compare(state, result, preparation)

    assert track.audio_analyzed
    assert track.audio_similarity == 100
    assert track.audio_features == asdict(sound())
    assert "(bandcamp preview)" in (track.audio_analysis_evidence or "")
    assert "1 positive reference" in (track.audio_analysis_evidence or "")
    assert state.metrics.audio_analyzed == 1


async def test_matcher_uses_marked_results_and_needs_a_positive_reference() -> None:
    track = DiscoveryTrack(artist="Duster", title="Inside Out")
    negative_only = [{"audio": asdict(sound()), "polarity": "negative", "weight": 2}]
    state, _, result = matcher_state(track, feedback=negative_only)
    state.recordings[track.id] = CatalogRecording(
        "Duster", "Inside Out", previews=(Preview("itunes", "p", "https://cdn.example/a.m4a"),)
    )
    fetch = FakeFetch({})

    await AudioMatcher(AudioAnalyzer(fetch, FakeDecoder(), MemoryCache())).compare(
        state, result, Preparation()
    )

    assert not track.audio_analyzed
    assert fetch.requests == []


async def test_matcher_without_previews_or_analyzer_changes_nothing() -> None:
    measured_track = DiscoveryTrack(artist="A", title="B")
    positive = [{"audio": asdict(sound()), "polarity": "positive", "weight": 2}]
    state, _, result = matcher_state(measured_track, feedback=positive)
    analyzer = AudioAnalyzer(FakeFetch({}), FakeDecoder(), MemoryCache())

    await AudioMatcher(analyzer).compare(state, result, Preparation())
    await AudioMatcher(None).compare(state, result, Preparation())
    await AudioMatcher(analyzer).compare(state, result, None)

    assert measured_track.audio_features is None
