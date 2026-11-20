import asyncio
import logging
from dataclasses import asdict

import librosa
import numpy as np

from mixhound.models.domain import DiscoveryResearchResult, DiscoveryTrack
from mixhound.services.audio_features import AudioFeatures, ReferenceAudio, reference_similarity
from mixhound.services.cache import DAY, CacheCodec, CacheSpace, remember
from mixhound.services.ports import AUDIO, AudioDecoder, Cache, FetchError, PageFetch, Samples
from mixhound.services.preparation import Preparation
from mixhound.services.run_state import RunState, bounded

log = logging.getLogger(__name__)

SAMPLE_RATE = 22_050
FRAME = 2048
HOP = 512
MAX_AUDIO_BYTES = 12_000_000
FEATURES = CacheSpace("audio-features", 90 * DAY)


# Band edges in Hz of the low, mid and high energy shares.
BANDS = ((0, 150), (150, 2000), (6000, SAMPLE_RATE))
# Frames quieter than this many dB below the loudest frame are silence or fades.
AUDIBLE_RANGE_DB = 50


def _tempo(envelope: Samples) -> tuple[float, float]:
    """Tempo in BPM and the strength of the periodicity behind it, 0..1."""
    centred = envelope - envelope.mean()
    if not centred.any():
        return 0.0, 0.0
    tempo = float(librosa.feature.tempo(onset_envelope=envelope, sr=SAMPLE_RATE, hop_length=HOP)[0])
    correlation = librosa.autocorrelate(centred)
    correlation = correlation / correlation[0]
    lag = round(60 * SAMPLE_RATE / HOP / tempo)
    pulse = float(correlation[lag]) if lag < correlation.size else 0.0
    return tempo, min(1.0, max(0.0, pulse))


def features(samples: Samples) -> AudioFeatures:
    """Compute the features of mono samples; raise ValueError for a too short clip."""
    if samples.size < FRAME * 4:
        raise ValueError("Audio clip is too short")
    # The floor keeps silent frames finite, as V1 did.
    magnitude = np.abs(librosa.stft(samples, n_fft=FRAME, hop_length=HOP, center=False)) + 1e-10
    power = magnitude**2
    rms = librosa.feature.rms(y=samples, frame_length=FRAME, hop_length=HOP, center=False)[0]
    level = librosa.amplitude_to_db(rms, amin=1e-9, top_db=None)
    audible = level > level.max() - AUDIBLE_RANGE_DB
    centroid = librosa.feature.spectral_centroid(S=power, sr=SAMPLE_RATE, n_fft=FRAME)[0]
    rolloff = librosa.feature.spectral_rolloff(
        S=power, sr=SAMPLE_RATE, n_fft=FRAME, roll_percent=0.85
    )[0]
    flatness = librosa.feature.spectral_flatness(S=magnitude, power=1.0)[0]
    frequencies = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=FRAME)
    heard = power[:, audible]
    energy = heard.sum()
    bands = [
        heard[(frequencies >= low) & (frequencies < high)].sum() / energy for low, high in BANDS
    ]
    envelope = librosa.onset.onset_strength(y=samples, sr=SAMPLE_RATE, hop_length=HOP)
    tempo, pulse = _tempo(envelope)
    onsets = librosa.onset.onset_detect(onset_envelope=envelope, sr=SAMPLE_RATE, hop_length=HOP)
    return AudioFeatures(
        tempo=tempo,
        pulse=pulse,
        loudness_db=float(level[audible].mean()),
        dynamics_db=float(level[audible].std()),
        centroid_hz=float(centroid[audible].mean()),
        rolloff_hz=float(rolloff[audible].mean()),
        flatness=float(flatness[audible].mean()),
        low=float(bands[0]),
        mid=float(bands[1]),
        high=float(bands[2]),
        onset_rate=float(len(onsets) / (samples.size / SAMPLE_RATE)),
    )


def _decode_features(raw: object) -> AudioFeatures:
    if not isinstance(raw, dict):
        raise TypeError("Cached audio features are not an object")
    return AudioFeatures(**raw)


_CODEC = CacheCodec[AudioFeatures | None](
    encode=lambda value: asdict(value) if value else None,
    decode=_decode_features,
)


class AudioAnalyzer:
    """Features of public audio clips, cached by the recording they belong to."""

    def __init__(self, fetch: PageFetch, decoder: AudioDecoder, cache: Cache) -> None:
        self.fetch = fetch
        self.decoder = decoder
        self.cache = cache

    def available(self) -> bool:
        """Whether clips can be decoded on this host."""
        return self.decoder.available()

    async def url(self, url: str, key: str | None = None) -> AudioFeatures | None:
        """A public audio URL (a preview or stream), or None when it cannot be measured.

        `key` identifies the recording when the URL itself carries an expiring token.
        """

        async def compute() -> AudioFeatures | None:
            try:
                _, body = await self.fetch.page(
                    url, max_bytes=MAX_AUDIO_BYTES, truncate=True, accept=AUDIO
                )
                samples = await self.decoder.decode(body)
                if samples is None:
                    return None
                return await asyncio.to_thread(features, samples)
            except (FetchError, OSError, ValueError, TimeoutError) as error:
                log.info("audio_features_failed key=%s error=%s", (key or url)[:120], error)
                return None

        return await remember(self.cache, FEATURES, key or url, compute, _CODEC)


def marked_audio(feedback: list[dict[str, object]]) -> list[ReferenceAudio]:
    """Earlier results marked positive or negative, with measured clips, as audio references."""
    references = []
    for item in feedback:
        audio, polarity, weight = item.get("audio"), item.get("polarity"), item.get("weight")
        if (
            isinstance(audio, dict)
            and polarity in ("positive", "negative")
            and isinstance(weight, int)
        ):
            try:
                references.append(ReferenceAudio(AudioFeatures(**audio), str(polarity), weight))
            except TypeError:
                continue
    return references


class AudioMatcher:
    """Measured audio similarity of research results to the reference recordings."""

    def __init__(self, audio: AudioAnalyzer | None) -> None:
        self.audio = audio

    async def compare(
        self,
        state: RunState,
        result: DiscoveryResearchResult,
        preparation: Preparation | None,
    ) -> None:
        """Measure each candidate's clip against the references' clips."""
        if self.audio is None or preparation is None:
            return
        polarity = {reference.id: reference for reference in state.run.input.references}
        references = [
            ReferenceAudio(
                facts.audio,
                polarity[reference_id].polarity,
                polarity[reference_id].priority,
            )
            for reference_id, facts in preparation.facts.items()
            if facts.audio is not None and reference_id in polarity
        ]
        references += marked_audio(state.run.input.feedback)
        if not any(item.polarity == "positive" for item in references):
            return
        audio = self.audio

        async def compare(track: DiscoveryTrack) -> None:
            state.check()
            found, origin = await self._clip_features(audio, state, track)
            if found is None:
                return
            track.audio_features = asdict(found)
            track.audio_similarity = reference_similarity(found, references)
            if track.audio_similarity is None:
                return
            track.audio_analyzed = True
            described = found.describe()
            track.audio_analysis_evidence = (
                f"Measured from a 30 s clip ({origin}): about {described['tempo_bpm']} BPM, "
                f"{described['brightness']}, {described['dynamics']}, {described['texture']}, "
                f"{described['density']}; compared with "
                f"{sum(item.polarity == 'positive' for item in references)} positive reference "
                "or marked result clip(s) "
                "by tempo, loudness, spectrum, and onset features."
            )
            state.metrics.audio_analyzed += 1

        await bounded(result.results + result.unverified_leads, compare)

    @staticmethod
    async def _clip_features(
        audio: AudioAnalyzer, state: RunState, track: DiscoveryTrack
    ) -> tuple[AudioFeatures | None, str]:
        """A clip from the first catalog preview that decodes."""
        recording = state.recordings.get(track.id)
        for preview in recording.previews if recording else ():
            found = await audio.url(preview.url, key=f"{preview.source}:{preview.page_url}")
            if found:
                return found, f"{preview.source} preview"
        return None, ""
