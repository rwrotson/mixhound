import math
from dataclasses import dataclass

# Label bounds: values below the first are low, above the second high, otherwise middle.
PULSE_BOUNDS = (0.2, 0.45)
BRIGHTNESS_HZ = (1200, 2800)
DYNAMICS_DB = (3.5, 9)
FLATNESS_BOUNDS = (0.06, 0.25)
ONSETS_PER_SECOND = (1.5, 4.5)


@dataclass(frozen=True)
class AudioFeatures:
    """Descriptive features of an audio clip."""

    tempo: float  # Beats per minute.
    pulse: float  # 0..1 strength of the periodicity behind the tempo.
    loudness_db: float  # Mean frame RMS, dBFS.
    dynamics_db: float  # Standard deviation of frame RMS, dB.
    centroid_hz: float  # Mean spectral centroid.
    rolloff_hz: float  # Mean 85% spectral roll-off.
    flatness: float  # Mean spectral flatness, 0 (tonal) .. 1 (noise).
    low: float  # Share of energy below 150 Hz.
    mid: float  # Share between 150 Hz and 2 kHz.
    high: float  # Share above 6 kHz.
    onset_rate: float  # Onsets per second.

    def describe(self) -> dict[str, object]:
        """Rounded values with plain-language labels for the research prompt."""
        return {
            "tempo_bpm": round(self.tempo),
            "pulse": _band(self.pulse, PULSE_BOUNDS, ("weak", "moderate", "strong")),
            "brightness": _band(self.centroid_hz, BRIGHTNESS_HZ, ("dark", "balanced", "bright")),
            "dynamics": _band(
                self.dynamics_db,
                DYNAMICS_DB,
                ("compressed", "moderately dynamic", "very dynamic"),
            ),
            "texture": _band(self.flatness, FLATNESS_BOUNDS, ("tonal", "mixed", "noisy/textural")),
            "density": _band(
                self.onset_rate,
                ONSETS_PER_SECOND,
                ("sparse", "medium density", "dense"),
            ),
            "low_end_share": round(self.low, 2),
        }


# (feature, transform, scale): differences are divided by `scale` after the transform.
_SCALES: tuple[tuple[str, str, float], ...] = (
    ("loudness_db", "linear", 6.0),
    ("dynamics_db", "linear", 3.0),
    ("centroid_hz", "log2", 0.6),
    ("rolloff_hz", "log2", 0.7),
    ("flatness", "sqrt", 0.12),
    ("low", "linear", 0.12),
    ("mid", "linear", 0.15),
    ("high", "linear", 0.08),
    ("onset_rate", "log2", 0.8),
)


def _band(value: float, bounds: tuple[float, float], labels: tuple[str, str, str]) -> str:
    """The label for a value below, between or above the two bounds."""
    low, high = bounds
    return labels[0] if value < low else labels[2] if value > high else labels[1]


SHARPNESS = 0.25
_WEIGHTS = {
    "tempo": 1.2,
    "loudness_db": 0.5,
    "dynamics_db": 0.8,
    "centroid_hz": 1.2,
    "rolloff_hz": 0.6,
    "flatness": 1.0,
    "low": 0.8,
    "mid": 0.5,
    "high": 0.7,
    "onset_rate": 1.0,
}


def _transform(value: float, kind: str) -> float:
    if kind == "log2":
        return math.log2(max(value, 1e-6))
    if kind == "sqrt":
        return math.sqrt(max(value, 0.0))
    return value


def _tempo_distance(one: float, two: float) -> float:
    """Octaves apart, treating half and double time as close (a 0.25-octave penalty)."""
    ratio = abs(math.log2(max(one, 1) / max(two, 1)))
    return min(ratio, abs(ratio - 1) + 0.25, abs(ratio - 2) + 0.5)


def similarity(one: AudioFeatures, two: AudioFeatures) -> float:
    """0..100 similarity of two clips' sonic character."""
    # Tempo counts fully only when both clips have a clear pulse.
    tempo_weight = _WEIGHTS["tempo"] * min(1.0, min(one.pulse, two.pulse) / 0.4)
    weighted = tempo_weight * (_tempo_distance(one.tempo, two.tempo) / 0.15) ** 2
    total = _WEIGHTS["tempo"]
    for name, kind, scale in _SCALES:
        difference = (
            _transform(getattr(one, name), kind) - _transform(getattr(two, name), kind)
        ) / scale
        weighted += _WEIGHTS[name] * min(difference**2, 9.0)
        total += _WEIGHTS[name]
    return 100.0 * math.exp(-SHARPNESS * weighted / total)


@dataclass(frozen=True)
class ReferenceAudio:
    """A reference recording's features and weight."""

    features: AudioFeatures
    polarity: str
    priority: int


def reference_similarity(candidate: AudioFeatures, references: list[ReferenceAudio]) -> int | None:
    """Priority-weighted similarity to positive references, reduced by closeness to negatives."""
    positives = [item for item in references if item.polarity == "positive"]
    negatives = [item for item in references if item.polarity == "negative"]
    if not positives:
        return None
    weights = sum(item.priority for item in positives)
    score = (
        sum(item.priority * similarity(candidate, item.features) for item in positives) / weights
    )
    if negatives:
        closest = max(similarity(candidate, item.features) for item in negatives)
        score -= 0.5 * max(0.0, closest - score)
    return max(0, min(100, round(score)))
