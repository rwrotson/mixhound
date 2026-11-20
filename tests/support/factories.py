import math

import numpy as np

from mixhound.models.research import ResearchCandidate
from mixhound.services.audio import SAMPLE_RATE
from mixhound.services.ports import Samples


def candidate(**changes: object) -> ResearchCandidate:
    values: dict[str, object] = {
        "artist": "Fogh Depot",
        "title": "Anticyclone",
        "artist_alt": None,
        "title_alt": None,
        "album": None,
        "year": 2015,
        "version_qualifier": None,
        "original_artist": None,
        "original_title": None,
        "genres": [],
        "similarity_score": 70,
        "confidence": "medium",
        "researched": True,
        "similarities": ["Slow brushed drums"],
        "differences": [],
        "closest_reference_ids": [],
        "discovery_paths": [],
        "sources": [],
        "listening_urls": [],
    }
    values.update(changes)
    return ResearchCandidate.model_validate(values)


def clicks(bpm: float, seconds: float = 12.0, tone: float = 0.0) -> Samples:
    """A quiet hum with noise bursts at `bpm`, optionally with a loud tone."""
    time = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    signal = 0.02 * np.sin(2 * math.pi * 110 * time)
    if tone:
        signal += 0.2 * np.sin(2 * math.pi * tone * time)
    period = int(SAMPLE_RATE * 60 / bpm)
    for start in range(0, signal.size - 600, period):
        signal[start : start + 600] += np.random.default_rng(start).normal(
            0,
            0.5,
            600,
        ) * np.hanning(600)
    return signal
