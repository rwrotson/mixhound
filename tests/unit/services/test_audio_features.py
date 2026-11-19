from dataclasses import replace

from mixhound.services.audio_features import (
    AudioFeatures,
    ReferenceAudio,
    reference_similarity,
    similarity,
)

CLIP = AudioFeatures(100.0, 0.6, -20.0, 5.0, 1500.0, 3000.0, 0.1, 0.4, 0.4, 0.05, 2.0)


def test_identical_clips_are_fully_similar_and_half_time_is_close() -> None:
    assert similarity(CLIP, CLIP) == 100.0
    assert similarity(CLIP, replace(CLIP, tempo=50.0)) > similarity(CLIP, replace(CLIP, tempo=70.0))


def test_descriptions_use_plain_labels() -> None:
    assert CLIP.describe() == {
        "tempo_bpm": 100,
        "pulse": "strong",
        "brightness": "balanced",
        "dynamics": "moderately dynamic",
        "texture": "mixed",
        "density": "medium density",
        "low_end_share": 0.4,
    }


def test_reference_similarity_weighs_positives_and_penalizes_negatives() -> None:
    bright = replace(CLIP, centroid_hz=4000.0, flatness=0.4)
    positive = ReferenceAudio(CLIP, "positive", 5)
    negative = ReferenceAudio(bright, "negative", 3)

    assert reference_similarity(CLIP, [negative]) is None
    assert reference_similarity(CLIP, [positive]) == 100
    near_negative = reference_similarity(bright, [positive, negative])
    assert near_negative is not None
    assert near_negative < round(similarity(bright, CLIP))
