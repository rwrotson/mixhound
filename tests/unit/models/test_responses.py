from pydantic import BaseModel

from mixhound.models.domain import (
    DiscoveryListeningLink,
    DiscoveryPath,
    DiscoveryReference,
    DiscoveryRun,
    DiscoveryRunInput,
    DiscoverySession,
    DiscoverySettings,
    DiscoverySource,
    DiscoveryTrack,
    FeedbackEvent,
    FeedbackState,
    RunMetrics,
)
from mixhound.models.requests import ReferenceIn
from mixhound.models.responses import (
    FeedbackEventOut,
    FeedbackStateOut,
    ListeningLinkOut,
    PathOut,
    ReferenceOut,
    ResultTrackOut,
    RunInputOut,
    RunMetricsOut,
    RunOut,
    SessionOut,
    SettingsOut,
    SourceOut,
)


def test_out_models_hide_only_internal_fields() -> None:
    public: set[tuple[type[BaseModel], type[BaseModel]]] = {
        (DiscoverySettings, SettingsOut),
        (DiscoveryReference, ReferenceOut),
        (FeedbackEvent, FeedbackEventOut),
        (DiscoverySession, SessionOut),
        (DiscoverySource, SourceOut),
        (DiscoveryListeningLink, ListeningLinkOut),
        (DiscoveryPath, PathOut),
        (DiscoveryRunInput, RunInputOut),
        (RunMetrics, RunMetricsOut),
        (DiscoveryRun, RunOut),
        (FeedbackState, FeedbackStateOut),
    }
    for stored, out in public:
        assert set(stored.model_fields) - set(out.model_fields) == {"version"}, out.__name__
        assert set(out.model_fields) <= set(stored.model_fields), out.__name__

    assert set(DiscoveryTrack.model_fields) - set(ResultTrackOut.model_fields) == {
        "version",
        "audio_features",
    }
    assert set(ResultTrackOut.model_fields) <= set(DiscoveryTrack.model_fields)


def test_reference_input_takes_every_client_field() -> None:
    assert set(DiscoveryReference.model_fields) - set(ReferenceIn.model_fields) == {
        "version",
        "id",
        "created_at",
    }
