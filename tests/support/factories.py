from mixhound.models.research import ResearchCandidate


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
