from pathlib import Path

import pytest

from mixhound.config import DeadwaxSettings
from mixhound.infrastructure.catalogs.deadwax import open_catalog

pytestmark = [pytest.mark.live, pytest.mark.enable_socket, pytest.mark.timeout(120)]


async def test_a_known_recording_is_confirmed_with_leads(tmp_path: Path) -> None:
    settings = DeadwaxSettings(
        sources=["musicbrainz", "itunes", "listenbrainz"], cache_enabled=False
    )

    async with open_catalog(settings, tmp_path) as catalog:
        found = await catalog.reference("Radiohead", "Reckoner")

    assert found is not None
    assert "musicbrainz" in found.confirmed_by
    assert found.mbid
    assert found.artist_mbids
    assert any(lead.group == "lb_radio" for lead in found.leads)
