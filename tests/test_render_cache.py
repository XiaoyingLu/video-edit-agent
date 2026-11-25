"""Tests for render cache fingerprints and cache-key invalidation."""

from pathlib import Path

from video_edit_agent.rendering import render_cache


def test_render_cache_reuses_hash_when_file_metadata_is_unchanged(tmp_path: Path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"large source placeholder")
    index = tmp_path / "cache" / "fingerprints.json"
    calls = 0
    original = render_cache._file_digest

    def counted(path: Path) -> str:
        nonlocal calls
        calls += 1
        return original(path)

    monkeypatch.setattr(render_cache, "_file_digest", counted)
    first = render_cache.build_render_cache_key([source], {"quality": "high"}, fingerprint_index_path=index)
    second = render_cache.build_render_cache_key([source], {"quality": "high"}, fingerprint_index_path=index)

    assert first == second
    assert calls == 1


def test_render_cache_rehashes_when_file_metadata_changes(tmp_path: Path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"first")
    index = tmp_path / "cache" / "fingerprints.json"
    calls = 0
    original = render_cache._file_digest

    def counted(path: Path) -> str:
        nonlocal calls
        calls += 1
        return original(path)

    monkeypatch.setattr(render_cache, "_file_digest", counted)
    first = render_cache.build_render_cache_key([source], {}, fingerprint_index_path=index)
    source.write_bytes(b"second content")
    second = render_cache.build_render_cache_key([source], {}, fingerprint_index_path=index)

    assert first != second
    assert calls == 2
