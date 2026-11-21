def test_legacy_package_namespace_resolves_canonical_modules() -> None:
    import travel_video_agent
    import video_edit_agent
    from travel_video_agent import models as legacy_models

    assert travel_video_agent.__version__ == video_edit_agent.__version__
    assert travel_video_agent.__path__ == video_edit_agent.__path__
    assert legacy_models.MediaInfo.__name__ == "MediaInfo"
