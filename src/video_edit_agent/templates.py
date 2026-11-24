

def available_template_ids(project_root: Path | None = None) -> list[str]:
    """Return template IDs visible from project, home, and built-in directories."""
    template_ids: set[str] = set()
    for directory in _template_directories(project_root, None, None):
        if not directory.is_dir():
            continue
        for path in (*directory.glob("*.yaml"), *directory.glob("*.yml")):
            if _TEMPLATE_ID.fullmatch(path.stem) and path.is_file():
                template_ids.add(path.stem)
    return sorted(template_ids)
