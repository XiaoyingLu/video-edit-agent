# Releasing

The project is currently pre-release. Do not create a version tag until the
corresponding package build, CI matrix, Docker build, and end-to-end smoke test
have passed.

For the first release:

1. Update `version` in `pyproject.toml`.
2. Move the relevant entries from `Unreleased` in `CHANGELOG.md` into a
   dated version heading.
3. Run the checks in [START_HERE.md](START_HERE.md).
4. Merge the release PR, then create an annotated tag on the merge commit:

   ```bash
   git switch main
   git pull --ff-only
   git tag -a v0.1.0 -m "Video Edit Agent 0.1.0"
   git push origin v0.1.0
   ```

5. Publish the matching GitHub Release and package artifact only after checking
   the tag points at the reviewed commit.
