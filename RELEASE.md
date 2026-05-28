# Release Preparation Checklist

## Pre-Release
- [ ] Update version in `src/config/settings.py` (if applicable)
- [ ] Update version in `pyproject.toml` (if applicable)
- [ ] Update version in `README.md` badges (if applicable)
- [ ] Ensure all tests pass: `pytest`
- [ ] Update `CHANGELOG.md` with new release notes
- [ ] Update documentation if needed
- [ ] Tag the release in Git

## Release
- [ ] Build and push Docker image (if applicable)
- [ ] Create GitHub release
- [ ] Upload release notes to GitHub release
- [ ] Announce release via appropriate channels

## Post-Release
- [ ] Monitor for any issues
- [ ] Respond to user feedback
- [ ] Plan next release cycle

## Versioning
We follow Semantic Versioning (SemVer):
- MAJOR version for incompatible API changes
- MINOR version for backward-compatible functionality
- PATCH version for backward-compatible bug fixes

## Current Version
0.1.0 (Initial release)