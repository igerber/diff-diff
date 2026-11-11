### Internal
- **Release wheel build moved off the retired `macos-14` runner**:
  `build-wheels.yml`'s macOS ARM64 job now runs on `macos-15`. GitHub retired
  the macOS 14 images from GitHub Actions on 2026-11-02, so every macOS wheel
  job was rejected before starting and the next release could not have
  published. No change to the library or to the published wheel set.
