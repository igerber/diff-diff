# Carousel decks

LinkedIn launch carousels for diff-diff estimators. Each deck is a
`generate_*.py` script that renders its committed PDF; the PDF is a build
artifact of the script and must never be edited independently.

## Regenerating

```bash
python carousel/generate_lpdid_carousel.py     # -> diff-diff-lpdid-carousel.pdf
python carousel/generate_spillover_carousel.py # -> diff-diff-spillover-carousel.pdf
# ... one generator per deck, same pattern
```

**Whenever a generator changes, regenerate its PDF in the same commit.** The
decks carry methodology claims (equivalences, parity tolerances, validation
scope), so a stale PDF can ship an overclaim that the script no longer makes.

## Dependencies

Carousel generation additionally requires `fpdf2` and `Pillow`, which are
intentionally NOT part of the library's install or dev extras; `matplotlib`
is already available through the dev/docs extras but is not a runtime
install dependency. Python >= 3.9.

```bash
python -m pip install fpdf2 Pillow
```

## Claim discipline

Slide copy that states methodology facts (equivalence targets, tolerances,
validated scope) must match `docs/methodology/REGISTRY.md` and the paper
review under `docs/methodology/papers/`. Qualifiers are part of the claim --
e.g. "Cengiz-style" stacking, "BJS-style" imputation (exact only in the
single-cohort PMD case), "default path" for LPDiD survey support. See the
"Claim discipline" section in each generator's module docstring for the
deck-specific rules, and keep footer version labels derived from
`pyproject.toml` (never hard-coded).
