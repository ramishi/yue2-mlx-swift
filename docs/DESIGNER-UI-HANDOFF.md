# UI designer handoff — engine capabilities

**Message for designers:** this public repository is the authority for what the YuE2 engine and pipeline can actually do. The separate private app repository (`ramishi/yue2-desktop`) holds the Co-Producer product, UX and layout specs. There is **no backend HTTP API**: the app runs this engine in-process through the pinned Swift package.

This document is an index. For the full contract, read [`docs/API.md`](API.md), especially
[App and UI integration: measured constraints](API.md#app-and-ui-integration-measured-constraints),
and [`docs/CLI.md`](CLI.md) for the complete option inventory.

## Version to review

The private app pins this engine at:

- commit `d432f6f9637ba2565ecae61bad6d1b9bbb761f1c`
- tag `v1.10.0-ramishi.1`

Use the pinned tag when reading API docs for the current app. `fork/main` may gain documentation-only changes after that pin.

## Capability map

| Capability | Public surface | Notes for UI design |
|---|---|---|
| Full generation | `YuE2Pipeline.generate(request:onEvent:)` | Stages are Plan → Semantic → Synthesise → VAE. Use real stage events for progress; there is no “Mixing” stage. |
| Planner / score | `Planner.plan`, `SongRequest.abc`, `SongRequest.abcPrefix` | A score can be imposed or continued. A run directory’s artifacts are the song/take record. |
| Melody from recording | `MelodyTranscriber` | CPU pitch tracking for a hum/sung line. Returns notes, key, tempo and voiced ratio. Low voiced ratio needs coaching, not a silent failure. |
| Cover transcription | `SheetSage2Core` | Mixed recording → ABC score, warnings and status. No per-bar confidence heatmap exists. |
| Variation | `YuE2Pipeline.vary` | SDEdit on saved latents: same performance, different texture. Not an energy/arrangement control. |
| Regenerate from a point | `YuE2Pipeline.regenerate` | Keeps audio before the chosen frame bit-for-bit and rewrites the rest. Good for a bigger ending. |
| Timeline / karaoke | `LyricTimeline`, `AttentionTimeline`, `timeline.json` | Bars, sung notes, lines, words and syllables timed on the actual render. Bind sing-along per version. |
| Instrumental | `Instrumental.renderRequest` + proxy voice check | A workflow, not a separate model. It is not a vocal-free guarantee; copy must say “voice check: N sung notes (proxy)”. |
| Profiles / downloads | `YuE2ReferenceProfile`, `ModelDownloader` | Seven profiles are speed/memory presets over the same weights. Do not label them low/medium/high quality. |
| Resilience | `PipelineEvent`, `NARCheckpoint`, `YuE2GPUGate` | Per-step checkpointing, cancellation and GPU-loss handling exist. Queue and notification UX remain app-side work. |
| VAE backends | `VAEDecoding`, `loadVAEBackend` | MLX always; Core AI GPU on 27+, ANE is a dead end. VAE tile size is not a quality knob. |

## Controls to avoid

- “Polish”, “mix” or mastering: no such engine operation.
- Named singers, voice catalogue or voice cloning: not supported and prohibited by product rules.
- Quality ladder based on reference profiles: wrong technical meaning.
- Adherence percentage: a beat-grid artifact, not a reliable score.
- VAE tile slider: tiles 64/640/1024 produce identical audio.
- Per-bar transcription confidence: SheetSage2 does not emit it.
- LoRA or adapter loading: absent from this clean engine base.

## Where the product specs live

In the private app repo, designers should read:

- `docs/coproducer/DESIGNER-API-REVIEW.md` — full cross-repo index and suggested review output
- `docs/coproducer/HANDOFF-coproducer-dev.md`
- `docs/coproducer/PRD-coproducer.md`
- `docs/coproducer/ux-spec-coproducer.md`, especially §4
- `docs/coproducer/layout-coproducer.md`

If you do not have access to the private repo, ask for it. This public repo intentionally contains engine capability and constraints only; it does not copy private product copy, wireframes or app source.
