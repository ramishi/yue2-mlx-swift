# `YuE2Core` Swift API

Everything below is `public`, documented in the code (doc comments) and covered by tests. Package `YuE2Swift`, product `YuE2Core`; pinned dependencies (`mlx-swift` 0.31.6 exact, `swift-mlx-profiler` ≥ 1.5). macOS 15+ / iOS 27+, Swift 6, strict concurrency.

```swift
dependencies: [.package(url: "https://github.com/VincentGourbin/yue2-mlx-swift", exact: "1.0.0")]
```

## The shortest path

```swift
import YuE2Core

let profile = YuE2ReferenceProfile.named("4bit-lean")!
profile.applyGlobalPolicy()                       // memory profile, NAR compute, compiled decode

let session = try await ModelSession.load(
    modelsDir: modelsDir, quant: profile.quant, quantizeHead: profile.quantizeHead,
    precision: profile.precision,
    residency: profile.releaseWeightsBetweenStages ? [.ar] : .all)
let vae = try await loadVAEBackend(.mlx, directory: modelsDir.appendingPathComponent("YuE2-Vae"),
                                   precision: profile.vaePrecision)
let pipeline = YuE2Pipeline(
    session: session, vae: vae, config: session.config,
    vaeCoreFrames: profile.vaeCoreFrames,
    releaseWeightsBetweenStages: profile.releaseWeightsBetweenStages)

let song = try await pipeline.generate(request: request) { event in /* progress */ }
try song.saveArtifacts(to: outputDirectory)
```

`SongRequest` (style, lyrics, `cot`, seed, optional `abc`) is `Codable` and reads the CLI's JSON. `SongResult` carries the audio (`[1, samples, 2]` fp32), the latents, the tokens and the per-phase timings.

## Model and residency

```swift
public final class ModelSession {
    static func load(modelsDir:quant:quantizeHead:precision:residency:) async throws -> ModelSession
    var model: YuE2ForCausalLM; var tokenizer: YuE2Tokenizer; var config: GenerationConfig
    private(set) var resident: YuE2WeightResidency      // .ar, .nar, .all
    func loadWeights(of: YuE2WeightPath) throws           // no-op when already resident
    func releaseWeights(of: YuE2WeightPath) -> Int        // bytes released
    func releaseAllWeights() -> Int
}
```

A released branch is unusable until reloaded (it would compute on zeros): that is the contract of stage-scoped residency. Safetensors loading is lazy; a tensor that is never evaluated is never allocated. `applyPrecision(_:where:)` only casts resident branches (casting a parked branch materializes it: 1.44 GB, the iPhone peak of 0.2.1).

`YuE2ForCausalLM.dequantizeWeights(of: .nar)` replaces the 8-bit projections with bf16 `Linear`s (fast profiles); the branch can no longer be reloaded, so recreate the session for the next song.

## Pipeline and stages

```swift
public final class YuE2Pipeline {
    init(session:vae: any VAEDecoding, config:, vaeCoreFrames: Int? = nil, releaseWeightsBetweenStages: Bool? = nil)
    func generate(request:abcSampling:semanticSampling:profiling:timeline:onEvent:narResume:onNARStep:cancel:) async throws -> SongResult
}
```

The `nil` defaults come from the memory profile (Mac: 1024 / everything resident; mobile: 256 / stage-scoped residency). `PipelineEvent`: `.stage`, `.abcToken`, `.semanticToken`, `.narProgress`, `.vaeProgress`.

The stages exist separately, each resumable from its artifacts:

| Stage | Type | Input → output |
|---|---|---|
| 1 | `Planner.plan(request:abcSampling:onToken:cancel:)` | request → `SymbolicPlan` (score, prefix) |
| 2 | `SemanticGenerator.generateSemantic(plan:sampling:onToken:cancel:)` | plan → semantic tokens + KV cache |
| 3 | `Synthesizer.synthesize(prefix:codec:seed:steps:reuseCache:onBeforeChunk:resume:onStep:cancel:)` | tokens → latents `[T, 64]` |
| 4 | `decodeTiled(using: any VAEDecoding, _:coreFrames:haloFrames:)` | latents → audio |

`onBeforeChunk(needsARPath:)` is where residency releases the AR branch and loads the NAR one; `onStep` delivers a `NARCheckpoint` after every ODE step and `resume:` restarts from one (bit-exact resume, single-chunk songs).

```swift
public struct NARCheckpoint { chunkIndex, step, steps, state: MLXArray
    func save(to: URL) throws; static func load(from: URL) throws -> NARCheckpoint?; static func clear(in: URL) }
```

## Audio in and song edits

The model has no audio understanding at inference; its entries are its own artifacts. `YuE2Pipeline+Edit.swift` exposes the two edits an app needs, `MelodyTranscriber` turns a recording into the plan.

```swift
// hummed / sung melody → ABC score → song on that melody
let samples = try AudioImporter.loadAudio(url: recording)                     // [1, n, 2] fp32 at 48 kHz, or [Float]
var options = MelodyTranscriber.Options(); options.bpm = 110                   // tempo of the recording, section label, note floor
let melody = MelodyTranscriber.transcribe(audio: samples, sampleRate: 48_000, options: options)
melody.notes            // [MelodyNote(midi:startSixteenth:lengthSixteenths:)], melody.key, melody.voicedRatio
var request = SongRequest(style: style, lyrics: lyrics); request.abc = melody.abc   // the score is the plan, no planner run
let song = try await pipeline.generate(request: request)

// variation: same score, same tokens, SDEdit on the latents (strength 0 = unchanged, 1 = fresh)
let variation = try await pipeline.vary(song, strength: 0.4, seed: 2, onEvent: onEvent)
// "regenerate from here": keep up to a frame (40 ms), re-sample the tokens after it, solve against the kept part
let ending = try await pipeline.regenerate(song, fromFrame: 20 * 25, seed: 3, onEvent: onEvent)   // length: .keepSource by default, .free lets the AR decide

let saved = try SongResult.load(from: directory)                               // a generate --out directory, for later edits
```

Underneath: `SemanticGenerator.generateSemantic(plan:sampling:continuation:onToken:cancel:)` takes the kept tokens as the AR's past; `Synthesizer.synthesize(... edit: NAREdit?)` with `.variation(latents:strength:)` (start state `noise·t + latents·(1−t)`, `round((1−strength)·steps)` steps skipped) or `.keep(latents:frames:)` (kept frames re-imposed after every step, returned bit-exact); `CachedNAR.solve(... keep:)` is the inpainting mask. `MelodyTranscriber` is pure CPU (vDSP), deterministic, tested on synthetic tones; `NAREditTests` covers the two edits on the tiny model.

## Song timeline: bars, notes and lyrics on the audio

A player follows a generated song from `timeline.json` (`LyricTimeline`): the start of every score bar and beat, every sung note, every line, word and syllable, each with its seconds and its musical position (`bar` 0-based, `beat` 1-based and fractional). The times come from the LM's own alignment heads, read right after the semantic phase: no audio analysis, no extra model. Method and measurements: [CLI.md](CLI.md#timing-timelinejson---timeline-karaoke).

```swift
// in a staged run: after the semantic phase, before the AR weights are released
let semantic = try SemanticGenerator(model: s.model, tokenizer: s.tokenizer, config: config)
    .generateSemantic(plan: plan, sampling: sampling, onToken: onToken, cancel: cancel)
let timeline = try AttentionTimeline.timeline(model: s.model, tokenizer: s.tokenizer, semantic: semantic, cancel: cancel)
try timeline.write(to: songDirectory.appendingPathComponent("timeline.json"))

// with the pipeline
let song = try await pipeline.generate(request: request, timeline: true)   // song.timeline, saved by saveArtifacts

// migration: songs saved before timelines existed (AR weights loaded once, any number of songs)
for directory in songDirectories {        // plan.json + prefix.npy beside semantic.npy, or in a plan/ subdirectory
    let timeline = try AttentionTimeline.reanalyze(song: directory, model: s.model, tokenizer: s.tokenizer, cancel: cancel)
    try timeline.write(to: directory.appendingPathComponent("timeline.json"))
}

// playback: binary searches, nothing recomputed
let timeline = try LyricTimeline.load(from: url)
let now = timeline.position(at: player.currentTime)   // bar, beat, note (while it sounds), line, word, syllable
```

| Type | Role |
|---|---|
| `LyricTimeline` | `bars` (`start`, `beats`, `section`), `notes` (`start`, `end`, `midi`, `bar`, `beat`, `beats`), `lines` → `words` (`onScore`) → `syllables` (`notes`, `firstNote`), `duration`, `grid`; `position(at:)`, `write(to:)`, `load(from:)`; `wordsOnScore` (under 0.9: the render left its score); `fit(abc:lyrics:language:)` (notes per syllable of an imposed score, before generating); `build(abc:lyrics:language:grid:heard:anchors:duration:)` places a score on any clock |
| `AttentionTimeline` | `timeline(model:tokenizer:semantic:)` (fresh song, reuses the KV cache), `timeline(model:tokenizer:plan:semantic:)`, `reanalyze(song:model:tokenizer:)`, `SavedSong(directory:)`, `read(...)` (raw bar and word starts); the heads: `barHeads`, `wordHead`, `barLead` |
| `ABCScore` | reader of the score dialect (two voices, sections, multi-bar rests, ties, inline `M:`/`K:` changes): bars, sounding notes, `barStarts` |
| `YuE2ForCausalLM.prefixAttention(...)` | the teacher-forced pass: attention mass of chosen heads on prompt segments per frame |

The pass runs layers 0–18 over prompt + song, one layer per unit of GPU work: it waits on `YuE2GPUGate` and throws `.cancelled` like the other stages. With the generation's cache (no CFG) it adds no cache memory; `reanalyze` builds its own cache for those 19 layers. A song without a score (`cot off`) throws; an instrumental gets bars and notes and no lines. `Language` (`en`, `fr`) is guessed from the style and decides syllabification. Lines, words and syllables are in lyric order with increasing starts; a syllable never runs into the next.

## Instrumental songs

```swift
let planned = try Planner(model: s.model, tokenizer: s.tokenizer, config: config).plan(request: request)   // lyrics: section tags if none
let (render, transfer) = try Instrumental.renderRequest(from: request, plannedABC: planned.abc!)
// render: Vocal notes moved to Ins, section tags as lyrics, "no vocals…" style, negativeStyle + cfg 1.5
```

`SongRequest.negativeStyle` (EXPERIMENTAL) conditions the negative CFG branch on tags (`[EOD] + instruction + [Tags] negativeStyle [Lyrics]`); it acts only when `guidance` ≠ 1. A render can still hum the verse (2 seeds in 8): check it (SheetSage2 track-0 notes, `yue2 generate --instrumental` re-rolls the seed). On 3–4 minute songs a single 10–30 s hum can exceed the whole-track threshold and trigger a full re-render; the recipe doubles AR time and is not a vocal-free guarantee. See [App and UI integration](#app-and-ui-integration-measured-constraints) before exposing it in a UI.

## Transcription: `SheetSage2Core` (recording → ABC)

A separate library product, independent of `YuE2Core` (MLX only): the SheetSage2 port. Its output is the `SongRequest.abc` / `abcPrefix` of a cover with `cot: melody`.

```swift
import SheetSage2Core

// the upstream adapter release + its MERT-v2 parent (`yue2 download --model sheetsage2`), merged at load
let model = try SheetSage2Model.load(directory: modelsDir.appendingPathComponent("SheetSage2"))   // fp16 by default
let waveform: MLXArray = …                     // [samples] float32, 24 kHz mono, not normalized
let result = try SheetSage2Transcriber(model: model).transcribe(waveform, melodyOnly: true) { window, token in … }
result.abc               // String? — nil when the events do not make a score (result.abcError says why)
result.events            // [SheetSage2Event]: time, beat/meter, key, chord, structure, melody notes
result.tokens            // [[Int]], one greedy sequence per 300 s window

// iOS: pause in the background, cancel, progress (called before every unit of GPU work)
var gated = SheetSage2Transcriber(model: model, profile: .named("16bit-lean")!)
gated.checkpoint = { progress in            // SheetSage2Progress: stage, window/windows, encoderFraction, tokens, overallFraction
    guard YuE2GPUGate.shared.wait(cancel: { cancelled }) else { throw CancellationError() }
}

var request = SongRequest(style: style, lyrics: lyrics)
request.abc = result.abc                       // or request.abcPrefix: the planner continues it
request.cot = .melody
```

Underneath: `SheetSage2Model.encode(_:)` (log-mel, ConvNeXt, 24 Conformer blocks, layer mix: `[1, 7500, 512]` per window), `decode(_:memory:cache:)` (BART, preallocated KV cache), `SheetSage2Generator` (grammar-masked greedy decoding), `SheetSage2Tokenizer.decodeSequence`, `AbcNotation.abc(events:duration:melodyOnly:)`. `SheetSage2Weights.mergeAdapters` merges the LoRA factors (`W += B·A·α/r`, float32, CPU). Parity: tokens and ABC byte-identical to upstream in float32 (tiny fixture, two excerpts, a 500 s song in four windows); see plan/14-sheetsage2.md.

## Losing the GPU in the background (iOS)

iOS refuses every GPU submission from an app that is not in the foreground, and mlx-core raises the error inside the Metal completion handler, where it cannot be caught. The engine stops before that:

```swift
YuE2GPUGate.shared.suspend()   // scenePhase == .inactive
YuE2GPUGate.shared.resume()    // .active, and on every cancellation
```

The gate is awaited before every token, every NAR layer (when `YuE2ExecutionPolicy.evalPerLayer`, the mobile default: 28 units of 0.2-0.4 s instead of one 10-20 s graph) and every VAE tile; `wait(cancel:)` returns `false` when the cancellation lands during the wait. With per-step checkpoints, what remains costs one step.

## VAE backends

```swift
public protocol VAEDecoding {
    var config: VAEConfig { get }
    var enumeratedFrameCounts: [Int]? { get }        // nil = any length (MLX)
    func decode(_ z: MLXArray) async throws -> MLXArray
}
func loadVAEBackend(_ kind: VAEBackendKind, directory: URL, precision: VAEPrecision) async throws -> any VAEDecoding
func planVAETiles(frames:coreFrames:haloFrames:enumerated:) -> [VAETileWindow]
```

`.mlx` (always), `.coreaiGPU` (macOS/iOS 27, 54.7 dB parity thanks to `planVAETiles`' exact-shape windows), `.coreaiANE` (dead end with `coreai-torch` 0.4.2). `loadVAEBackend` throws `.backendUnavailable`, never substitutes silently: the MLX fallback belongs to the caller, logged. Never `.cpuOnly` for this model (transposed convolutions are wrong on CPU).

## Memory and execution policy

```swift
YuE2MemoryManager.profile                  // .mac | .mobile (#if os(iOS), YUE2_MEMORY_PROFILE)
YuE2MemoryManager.configure(for: .ar | .nar | .vae | .load)
YuE2MemoryManager.mobileLimitsMB()         // cache ≈ 1 GB, threshold = available − 1.25 GB
YuE2ExecutionPolicy.narCompute             // .packed | .dequantized
YuE2ExecutionPolicy.evalPerLayer           // mobile default
YuE2ExecutionPolicy.compiledDecode         // off, measured without gain
let sampler = FootprintSampler(); sampler.start(); …; sampler.stop(); sampler.peakMB; sampler.mlxActivePeakMB
```

`FootprintSampler` tracks `phys_footprint` (the number jetsam judges) on a dedicated thread, including what MLX does not see (Core AI, Metal).

## Reference profiles

```swift
public struct YuE2ReferenceProfile { id, bits, kind, quant, quantizeHead, precision, narCompute, compiledDecode,
    vaePrecision, vaeCoreFrames, releaseWeightsBetweenStages, memoryProfile, odeSteps, summary
    static let all: [YuE2ReferenceProfile]; static func named(_:) -> YuE2ReferenceProfile?; func applyGlobalPolicy() }
```

See [References.md](References.md) for the seven profiles and their measurements (`4bit-tiny` is the seventh; `generate --help` still says "six"). Before putting a profile picker in a UI, read [App and UI integration](#app-and-ui-integration-measured-constraints): these are memory/speed presets over the same weights, not quality tiers.

## Quantization

`YuE2Quantization`: `.none`, `.qint8`, `.int4` (AR path), `.qint8All`, `.int4All`, `.int4Mixed` (4-bit AR path, 8-bit NAR, 4-bit embeddings); `quantizeHead` adds `lm_head`. `LMWeightLoader.load` quantizes on the fly then exports `mlx-prequantized/<preset>[-head]/`; later loads read the export. Parities: AR int4 greedy 8/8, NAR 8-bit rel 0.049 over 32 steps (a 4-bit NAR fails: 0.15).

## App and UI integration: measured constraints

Read this before designing a generation UI. Everything here was measured on the
`fix/non-finite-wav-export` fork (M5 / 32 GB, macOS 27.2.1, 2026-11-15/16): a 12 s smoke test,
a 4.58-minute song, a 3.33-minute fixed-score instrumental cover and a 3-seed x 2-profile
control. Full tables live in `.local-runs/instrumental-longsong-20261116/`,
`.local-runs/instrumental-seeds-20261116/`, `.local-runs/tiny-eval-20261115/` and
`.local-runs/cover-eval-20261115/` (gitignored, machine-local) and are summarised in
`docs/knowledge/log.md`. Each rule below is a control a UI might be tempted to expose.

### 1. A profile picker is a memory/speed control, not a quality ladder

All seven reference profiles load the **same `YuE2-3B` weights**; `--instrumental` is orthogonal
to `--reference` (no per-profile branch in `GenerateCommand`). Do not label the picker
"low / medium / high quality", do not rank profiles by audio quality, and do not build a
"better instrumental model" toggle: this engine has no LoRA/adapter loading (the separate
`YuE2-Studio` project uses a community `yue2-instrumental` LoRA that is not portable here).
`4bit-tiny` is byte-identical to `4bit-lean` in every artifact, i.e. only VAE tile 64 / 256 MB
cache / 3072 MB GC threshold differ. On a 32 GB Mac the default is **`4bit-fast`**:
cooled 70 s bench 67.8 s / 8166 MB vs tiny 82.3 s / 2471 MB (+21 %; +10 % drift-adjusted),
120 s cover +10–20 %; 200 s instrumental 354–474 s / 10.63 GB vs 525–641 s / 5.02–5.07 GB
(**1.35–1.48x slower** on tiny). Tiny's long-song peak (~5 GB) is also above its own 3072 MB
threshold, so it no longer fits the ~3 GB device it was designed for. Enumerate profiles from
`yue2 references` at runtime: `generate --help` still says "six" and omits `4bit-tiny`, while
there are seven.

### 2. "Instrumental" is a workflow, and it is not a guarantee

`--instrumental` = move `Vocal` notes to `Ins` (871 on the 143-bar test score), replace lyrics with
section tags, append "no vocals, no singing…" to the style, add a negative voice-tag CFG branch
(`cfg_scale` 1.5) and check the render with SheetSage2. The official upstream skill states
`absence_of_vocals_checked=False`; a file check is not proof. Historically 6/8 first seeds were
clean (~25 % leak per seed, ~2 % after two retries); a leak hums **one verse for 10–30 s**, not
the whole song. UI copy must say "voice check: N sung notes (proxy)" rather than "vocal-free",
always offer listening, and expose an explicit re-roll. There is no local Demucs / AST
vocal-energy measurement to turn into a percentage.

### 3. Do not gate a long song on a whole-track sung-note count

`GenerateCommand.voiceNoteTolerance = 4` is an **absolute count over the entire track**, and a
retry re-renders the **whole song** with `seed + 1` (no re-plan; the imposed ABC is kept). On a
3.33-minute cover, one 12 s hum at 120–132 s produced exactly 12 track-0 notes and would
trigger up to three renders (~6 min each on `4bit-fast`, ~9–11 min on `4bit-tiny`). The longer
the song, the more likely one local hum is. Prefer a per-minute / 30 s window check, or
`--instrumental-retries 0` followed by an independent `yue2 transcribe --profile 16bit-fast`
on the saved WAV (~14 s) and a user-visible "re-roll with seed N+1" action. If you keep automatic
retries, show the seed, the attempt number, the estimated cost and a cancel button.

### 4. Never infer profile superiority from one render

Seed-matched, fixed-score control: `4bit-fast` 12 / 0 / 0 sung notes and `4bit-tiny` 0 / 0 / 0
over seeds 12300–12302 (one-sided Fisher exact p = 0.5; under a 25 % leak rate,
`P(0/3) = 0.42`). Both are compatible with the same leak rate; resolving a difference needs
20+ seeds per profile. The 12-note leak disappeared on the very first retry seed, which is
exactly what `--instrumental-retries 2` does. Do not use a single clean render to change the
default profile or to write "profile X is better at instrumental" into the UI.

### 5. Budget the negative CFG branch, especially on small-device presets

CFG 1.5 doubles the AR work: semantic 82.6 -> 132.7 s (+61 %) and peak process 9387 ->
10632 MB (+13 %) on `4bit-fast`; on `4bit-tiny` a 12 s instrumental went 2405 -> 3265 MB
(+36 %) and a 200 s one to ~5.0 GB. Most of that is CFG, not the voice check (retries 0 vs 1
differ by 7 MB). A hard ~2.5–3 GB device should not silently get the full recipe:
`--cfg-scale 1 --instrumental-retries 0` is the memory-safe combination, but it is unverified
for leak rate — surface it as an explicit "low memory" option with that caveat.

### 6. Comparing profiles fairly requires the cover method, and still cannot rank quality

Same profile + seed + fixed score -> bit-identical `semantic.npy`, `latent.npy`, `audio.wav`.
Different quantisation diverges within the first ~17 semantic tokens even with score, lyrics,
style, seed and length locked: it is a **different rendition**, not a degraded copy. So an A/B
feature must use `yue2 transcribe` -> one fixed `--abc-file` -> `cot melody`, and must not show a
quality score. `adherence.py` numbers (fast 100/98 % vs lean/tiny 70/68 %) are a beat-grid
artifact: a single global offset cannot follow SheetSage2 drift, refitting per 20 s gives lean
`98/100/100/80/91/96 %` on one seed, and `alignmentCost` barely moves (0.3 % apart on another
seed despite 100 % vs 57 % adherence). Do not display "adherence %".

### 7. Length, ETA and progress

A 4.58-minute song rendered in 274.9 s with steady RMS across 28 segments; a 3.33-minute
instrumental cover was clean on both profiles with 0 non-finite / 0 clipped samples and no
long silence. VAE scales linearly, NAR dominates and is superlinear; the planner does not grow
with the lyrics. Score planning and the `Vocal`→`Ins` transfer run in the CLI **before**
`pipeline.generate`, so the e2e number in `result.json` excludes them (with `--abc-file` the
planner is skipped entirely); do not use it as a speed comparison. Wall time drifts with
GPU thermals: the same batch saw `4bit-fast` 354 -> 474 s and `4bit-tiny` 525 -> 641 s. Compute
ETA from live phase events (`.stage`, `.semanticToken`, `.narProgress`, `.vaeProgress`), never
from one hardcoded constant.

### 8. Artifacts and failure states the UI must not misread

`AudioExporter` (the single write path for all 7 call sites / 6 CLI commands) replaces
non-finite samples with silence; finite samples are bit-identical, and `status: complete` in
`result.json` is unchanged. `audio.wav` is written **before** `semantic.npy` / `latent.npy` /
`result.json`, so a crash used to leave a partial directory: treat a missing `result.json` as
incomplete even if a WAV exists. A forced `--semantic-max-tokens` sets
`truncated.semantic: true` — that is an intentional cap, not an error. VAE tile size is **not**
a quality knob: tiles 64 / 640 / 1024 produce SHA-identical float32 audio, and the fp16
shape-safe guard halves the tile and re-decodes only when a crop is non-finite (1024 -> 512).
Do not expose a tile-size slider.

### 9. Provenance of these fixes (as of 2026-11-16)

The WAV sanitizer, the shape-independent VAE retry guard and the `4bit-fast` 640-frame default
live on the local fork branch `fix/non-finite-wav-export` (`f5961787`), with PR #6 open
upstream; the fork's default `main` does **not** contain them. If the UI depends on these
guarantees, pin that branch or commit rather than `main`.

## Tests

`Scripts/run-tests.sh` (swift-testing, parallelization 1: a known deadlock in mlx-swift). Two tiers: weight-free (tiny fixtures under `parity/`, always green, 156 tests) and real weights (`YUE2_MODELS_DIR`, suites `Real*`, `Quantization`, `GenerationSmoke`). Every numerical change goes through `RealLMParityTests` (greedy 8/8 on both phases) and `RealNARParityTests`.
