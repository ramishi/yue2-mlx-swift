# This fork: a clean engine base for a macOS app

**Upstream:** [`VincentGourbin/yue2-mlx-swift`](https://github.com/VincentGourbin/yue2-mlx-swift) (`v1.10.0`, `ea752813`)
**Fork:** [`ramishi/yue2-mlx-swift`](https://github.com/ramishi/yue2-mlx-swift), default branch **`fork/main`**
**Purpose:** an engine-only, reproducible Swift package that a **separate macOS SwiftUI app** can pin by commit/tag and ship as one signed `.app` inside a DMG. Weights are *not* in the bundle: the app downloads them from Hugging Face on first launch.

Read [docs/API.md](docs/API.md#app-and-ui-integration-measured-constraints) before designing the app UI.

## Branch topology

| Branch | Tracks | Role |
|---|---|---|
| `main` | `origin/main` | **Read-only mirror of upstream.** Never commit here. |
| `fork/main` | `myfork/fork/main` | Our mainline (the fork's default branch): upstream + our patches + our docs. |
| `fix/non-finite-wav-export` | `myfork/fix/non-finite-wav-export` | **Frozen head of [PR #6](https://github.com/VincentGourbin/yue2-mlx-swift/pull/6).** No new commits: anything added here lands in the PR. |
| `investigate/fp16-vae-nan` | local only | Ruled-out hypotheses (`bf16`, input/weight scaling). Not published. |

`fork/main` starts at `f5961787` (the PR head) plus one documentation/cleanup commit, tagged
`v1.10.0-ramishi.1`. A plain `git clone` of the fork therefore gets the fixes.

## Our delta from upstream

Three small, independently revertible commits, submitted as PR #6 (OPEN, mergeable):

| Commit | File(s) | Why |
|---|---|---|
| `77b6ed4` | `AudioExporter.swift`, `NumpyAndWavTests.swift` | Non-finite samples trapped the Int16 WAV writer (`rc=133`) and lost the whole render. Sanitise at the single WAV entry point; finite audio is bit-identical. |
| `39d9e34` | `ReferenceProfiles.swift` | `vaeCoreFrames` 1024 → 640 on the three `*-fast` profiles: a safe default, no longer the only defence. |
| `f596178` | `VAEDecoding.swift`, `YuE2VAE.swift`, `VAETilingTests.swift` | Shape-independent runtime guard: on a non-finite decode crop, halve the tile (1024→512→…→64) and retry. Both decode entry points are covered (`generate` and `synthesize`/`remix`/`encode`/`parity`). Normal path costs nothing. |

Upstream issues [#3](https://github.com/VincentGourbin/yue2-mlx-swift/issues/3),
[#4](https://github.com/VincentGourbin/yue2-mlx-swift/issues/4) and
[#5](https://github.com/VincentGourbin/yue2-mlx-swift/issues/5) record the reports; the working
draft is archived in `docs/upstream/`. Measured UI constraints live in
`docs/API.md`, `docs/CLI.md` and `docs/knowledge/`.

## Keeping in sync with upstream

The rule that makes this work: **`main` is a pure mirror, all our work lives on `fork/main` in small commits.**

Regular sync (upstream released a new version, PR still open):

```bash
git fetch origin --tags
git switch main && git merge --ff-only origin/main
git switch fork/main && git rebase main          # replay our patches + docs on the new tip
git push myfork fork/main --force-with-lease    # only after a rebase rewrote our commits
```

**If PR #6 is accepted** (upstream now contains the fix, typically squashed, so patch-ids differ):
drop our three fix commits and replay only what comes after them. `f5961787` is the last fix commit,
so `f5961787..fork/main` is exactly our documentation layer:

```bash
git fetch origin --tags
git switch main && git merge --ff-only origin/main
git switch fork/main
git rebase --onto main f5961787 fork/main       # keeps docs, discards already-merged fixes
git tag v1.11.0-ramishi.1                      # or the matching upstream version
git push myfork fork/main --force-with-lease && git push myfork v1.11.0-ramishi.1
```

If upstream cherry-picked the commits verbatim, a plain `git rebase main` skips them automatically
by patch-id. If upstream fixed the same defects differently, prefer `git merge main` into
`fork/main` (one recorded conflict resolution) over a rebase, then retest. Expect conflicts in
`docs/API.md` / `docs/CLI.md`: keep our measured section, re-read upstream's new text.

Never force-push `fix/non-finite-wav-export` while PR #6 is open. Never edit `main` locally.

After each sync, re-run `Scripts/run-tests.sh` (or `swift test --disable-sandbox`) and bump the
app's `ENGINE_VERSION`: the app pins a SHA, not a branch.

## Consuming this engine from the app repository

The app is a **separate repository** (the upstream project already follows this pattern:
`docs/iOS.md` describes `YuE2Studio` in a separate repo pinning the package exactly).

```
yue2-desktop/
├── ENGINE_VERSION          # fork commit SHA + tag, the single source of truth
├── YuE2Desktop.xcodeproj   # real Xcode app target (SwiftUI App lifecycle)
├── YuE2Desktop/            # app sources; seed from Sources/YuE2BenchUI
├── scripts/sync-engine.sh  # fetch, verify SHA, update ENGINE_VERSION
└── README.md
```

Pin an immutable revision, never `.branch("fork/main")`:

```swift
// Package.swift (or the Xcode project's package reference)
.package(url: "https://github.com/ramishi/yue2-mlx-swift.git",
         .revision("<ENGINE_VERSION sha>"))   // or .exact("1.10.0-ramishi.1")
```

Link the libraries, not the CLI: `YuE2Core` (pipeline, profiles, downloader, WAV export) and
`SheetSage2Core` (transcription). A native SwiftUI app calls `YuE2Pipeline.generate(request:)` and
its `PipelineEvent` stream directly — **no HTTP server, no SSE bridge, no subprocess** (that was
only needed for a web UI). `docs/API.md` has the exact `ModelSession.load` / `loadVAEBackend` /
`YuE2Pipeline` setup.

**Seed for the UI:** `Sources/YuE2BenchUI/` is already a 761-line SwiftUI macOS app
(`GenerationViewModel` driving the pipeline, `ModelPicker`, `MetricsPanel`, `HistoryView`,
`AudioPlayer`, `TabView`). Copy its patterns into the app repo. It is a SwiftPM *executable*
target with an `NSApplicationDelegate` hack to force `.regular` activation — that is fine for a
dev tool but **not** a shippable `.app`. Use a real Xcode app target for the DMG.

## Shipping one `.app` in a DMG

- **Build with `xcodebuild`, never `swift build` for a shippable binary.** SPM executables cannot
  find MLX's `default.metallib` (README, `docs/CLI.md`, `CLAUDE.md`). An Xcode app target compiles
  and bundles it — documented for iOS in `docs/iOS.md`; **smoke-test this on macOS** with a 1 s
  generation on a clean machine before shipping.
- **Weights stay outside the DMG.** `ModelDownloader` (in `YuE2Core`) does resumable chunked
  Hugging Face downloads with SHA-256 verification and `DownloadProgress` callbacks:
  `download(_ model: YuE2Model, progress:)`, `download(pack: YuE2Pack, progress:)`, plus
  `downloadPackSupport`. Packs: `.int4MixedHead`, `.qint8AllHead`, `.int4Head`; support files and
  the LM/VAE/SheetSage2 checkpoints come from `YuE2Model`. Optional `HF_TOKEN` for gated repos.
- **Pass an explicit models directory.** `ModelSession.load(modelsDir:)` accepts a URL, so the app
  can use `~/Library/Application Support/<bundle-id>/Models`. On macOS
  `YuE2ModelsDirectory.resolveDefault()` returns `nil` unless `$YUE2_MODELS_DIR` is set, and a
  double-clicked `.app` inherits no shell environment — `YuE2BenchUI/ModelPicker.swift` already
  persists a `UserDefaults` override for exactly this reason. Do not rely on env vars.
- **Licence:** code is MIT, but the YuE2 LM/VAE/tokenizer weights and generated audio are
  **CC-BY-NC-4.0 (non-commercial)** (README "Licences", `YuE2License.ccByNc4`). Surface that in
  the first-launch/download screen; never pre-bundle weights in the DMG.
- DMG specifics: disable App Sandbox (Metal/MLX + arbitrary model files), keep Hardened Runtime
  and outgoing-network entitlement, `xcodebuild archive` → notarise with `notarytool` → staple →
  wrap in DMG. The increased-memory-limit entitlement in `docs/iOS.md` is iOS-only.
- UI rules from measured data: profile picker = speed/memory, not quality; say "voice check: N
  sung notes (proxy)", never "vocal-free"; judge long songs per window instead of re-rendering the
  whole song; ETA from stage events; no adherence% display; no VAE tile slider; a missing
  `result.json` means incomplete.

## Deliberately absent from this clean base

App/UI source (other than upstream's own `YuE2BenchUI` target), model weights and WAVs,
`.local-runs/`, `run/`, `.workbuddy/`, `.vscode/`, machine-local `Scripts/bench-m5-*.sh` with
absolute paths. All are ignored by `.gitignore`, not deleted locally.

## Verify the base

```bash
git log --oneline main..fork/main          # 3 fix commits + 1 docs commit
git diff --stat main..fork/main            # 6 code/test files + docs only
git tag --points-at HEAD                   # v1.10.0-ramishi.1
gh pr view 6 --repo VincentGourbin/yue2-mlx-swift
```
