# Bug report draft — non-finite samples trap the Int16 WAV exporter; fp16 VAE + 1024-frame tile emits NaN audio

**Status:** superseded — the three sections were submitted on 2026-11-15 as GitHub issues
`#3` (A), `#4` (B) and `#5` (C) on this repository; A is being fixed by PR `#6` from the fork
`ramishi/yue2-mlx-swift` (branch `fix/non-finite-wav-export`). B's follow-up investigation is
recorded in issue `#4` and `.local-runs/investigate-b-20261115/FINDINGS.md`.

This file is kept as the working draft only; do not resubmit it. The sections below are deliberately
independent so they can be split into separate reports.

**Target:** the maintainer of `VincentGourbin/yue2-mlx-swift` — i.e. **this repository**, not
`multimodal-art-projection/YuE` and not `ml-explore/mlx`. This is an independent port
(`README.md`: "Independent port, no copied code"); `AudioExporter.swift`, `YuE2VAE.decodeTiled` and
`ReferenceProfiles.swift` are all local code, and the reference implementation is PyTorch.

**Commit tested:** `ea75281343a55dba5ccc2d5d6d1644585219a8d0` ("feat: plan layout and lead-in trimming,
yue2 score (v1.10.0)"). `origin/main` still points at this commit as of 2026-11-15, so the defect is
unfixed; `CHANGELOG.md` 1.10.0 has no related entry.

**Host:** Apple M5, 32 GB unified memory, macOS 27.2.1 (arm64), Xcode 27.2, Release build via
`xcodebuild` (`.xcodebuild/Build/Products/Release/yue2`). Models in
`$YUE2_MODELS_DIR` (the local download cache, see `yue2 download --model packs`):
`int4-mixed-head`, `qint8-all-head`, bf16 checkpoint, VAE.

**Raw evidence:** `.local-runs/repro-20261115-1056/` (logs, WAVs, constructed `.npy`), earlier sittings
in `.local-runs/bench.noindex/m5-20261115-*`, and the crash reports listed below. Nothing was edited
after the fact.

---

## Summary — three separate defects

| | Defect | Confidence | Action |
|---|---|---|---|
| **A** | `AudioExporter.exportToWAV` traps on any non-finite sample in `.int16` mode, killing the process and losing the whole render's artifacts | **Proven**, deterministic, minimal repro | Fix now |
| **B** | fp16 VAE weights with a 1024-frame decode tile produce NaN samples on long renders | **Proven** as a sufficient condition in isolated `decode`; mechanism and non-determinism **open** | Investigate |
| **C** | `--reference` silently overwrites six per-call CLI knobs, making profile bisection from the CLI misleading | **Proven** by A/B runs | Warn (documentation-level fix) |

A is the reason a bad take is *lost*; B is the reason a take is *bad*. They share a symptom but have
different owners, different fixes and different test strategies. C is what made the original
bisection data unreliable and belongs in its own small report.

---

## A. The Int16 exporter traps instead of degrading

`AudioExporter.swift:50` converts with `Int32((sample * 32767.0).rounded())` and has **no non-finite
guard anywhere in the file**. Swift traps on NaN/Inf here:

```
Swift runtime failure: Float value cannot be converted to Int32 because it is either infinite or NaN
```

Crash path: `AudioExporter.exportToWAV(...)` → `SongResult.save(to:format:)` →
`SongResult.saveArtifacts(to:format:)` → `GenerateCommand.run()` (`GenerateCommand.swift:189`).
Termination is `EXC_BREAKPOINT / SIGTRAP` ("Trace/BPT trap: 5"), not a segfault — confirmed in
`~/Library/Logs/DiagnosticReports/yue2-2026-11-15-105921.ips`, which names `AudioExporter.swift:50`.

`.float32` export (`AudioExporter.swift:58-60`) writes `sample.bitPattern` without a numeric
conversion, so it survives and **silently writes the NaN into the WAV**. It is not a clean bypass:
samples still go through gain, `MLX.clip`, transpose/reshape and `asType` (`:26-38`); only the integer
conversion is skipped. `MLX.clip` propagates NaN — see the minimal repro below.

### Deterministic minimal repro (no generation, no model download beyond the VAE)

Construct an fp32-finite latent whose values overflow fp16 (`1e8 > 65504`), then decode:

```python
import struct
T, D = 64, 64
h = "{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d), }" % (T, D)
h += ' ' * ((64 - (10 + len(h) + 1) % 64) % 64) + '\n'
b = b'\x93NUMPY' + bytes([1, 0]) + struct.pack('<H', len(h)) + h.encode()
b += struct.pack('<f', 1e8) * (T * D)
open('fp16-overflow-latent.npy', 'wb').write(b)
```

```bash
yue2 decode --latents fp16-overflow-latent.npy --out overflow-int16.wav --precision fp16 --format int16
# rc=133, same Swift runtime failure

yue2 decode --latents fp16-overflow-latent.npy --out overflow-f32.wav  --precision fp16 --format float32
# rc=0, WAV is 100 % NaN, 0 Inf
```

This also settles a previously open question: fp16 overflow is converted to **pure NaN, with no
Inf left**, by the decoder's arithmetic. The "0 infinities" observation in every WAV is therefore
consistent with an fp16-overflow mechanism. It also confirms `MLX.clip` does not absorb NaN.

### Full-render repro (verified 2026-11-15)

Request: `run/bench-70s.json` (seed 42, `cot: full`), 1750 semantic tokens = 70 s.

```bash
export YUE2_MODELS_DIR="${YUE2_MODELS_DIR:-$HOME/.cache/yue2-mlx/models}"
YUE2=.xcodebuild/Build/Products/Release/yue2

$YUE2 generate --request run/bench-70s.json --reference 4bit-fast \
  --semantic-min-tokens 1750 --semantic-max-tokens 1750 --out out/int16-1
# rc=133 — repeated 3/3 times on this host

$YUE2 generate --request run/bench-70s.json --reference 4bit-fast \
  --semantic-min-tokens 1750 --semantic-max-tokens 1750 --format float32 --out out/f32
# rc=0, 70.0 s audio
```

`--profile` (a valid flag: `GenerateCommand.swift:44`) was **not** used in the verified runs; it is
optional and orthogonal. Earlier reported traps: `4bit-fast`, `8bit-fast`, `16bit-fast` at 1750
tokens, `16bit-fast` at 900 tokens; a 400-token render succeeds. Crash reports from 2026-11-14:
`yue2-2026-11-14-{222715,223031,231224,231531}.ips`, `yue2-2026-11-15-011840.ips` (8bit-fast).
Six more were produced by the 2026-11-15 reproduction sitting (`105921` … `110519`).

### Impact

- The WAV is never written. Because `SongResult.saveArtifacts` writes the plan first and the WAV
  before `semantic.npy` / `latent.npy` / `result.json` (`SongResult.swift:65-97`), a trapped run
  leaves **only the five plan files** (`score.abc`, `abc_tokens.npy`, `prefix.npy`, `plan.json`,
  `plan_manifest.json`). The entire take — including the expensive NAR solve — is unrecoverable.
- The exporter is shared by **7 call sites in 6 commands**: `GenerateCommand.swift:189`,
  `SynthesizeCommand.swift:91`, `EditCommands.swift:117` and `:151` (via `saveArtifacts`), plus
  `DecodeCommand.swift:102`, `RemixCommand.swift:101`, `EncodeCommand.swift:68` (directly). B can be
  fixed without ever touching A, but any non-finite audio from *any* source still crashes these.

### Suggested fix and test

In `AudioExporter.exportToWAV`, replace non-finite samples before scaling
(`let safe = sample.isFinite ? sample : 0`) and clamp the scaled value before the `Int32`
conversion; surface a `warnings` entry so callers can tell the user the take is degraded.

**Do not use a long-render regression test.** The original draft suggested rendering 1750 tokens with
`4bit-fast` and asserting rc=0; that would be flaky (see B's non-determinism) and slow. Use a
**deterministic unit test** instead: feed an `MLXArray` containing NaN/Inf to `exportToWAV` in both
formats and assert rc=0, a complete file, and a warning. `Tests/YuE2Tests/NumpyAndWavTests.swift`
already has int16/float32 cases to extend.

---

## B. fp16 VAE + 1024-frame tile produces NaN audio

### What is established: a single-variable isolation

Take the **`latent.npy` saved by the successful float32 render** — shape (1750, 64), float32,
**0 NaN / 0 Inf**, max abs 5.1562, and **bit-identical (SHA-256) to the `latent.npy` of the 2026-09-22
sitting6 runs B1/B2**. Decode that one file with `yue2 decode`, changing one knob at a time:

| VAE precision | `--core-frames` | Export | Result |
|---|---|---|---|
| fp16 | 1024 | int16 | **rc=133** (2/2, plus a repeat) |
| fp16 | 256 | int16 | rc=0 |
| fp16 | 64 | int16 | rc=0 |
| **fp32** | 1024 | int16 | rc=0 |
| fp16 | 1024 | float32 | rc=0, WAV contains **92,058 NaN** |

```bash
L=out/f32/latent.npy
$YUE2 decode --latents "$L" --out t1024.wav --precision fp16 --core-frames 1024 --format int16   # rc=133
$YUE2 decode --latents "$L" --out t256.wav  --precision fp16 --core-frames 256  --format int16   # rc=0
$YUE2 decode --latents "$L" --out tfp32.wav --precision fp32 --core-frames 1024 --format int16   # rc=0
```

Note `decode` uses `--core-frames`; `generate` uses `--vae-core-frames`. They are different flags.

This removes the LM, NAR solve, quantization pack and weight residency from the causal chain: **NaN
is created inside the VAE decode**, and **fp16 weights together with a 1024-frame tile are jointly
sufficient** — either one changed avoids the trap. All three `fast` profiles share
`vaePrecision: .fp16` + `vaeCoreFrames: 1024` (`ReferenceProfiles.swift:69-104`); all `lean`/`tiny`
profiles use 256/64, which is consistent with never reproducing the fault there.

### What is ruled out

The original draft named "bf16 overflow/underflow in the dequantised NAR solve" as the prime suspect.
That is contradicted four ways:

1. `16bit-fast` is `quant: .none`, `narCompute: .packed`, `precision: .bf16`
   (`ReferenceProfiles.swift:96-99`) — no quantization and no dequantised solve — and it traps too.
2. `m5-20261115-0930-diag5` run C passes `--nar-compute packed` explicitly and still traps.
3. `CachedNAR.solve` ends with `guard isFinite` → `throw .nonFiniteLatents`
   (`CachedNAR.swift:135`, `:184`); a NaN latent would produce a Swift error, not an `Int32` trap.
4. The saved `latent.npy` is finite and bit-identical across configurations.

The NAR/semantic stages are also exonerated by `YuE2VAE.swift:60`, which unconditionally
`precondition(isFinite(z))` on each tile's input, and by the artifact measurements below.

### Corrected measurements

Two independent sittings, both `--format float32` (which survives and exposes the corruption):

| Sitting | Tokens | Samples | NaN | Share | First NaN | Tail share that is NaN |
|---|---|---|---|---|---|---|
| 2026-11-14, `4bit-fast` | 900 | 3,455,872 | 132,472 | **3.8 %** | 1,969,058 (57 %) | 132,472 / 1,486,814 = **8.9 %** |
| 2026-11-14, `16bit-fast` | 1750 | 6,719,872 | 723,508 | **10.8 %** | 5,311,138 (79 %) | 723,508 / 1,408,734 = **51.4 %** |
| 2026-11-15, `4bit-fast` | 1750 | 6,719,872 | 133,112 | **1.98 %** | 5,805,218 (86.4 %) | 133,112 / 914,654 = **14.6 %** |

All rows: **0 Inf**, finite samples clipped to |1.0| (max finite amplitude 0.2600 in the 2026-11-15
row). The earlier draft's "~11 % at 900 tokens, ~33 % at 1750 tokens" is arithmetically wrong, and
"starts late and runs to the end" is inaccurate: the NaNs are **scattered** through the tail (8.9 %–
51.4 % of tail samples), not a contiguous corrupted suffix.

Intermediate artifacts are clean:

| File | Shape | NaN | Inf | max abs |
|---|---|---|---|---|
| `semantic.npy` | (900,) int32 | 0 | 0 | 32701 |
| `latent.npy` | (900, 64) float32 | 0 | 0 | 5.4375 |
| `latent.npy` (2026-11-15, 1750 tok) | (1750, 64) float32 | 0 | 0 | 5.1562 |

### What is still open

- **Non-determinism.** The same `decode` command repeated 4 times always puts the first NaN at
  sample **2,627,618**, but the NaN count moves between **92,058 and 157,654**. On the `generate`
  path, B1 completed and B2 trapped with the identical configuration, request and seed
  (`m5-20261115-0920-sitting6`, staged residency + 1024 tile), while another sitting with the same
  flags completed twice. Even the same latent decoded via `generate` vs. standalone `decode` differs
  (first NaN 5,805,218 / finite max 0.26 vs. 2,627,618 / 1.0). So the *onset point* is stable but the
  *spread* is not — unresolved, and likely the source of the reported intermittency.
- **Consequence for the original bisection.** Its four-row table varied tile size *and* residency at
  once, and `lean`/`tiny` additionally change `precision`, `narCompute` and `memoryProfile`. It
  supports correlation, **not** "1024 is the necessary condition" and not "resident weights raise
  the probability". In isolated `decode`, residency is absent entirely and fp16+1024 still traps.
- **Where in the decoder.** fp16 activations exceeding 65504 remain the most plausible mechanism,
  now backed by the 1e8-latent repro. `SnakeBeta.swift:43-45` has intermediate-layer `isFinite`
  assertions, but they only compile under `YUE2_DEBUG` (`YuE2Core.swift:35`), which Release does not
  set — a good place to instrument first.

### Suggested next steps (not a fix proposal)

Instrument the fp16 decode at tile boundaries and inside `SnakeBeta` in a debug build to find the
first non-finite tensor; check whether a 1024-frame tile's transients alone cross the fp16 ceiling;
and only then decide between smaller tiles, fp32 decode for large tiles, or clamping. Do **not**
pre-emptively clamp the NAR output (nothing indicates it is out of range) or change the default
residency on the strength of the current multi-variable data.

---

## C. `--reference` silently overwrites per-call knobs

`GenerateCommand.swift:87-99` unconditionally reassigns, when `--reference` is present:

| CLI flag | Reassigned local | Ref |
|---|---|---|
| `--quant` | `quant` | `:92` |
| `--quant-head` | `quantHead` | `:93` |
| `--precision` | `precision` | `:94` |
| `--vae-precision` | `vaePrecisionChoice` | `:95` |
| `--vae-core-frames` | `vaeTile` | `:96` |
| `--release-weights-between-stages` | `releaseFlag` | `:97` |

`--ode-steps` is **not** swallowed (`:98` uses `odeSteps ?? profile.odeSteps`), and neither are
`--nar-compute` (`:112`) or `--compiled-decode` (`:113`), which are applied after the profile block.
The profile's summary *is* printed to stderr (`:99`), but nothing warns that the six flags above were
discarded.

Empirical consequence: `m5-20261115-0930-diag5` runs A/B pass `--vae-core-frames 256` and
`--release-weights-between-stages` alongside `--reference 4bit-fast` and still trap — they actually
ran with tile 1024 and full residency. `m5-20261115-1010-diag6`, which spells the same configuration
out by hand without `--reference`, completes. This invalidates any bisection done through
`--reference` and is worth at least a warning (or making per-call flags win).

Unrelated cosmetic: the `--reference` help (`:74`) says "six reference configurations" and lists six,
while `ReferenceProfiles.all` has **seven** (`4bit-tiny` is missing from the help).

---

## Scope and caveats

- Single host (M5 / 32 GB / macOS 27.2.1). Worth re-checking on the M3 Max reference machine before
  concluding it is hardware-general; note that the isolated `decode` repro does not depend on the LM
  or on memory pressure, which argues against a pure "resident weights exhaust memory" explanation.
- Every claim about the exporter (A) is deterministic and host-independent; every claim about NaN
  generation (B) is specific to the runs listed above.
- The only file changed while rewriting this draft was this draft itself: no product code, no tests,
  no GitHub issues, and nothing was submitted externally. (A `fiche T-6.5 · en cours` line was added
  to `tasks/STATE.md` earlier in this session, before this investigation started; T-6.5 remains
  paused and is unrelated.)
