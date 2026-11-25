# Issues from the first runs on an NVIDIA card

Runs of 2026-11-25 on one PC: NVIDIA GeForce RTX 5090 (32-lane waves),
driver 32.0.16.2135, Windows 11 26200. Release and `diag` builds of this fork
(`main` after the merge of upstream `c96b4f2`). Shaders are named by hash
only (see `asset-policy.md`).

Status key: **open**, **fixed** (in this fork), **upstream** (worth raising
there), **by design**.

## What the runs show

The title boots, its menus and loading screens draw correctly, the opening
in-engine scene draws, and no shader fails to translate. Measured from the
`diag` build's per-frame op lists (`VICTORY_CAPTURE_WORLD=1`):

| Dropped per captured frame | Count |
|---|---|
| a stage of the ray-tracing structure builder, which is off | 440 to 3,453 |
| a compute pipeline the host refuses (64-lane waves) | 4 to 5 |
| a pipeline still being made | 0 to 36 |
| other | 1 to 11 |

So nearly all dropped work is the ray-tracing structure builder, which is off
by default (`VICTORY_BVH_BUILDER`), see `fork-ray-tracing.md`. Without it the
title's indirect light and lamp/sun shadows are missing.

## 1. Compute programs that need 64-lane waves are refused on NVIDIA (open)

D3D12, with the debug layer: `CreateComputeShader: Shader wants wave lane
count of 64 but maximum wave lane count for device is 32.` Seven programs in
the runs so far, all compute, all of which translate and validate:

| Hash | Group | Why it keeps 64-lane waves (`VICTORY_WAVE_WHY=1`) | With `VICTORY_WAVE_WIDE=1` |
|---|---|---|---|
| 2da4beb386509b77 | 256 | LDS in a group over 32, an LDS read and write in one lane-branch region, `v_readlane` | still 64 (several console waves) |
| 369cbe906ea72764 | 64 | LDS in a group over 32, an LDS read and write in one lane-branch region | **32** |
| 2cc4c197ee29e291 | 64 | as above, plus a mask as data, `v_mbcnt` | **32** |
| 02384a60fda6f2d0 | 128 | as above, plus `v_readlane` | still 64 (several console waves) |
| 6b82bc7d22016cb5 | 1024 | as 2cc4c197ee29e291 | still 64 (several console waves) |
| 5dc3b25986622949 | 64 | LDS in a group over 32, an LDS read and write in one lane-branch region, `v_readlane` | **32** |
| 6f225b641c5b0708 | 64 | as above | **32** |

The two halves of a console wave share LDS inside branches only some lanes
take, so they cannot be split into two host waves with group barriers alone.

- `VICTORY_WAVE_WIDE=1` (off by default) gets the four single-wave groups to
  32-lane waves that validate and that D3D12 accepts. Upstream turned it off
  after runs 121-124 lost the card. Next: a live run with it on.
- The three groups of several console waves are refused by `widen` on
  purpose: a group barrier inside a branch that different console waves take
  differently would hang the group. Making that work needs the branch decided
  for the whole group (an OR over every console wave, exchanged where every
  thread runs), with the waves that would not have taken it running through
  it inactive. Not started; upstream's wave64 code changes often, so this is
  worth raising there first.
- Reproduce offline, no game needed:
  `VICTORY_HOST_MAX_WAVE=32 VICTORY_WAVE_WHY=1 victory translate <file.cdxil> --pipelines`
  (with the debug layer: Windows "Graphics Tools" installed and
  `VICTORY_D3D12_DEBUG=1`, the refusal carries D3D12's reason).

## 2. The shader cache did not depend on the host's wave size (fixed)

`shader_cache.rs` keyed translations on the executable and the translation
switches but not on the host's widest wave, which picks the form of a wave64
program. A cache shared by a 64-lane and a 32-lane GPU (this PC has both an
AMD iGPU and the NVIDIA card) could serve one's translation to the other. The
key now includes `emit::host_max_wave()`.

## 3. Pipelines made through an unimplemented creation method (open)

The release run logged 29 `a pipeline creation method that is not
implemented: the title goes without this pipeline`, from several device
slots (0x1c8, 0x218 and others). Those pipelines never exist, so whatever
they draw is missing. Not yet measured how often their draws come up.

## 4. Slow frames while loading and in the first scenes (open)

0.1 to 0.8 frames per second while textures stream in: the stats lines show
80 to 400 ms per frame preparing (texture uploads of 100 to 550 MB) and up to
800 ms on the GPU. The first run also spent minutes compiling pipelines with
an empty cache; later runs did not. Not profiled yet (Nsight Graphics and
Nsight Systems are installed for that).

## 5. Letters missing from the opening credits (open, unconfirmed)

One grab of the in-engine credits showed gaps where letters should be. It may
be the title's reveal animation. To check: the same credit at a later frame.

## 6. Packed vertex and buffer formats (fixed or open, see below)

From a review of typed buffer loads (`lower.rs`):

- **fixed**: indexed typed fetch (`tbuffer_load_format_*` with `idxen`) was
  always refused; it now folds `index * stride` into the byte offset like raw
  loads.
- **fixed**: `tbuffer_load_format_d16_*` was not recognised as d16 and wrote
  one register per channel; it now packs two 16-bit channels per register.
- **fixed**: the descriptor's DST_SEL is applied to typed loads where the
  code fixes the descriptor (built or constant).
- **open**: the console-only DXGI formats 116, 117 and 189 to 191 reach the
  host unchanged (`resource.rs` `pc_format`); 189 and 190 mean sampler
  feedback formats on PC. No run has shown the title using them.
- **open**: `buffer_load_format_*` at an attribute offset is still refused.

## 7. Test environment notes (by design)

- The shader tests need a current DXIL validator: with the 2019 SDK's
  `dxil.dll` (no SM 6.6) the barycentric tests fail and the test process
  crashes; DXC 1.8.2407's validator also wrongly fails `emit::rect`. Point
  `VICTORY_DXIL` at DXC 1.9.2609's `dxil.dll`.
- The D3D12 debug layer (`VICTORY_D3D12_DEBUG=1`) needs Windows' optional
  "Graphics Tools" feature.
