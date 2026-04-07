# Stage Findings Summary for Stem Operator Evaluation

## Scope

This document summarizes one specific stage of the project. It only covers the operator study for the first two stem layers, `E0` and `E1`.

The task is RGB optical flow estimation. The model takes two RGB frames as a 6-channel input. The deployment target is Grove Vision AI Module V2 with Ethos-U55 and Vela.

This stage does not compare the earlier model variants such as skip, ASPP, or gate combinations. It only compares candidate operators for the first two downsampling layers.

## Evaluation Setup

We used `globalgate4x_bneckeca` as the reference model. We kept all later parts of the network unchanged unless a test explicitly modified them.

The common setup was:

- input resolution: `172x224`
- Vela optimization: `Size`
- deployment metric: `SRAM peak`, `Vela inference time`, `Network%`, and `Util%`

The reference stem was:

- `E0 = 7x7 stride-2`
- `E1 = 5x5 stride-2`

The reference model gave:

- `SRAM peak = 1386.00 KiB`
- `Vela inference time = 174.657 ms`

The `SRAM peak` hotspot stayed at the final `ResizeBilinear_1`.

## Tested Stem Configurations

We tested four stem settings.

| Setting | E0 | E1 |
| --- | --- | --- |
| Original stem | `7x7 stride-2` | `5x5 stride-2` |
| `stemdilate` | `3x3 dilated -> 3x3 stride-2` | `3x3 dilated -> 3x3 stride-2` |
| `stempostdilate` | `3x3 stride-2 -> 3x3 dilated` | `3x3 stride-2 -> 3x3 dilated` |
| `e0twolayer` | `3x3 stride-2 -> 3x3 stride-1` | `5x5 stride-2` |

## Main Vela Results

Table 1 shows the direct comparison.

| Setting | Vela SRAM peak | Vela inference time | FPS | Off-chip Flash | Summary |
| --- | --- | --- | --- | --- | --- |
| Original stem | 1386.00 KiB | 174.657 ms | 5.73 | 2813.94 KiB | Reference |
| `stemdilate` | 1386.00 KiB | 198.706 ms | 5.03 | 2822.53 KiB | Clearly slower |
| `stempostdilate` | 1386.00 KiB | 190.056 ms | 5.26 | 2826.62 KiB | Better than `stemdilate`, but still slower |
| `e0twolayer` | 1386.00 KiB | 174.150 ms | 5.74 | 2814.94 KiB | Almost identical to the original stem |

Three direct observations follow from this table:

1. None of the stem changes altered the `SRAM peak`.
2. `stempostdilate` was better than `stemdilate`.
3. `e0twolayer` stayed essentially tied with the original stem.

## E0 Results

Table 2 focuses on `E0`.

| Setting | E0 operator | Network% | Util% | Notes |
| --- | --- | --- | --- | --- |
| Original stem | `7x7 stride-2` | 3.10% | 65.33% | Reference |
| `stemdilate` | `E0 dilated` | 10.32% | 69.05% | Very high cost |
| `stemdilate` | `E0 downsample` | 1.69% | 103.23% | Added after dilated conv |
| `stempostdilate` | `E0 downsample` | 0.64% | 53.74% | First step |
| `stempostdilate` | `E0 dilated` | 10.17% | 97.68% | Main cost |
| `e0twolayer` | `E0 first 3x3 stride-2` | 0.69% | 53.74% | First step |
| `e0twolayer` | `E0 second 3x3 stride-1` | 2.12% | 93.79% | Second step |

For `E0`, the combined cost is:

- original stem: `3.10%`
- `stemdilate`: `10.32% + 1.69% = 12.01%`
- `stempostdilate`: `0.64% + 10.17% = 10.81%`
- `e0twolayer`: `0.69% + 2.12% = 2.81%`

This comparison is the clearest result in this stage.

- `E0` dilation was expensive.
- Moving dilation after downsampling reduced the cost, but not enough.
- A dense two-layer `3x3` replacement for `E0` stayed competitive with the original `7x7`.

## E1 Results

Table 3 focuses on `E1`.

| Setting | E1 operator | Network% | Util% | Notes |
| --- | --- | --- | --- | --- |
| Original stem | `5x5 stride-2` | 2.89% | 95.49% | Reference |
| `stemdilate` | `E1 dilated` | 3.57% | 97.67% | Main cost |
| `stemdilate` | `E1 downsample` | 1.79% | 97.63% | Added after dilated conv |
| `stempostdilate` | `E1 downsample` | 0.93% | 97.61% | First step |
| `stempostdilate` | `E1 dilated` | 1.87% | 97.54% | Second step |

For `E1`, the combined cost is:

- original stem: `2.89%`
- `stemdilate`: `3.57% + 1.79% = 5.36%`
- `stempostdilate`: `0.93% + 1.87% = 2.80%`

This result is narrower than the `E0` result.

- `E1` pre-dilation was clearly expensive.
- `E1` post-dilation stayed close to the original `5x5`.

## Relative Latency Changes

Table 4 shows the latency change against the original stem.

| Setting | Inference time | Change vs original |
| --- | --- | --- |
| Original stem | 174.657 ms | 0 |
| `stemdilate` | 198.706 ms | `+24.049 ms`, `+13.77%` |
| `stempostdilate` | 190.056 ms | `+15.399 ms`, `+8.82%` |
| `e0twolayer` | 174.150 ms | `-0.507 ms`, `-0.29%` |

This comparison shows the hardware trend clearly:

- pre-dilation was the worst option
- post-dilation was better
- the dense two-layer `E0` stayed neutral in deployment cost

## Stage Conclusions

This stage established five facts.

1. The first two stem layers did not control the current memory peak. Every tested stem still kept `SRAM peak = 1386.00 KiB`. The hotspot stayed at the final `ResizeBilinear_1`.
2. `E0` and `E1` behaved differently. `E0` dilation was expensive. `E1` post-dilation stayed close to the original cost.
3. The order of efficiency was stable. `stempostdilate` was better than `stemdilate`.
4. The dense two-layer `E0` replacement did not slow the model down. It stayed almost identical to the original `7x7`.
5. This stage gave a deployment-side answer only. It did not determine the accuracy ranking of these stem choices.
