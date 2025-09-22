# bb-poker-eval

A compact, bitboard-based poker hand evaluator that finds the best 5-card hand **without enumerating subsets**. It operates on suit–rank bitmasks and uses simple bitwise algebra plus tiny lookup tables to resolve straights and ranks. Implemented with [Numba](https://numba.pydata.org) for near-C speed.

The new evaluator (`optimized_bb_poker_eval.py`) introduces precomputed 8K-entry LUTs for popcount, straight endings, and highest-bit queries and packs results into a single `uint32` for very fast comparisons. The legacy evaluator (`bb_poker_eval.py`) is still included for reference and benchmarking.

---

## Foreword

This started as a personal project exploring a bitboard approach to hand evaluation. It’s not claiming to be state-of-the-art across all evaluators, but it’s fast, tiny, and fun. The current design uses \~32 KB of LUTs:

* `POPCNT13`: 8,192 × `uint8`  (≈ 8 KB)
* `STRAIGHT_END13`: 8,192 × `int16` (≈ 16 KB)
* `HIBIT13`: 8,192 × `int8` (≈ 8 KB)

The LUTs let us replace loops/branches with direct table lookups and bit-twiddling.

Personal goals were:

1. Find the best 5-card hand from ≥5 cards **without** visiting all `C(n,5)` subsets.
2. Keep memory use tiny (no giant full-hand LUTs).
3. Be fast.

Mission (mostly) accomplished.

---

# Installation

This is my first finished package; I haven’t published it yet. For now, clone the repo and use it locally (e.g., via a virtualenv). I’ll update this section once it’s on PyPI.

---

## Key Features

* **Arbitrary hand size**: Works with 5–9 cards (and fewer than 5: it reports the best category it can). At ≥10 cards, it can report a non-maximal flush since it snaps to the first flush it finds. This is fixable, but since I am essentially only using this for Texas Holdem I did not care enough about it.
* **Constant-time evaluation**: No subset enumeration; categories come from rank multiplicity masks (`GE2/GE3/GE4`), straights from a small LUT, kickers from top-bit scans.
* **Packed scoring (`uint32`)**: `(cat, r0..r4)` is packed into a single integer; higher is always better. There’s also a compatibility tuple API.
* **Tiny LUTs**: \~32 KB total.
* **Batch evaluation**: `batch_evaluate_u32(hands)` processes an `(N,4)` array of suit masks efficiently with Numba.
* **Minimal dependencies**: `numpy`, `numba`.

---

## Hand Encoding

* **Card**: `int32 = (suit << 4) | rank`

  * `suit ∈ {0,1,2,3}`
  * `rank ∈ {0,…,12}` mapping to {2,…,A}
* **Hand**: `np.ndarray(shape=(4,), dtype=np.uint16)`

  * Each entry is a 13-bit mask for ranks present in that suit.

Example: Ace of hearts (`suit=0, rank=12`) sets bit 12 in `hand[0]`.

---

## Return Format / Categories

Both evaluators ultimately define the same lexicographic hand strength:

```
(cat, r0, r1, r2, r3, r4)
```

where `cat` and `r*` (0..12) compare lexicographically. Packing uses:

```
[ cat:4 | r0:4 | r1:4 | r2:4 | r3:4 | r4:4 ]  -> uint32
```

| cat | Category        | Description (tie-breaking)                   |
| --- | --------------- | -------------------------------------------- |
| 8   | Straight flush  | `r0 = end_rank` (wheel → 3)                  |
| 7   | Four of a kind  | `r0 = quad_rank, r1 = kicker`                |
| 6   | Full house      | `r0 = trips_rank, r1 = pair_rank`            |
| 5   | Flush           | `r0..r4 = top 5 ranks in-suit`               |
| 4   | Straight        | `r0 = end_rank` (wheel → 3)                  |
| 3   | Three of a kind | `r0 = trips_rank, r1..r2 = kickers`          |
| 2   | Two pair        | `r0 = high_pair, r1 = low_pair, r2 = kicker` |
| 1   | One pair        | `r0 = pair_rank, r1..r3 = kickers`           |
| 0   | High card       | `r0..r4 = top 5 ranks overall`               |

---

## Example: Evaluate a 7-Card Hand

```python
import numpy as np
from numba import int32
from bb_poker_eval import (
    card_to_int, empty_hand, add_card, evaluate
)

def C(suit, rank):
    return card_to_int(int32(suit), int32(rank))

hand = empty_hand()
# Texas Hold’em: Ah Kh Qh Jh Th 9c 2d
for (s, r) in [(0,12), (0,11), (0,10), (0,9), (0,8), (1,7), (2,0)]:
    add_card(hand, C(s, r))

cat, r0, r1, r2, r3, r4 = evaluate(hand)
print(cat, r0)  # -> 8, 12  (straight flush to Ace)
```

You can also get the packed `uint32`:

```python
from bb_poker_eval import evaluate_u32
score = evaluate_u32(hand)  # higher is better
```

And batch-evaluate `(N, 4)` hands:

```python
from bb_poker_eval import batch_evaluate_u32
hands = np.stack([hand, hand, hand], axis=0)  # just an example
scores = batch_evaluate_u32(hands)
```

---

## Benchmarks

All numbers below are from your provided runs (single thread), after Numba compilation warms up. Hardware/OS details omitted here; results will vary by system.

### New evaluator (`optimized_bb_poker_eval.py) live timings

```
EvaluateAll5Cards   :  11 ns/op   (2,598,960 iterations)
EvaluateAll6Cards   :  12 ns/op  (20,358,520 iterations)
EvaluateAll7Cards   :  12 ns/op (100,000,000 iterations)
EvaluateAll7Cards   :  12 ns/op  (33,784,560 iterations)
EvaluateRandom5Cards:  13 ns/op   (5,000,000 iterations)
EvaluateRandom6Cards:  14 ns/op   (5,000,000 iterations)
EvaluateRandom7Cards:  14 ns/op   (5,000,000 iterations)

Number of Hands          Time Used    Hands per Second
All 5-card Hands         29.737 ms    87,399,660 /s
All 6-card Hands         239.858 ms   84,877,280 /s
All 7-card Hands         1.603 s      83,481,451 /s
Random 5-card Hands      67.278 ms    74,318,278 /s
Random 6-card Hands      68.245 ms    73,264,904 /s
Random 7-card Hands      71.436 ms    69,993,113 /s
```

### Old vs. New (7-card hands, random, best-of)

```
Generating 10,000,000 random 7-card hands...
Converting to 4×uint16 suit masks...
Warming up evaluator kernels...

old_evaluator : 1.037 s  ->  9,640,937 hands/s
new_evaluator : 0.141 s  -> 70,679,020 hands/s
```

That’s roughly a **7.3× speed-up** for the new path on this workload.


If we allow for paralell processing we can reach 200-400 M Hands per second. (At least on my machine)

---

## What Changed vs. the Old Evaluator?

* Replaced iterative rank scans with **precomputed LUTs**:

  * `POPCNT13` for `popcnt13(mask)`
  * `STRAIGHT_END13` for `straight_end_from_mask(mask)`
  * `HIBIT13` for `msb_index(mask)`
* Collapsed category selection into a **single fast path**:

  * Straight-flush detection checks each suit only if it has ≥5 cards.
  * Full house, trips, pairs use `GE2/GE3/GE4` masks directly.
* Introduced **packed scoring** (`pack_score` / `unpack_score`) to keep comparisons branch-free and cache-friendly.
* Added a **batch API** (`batch_evaluate_u32`) for evaluating many hands efficiently.
* Kept `evaluate(hand) -> (cat, r0..r4)` for **drop-in compatibility**.

Legacy helpers like `highest_n_from_mask`, `has_run5`, etc., are no longer needed in the new path.

---

## Mathematical Note (unchanged idea, faster execution)

Let the hand be four 13-bit masks `h0..h3`. Define:

* `U = h0 | h1 | h2 | h3` (all ranks present)
* `GE_k` via suitwise intersections/unions (e.g., `GE_4 = h0 & h1 & h2 & h3`)
* Straights from `STRAIGHT_END13[U]` (wheel handled by the LUT)
* Flushes when `popcnt13(hs) ≥ 5`, kickers via repeated `msb_index` on the suit mask
* Ties resolved lexicographically by `(cat, r0..r4)`

This yields the same maximum as exhaustive `C(n,5)` search, without enumerating subsets.

---

## Performance Tips

* First call to any Numba-compiled function triggers JIT compilation; subsequent calls are fast.
* Use `evaluate_u32` in inner loops; compare packed scores directly.
* For bulk work, feed `(N,4)` arrays to `batch_evaluate_u32`.

---

## API Summary

**Card & hand utilities**

* `card_to_int(suit, rank) -> int32`
* `decode_card(card) -> (suit, rank)`
* `empty_hand() -> np.ndarray[(4,), uint16]`
* `add_card(hand, card) -> int` (0 if newly added, 1 if duplicate)

**Main evaluators**

* `evaluate(hand) -> (cat, r0..r4)`  — tuple (compatibility)
* `evaluate_u32(hand) -> np.uint32`  — packed score (faster)
* `batch_evaluate_u32(hands: (N,4) uint16) -> np.ndarray[(N,), uint32]`

**Low-level (new path)**

* `popcnt13(mask: uint16) -> int32`
* `straight_end_from_mask(mask: uint16) -> int32`
* `msb_index(mask: uint16) -> int32`
* `top5_from_mask(mask: uint16) -> (r0..r4)`
* `pack_score(...) -> uint32`
* `unpack_score(uint32) -> (cat, r0..r4)`

**Legacy evaluator**
The old evaluator and its helper functions remain in the repo for comparison and reference.

---

## Requirements

* Python ≥ 3.9
* `numpy`
* `numba`

---

## License

bb-poker-eval is free software: you can redistribute it and/or modify it under the terms of the [GNU General Public License](LICENSE) as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

---

