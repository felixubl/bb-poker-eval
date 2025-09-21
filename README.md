# bb-poker-eval

This package provides a compact and efficient implementation for evaluating poker hands of arbitrary size. Unlike naive approaches that enumerate all 5-card subsets (e.g. 21 for Hold’em), this evaluator works **directly on bitmask representations** of cards. As a result, it determines the strongest 5-card hand **in constant time relative to hand size**, using only bitwise operations and rank masks.

All critical routines are JIT-compiled with [Numba](https://numba.pydata.org), yielding performance close to optimized C while retaining the readability and portability of Python.

## Foreword

This is simply the result of a personal project of mine. I am not sure if it is novel, though I am sure there are more performant poker evaluators out there. I have not seen this bitboard approach and thought it was fun to try. It's pretty fast and super light. I believe this only has a 22 byte LUT footprint. Even that felt a bit ugly, but I did not find any elegant approach on how to find straights without at least some sort of mask/LUT. Sorry.

My personal goals where: 
1. Find the best 5 card hand of a 7 card hand without going over all 21 possible combinations. 
2. Do not use LUTs to achieve this. 
3. Be fast

I think I have somewhat succeeded. Even without paralellization we can evaluate at least 10M Hands per second. 

---

# Installation

This is my first finished package. No idea how to publish it so it can be easily installed. Still figuring everything out.

---

## Key Features

* **Arbitrary hand size**: Input may contain 5, 7, and up to 9 cards right. Once we reach 10 cards it can identify lesser flushes as the best hand. The evaluator automatically extracts the best 5-card equivalent. It can also evaluate hands with less than 5 cards. For example a simple pair will be identified as one pair.
* **Constant-time evaluation**: No subset enumeration; best hands are derived by algebraic manipulation of rank/suit masks.
* **Lexicographic scoring**: Every hand is mapped to a fixed 6-tuple `(category, r0, r1, r2, r3, r4)`. Comparisons reduce to tuple ordering.
* **Wheel support**: A-2-3-4-5 straights and straight flushes are normalized to `end_rank = 3`.
* **Minimal dependencies**: Requires only `numpy` and `numba`.

---

## Hand Encoding

* **Card**: Encoded as a single `int32 = (suit << 4) | rank`.

  * `suit ∈ {0,1,2,3}`
  * `rank ∈ {0,…,12}` corresponding to {2,…,A}.
* **Hand**: A `numpy.ndarray` of shape `(4,)` and dtype `uint16`.

  * Each entry is a 13-bit mask representing the ranks present in that suit.

For example, the Ace of hearts (`suit=0, rank=12`) sets bit 12 in `hand[0]`.

---

## Evaluation Categories

The evaluator returns a **6-tuple**:

```
(cat, r0, r1, r2, r3, r4)
```

* `cat` identifies the hand category (see table).
* `r0..r4` contain rank descriptors (descending order), which resolve ties.
* Tuples compare lexicographically: higher categories, then higher ranks, dominate.

| cat | Category        | Description (tie-breaking)                   |
| --- | --------------- | -------------------------------------------- |
| 8   | Straight flush  | `r0 = end_rank` (0–12; wheel → 3)            |
| 7   | Four of a kind  | `r0 = quad_rank, r1 = kicker`                |
| 6   | Full house      | `r0 = trips_rank, r1 = pair_rank`            |
| 5   | Flush           | `r0..r4 = top 5 ranks in-suit`               |
| 4   | Straight        | `r0 = end_rank` (wheel → 3)                  |
| 3   | Three of a kind | `r0 = trips_rank, r1..r2 = kickers`          |
| 2   | Two pair        | `r0 = high_pair, r1 = low_pair, r2 = kicker` |
| 1   | One pair        | `r0 = pair_rank, r1..r3 = kickers`           |
| 0   | High card       | `r0..r4 = top 5 ranks overall`               |

---

## Example: Evaluating a 7-Card Hand

```python
import numpy as np
from numba import int32

def C(suit, rank):
    return card_to_int(int32(suit), int32(rank))

hand = empty_hand()
# Texas Hold’em: Ah Kh Qh Jh Th 9c 2d
for (s, r) in [(0,12), (0,11), (0,10), (0,9), (0,8), (1,7), (2,0)]:
    add_card(hand, C(s, r))

cat, r0, r1, r2, r3, r4 = evaluate(hand)
print(cat, r0)  # -> 8, 12  (straight flush to Ace)
```

Even though 7 cards are provided, the evaluator identifies the best 5-card subset (royal flush) without enumerating `C(7,5) = 21` possibilities.

---

## Benchmarks

Once I find the time to set up something where we can also compare this to other poker evaluators I will add some benchmarks here. 

---

## Mathematical Note: Why Bitmasks Are Sufficient

Let a **hand** be represented as four 13-bit vectors

$$
h = (h_0, h_1, h_2, h_3), \quad h_s \in \{0,1\}^{13},
$$

where $h_s[r] = 1$ if rank $r$ appears in suit $s$.

Define the following derived masks:

* **Union mask** $U = h_0 \lor h_1 \lor h_2 \lor h_3$: set of all ranks present.
* **Multiplicity masks**:

  * $GE_k[r] = 1 \iff \sum_{s} h_s[r] \ge k$.

Each poker category can be characterized purely in terms of these masks:

* *Flush*: $\exists s : \|h_s\|_1 \ge 5$. Best 5 cards are simply the 5 largest ranks in that suit.
* *Straight*: $\exists r : U[r]\cdots U[r-4] = 1$. Wheel handled via fixed mask.
* *Quads, Trips, Pairs*: Detected from $GE_4, GE_3, GE_2$. Kickers are the top unused ranks from $U$.
* *Full house*: Best triple from $GE_3 \setminus GE_4$; best pair from $GE_2 \setminus GE_3$.
* *High card*: Top 5 ranks from $U$.

Since all winning 5-card subsets must satisfy these category definitions, and since the evaluation selects **maximal ranks** within each mask, the bitmask method yields the same outcome as exhaustive subset search:

$$
\max_{\substack{S \subseteq \text{hand}\\|S|=5}} \text{score}(S) 
\;=\; \text{evaluate}(h).
$$

Thus, the algorithm is both **complete** (no hand is missed) and **optimal** (always returns the maximal 5-card hand).

---

## Performance Considerations

* All evaluators are decorated with `@njit(cache=True)`.
* First call incurs compilation overhead; subsequent calls are near-C speed.
* Bitwise representations allow operations in O(1) regardless of hand size.

---

## API Summary

* **Card & hand utilities**

  * `card_to_int(suit, rank) → int32`
  * `decode_card(card) → (suit, rank)`
  * `empty_hand() → np.ndarray`
  * `add_card(hand, card) → int`

* **Main evaluation**

  * `evaluate(hand) → (cat, r0..r4)`

* **Specialized evaluators** (for diagnostics or custom rules):
  `straight_flush`, `four_kind`, `full_house`, `flush`,
  `straight`, `three_kind`, `two_pair`, `one_pair`, `high_card`.

* **Low-level helpers**:
  `popcount16`, `highest_n_from_mask`, `ge_masks_from_suits`, `has_run5`.

---

## Requirements

* Python ≥ 3.9
* `numpy`
* `numba`

---

## License

Even though I do not believe there is any commercial use here and I want others to tinker with it since there are still some areas where one could be even more performant (at least I believe so). I am not a fan of how lax the normal MIT license is.

bb-poker-eval is free software: you can redistribute it and/or modify
it under the terms of the [GNU General Public License](LICENSE) as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.


