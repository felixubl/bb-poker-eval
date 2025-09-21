import numpy as np
from numba import njit, uint16, int32


MASK13 = uint16((1 << 13) - 1)
WHEEL_MASK = uint16((1 << 12) | (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3))
RUN5_MASKS = np.array([uint16(((1 << 5) - 1) << s) for s in range(9)], dtype=np.uint16)


@njit(cache=True)
def card_to_int(suit: int32, rank: int32) -> int32:
    """suit: 0..3, rank: 0..12 (2..A)"""
    return (suit << 4) | rank


@njit(cache=True)
def decode_card(card: int32):
    """Return (suit, rank)"""
    rank = card & 0xF
    suit = (card >> 4) & 0x3
    return suit, rank


@njit(cache=True)
def empty_hand():
    return np.zeros(4, dtype=np.uint16)


@njit(cache=True)
def add_card(hand, card: int32) -> int32:
    """Set the card bit. Returns 0 if newly added, 1 if it was already present."""
    suit, rank = decode_card(card)
    bit = uint16(1 << rank)
    old = hand[suit]
    if (old & bit) != 0:
        return 1
    hand[suit] = uint16((old | bit) & MASK13)
    return 0


@njit(cache=True)
def popcount16(x: uint16) -> int32:
    """Kernighan popcount for 13-bit values"""
    c = 0
    while x:
        x &= uint16(x - 1)
        c += 1
    return c


@njit(cache=True)
def highest_n_from_mask(mask: uint16, n: int32):
    """Collect top n ranks (desc). Return a 5-int tuple filled with -1 for unused."""
    r0 = r1 = r2 = r3 = r4 = -1
    cnt = 0
    for r in range(12, -1, -1):
        if (mask >> r) & 1:
            if cnt == 0:
                r0 = r
            elif cnt == 1:
                r1 = r
            elif cnt == 2:
                r2 = r
            elif cnt == 3:
                r3 = r
            elif cnt == 4:
                r4 = r
            cnt += 1
            if cnt == n:
                break
    return r0, r1, r2, r3, r4


@njit(cache=True)
def has_run5(mask: uint16) -> int32:
    """Highest end rank of any 5-long run in mask, else -1 (wheel handled outside)."""
    for s in range(8, -1, -1):
        m = RUN5_MASKS[s]
        if (mask & m) == m:
            return s + 4
    return -1


@njit(cache=True)
def ge_masks_from_suits(h0: uint16, h1: uint16, h2: uint16, h3: uint16):
    """Return GE1, GE2, GE3, GE4 masks over ranks (columns).
    GEk bit r = 1 iff that rank appears in >=k suits.
    """
    ge1 = uint16((h0 | h1 | h2 | h3) & MASK13)
    ge4 = uint16((h0 & h1 & h2 & h3) & MASK13)
    ge2 = uint16(
        ((h0 & h1) | (h0 & h2) | (h0 & h3) | (h1 & h2) | (h1 & h3) | (h2 & h3)) & MASK13
    )
    ge3 = uint16(
        ((h0 & h1 & h2) | (h0 & h1 & h3) | (h0 & h2 & h3) | (h1 & h2 & h3)) & MASK13
    )
    return ge1, ge2, ge3, ge4


@njit(cache=True)
def straight_flush(hand):
    """Return (found, end_rank). Wheel returns 3 (5-high)."""
    best = -1
    for s in range(4):
        m = hand[s]
        end = has_run5(m)
        if end > best:
            best = end
        if best < 3 and (m & WHEEL_MASK) == WHEEL_MASK:
            best = 3
    return (best >= 0), best


@njit(cache=True)
def four_kind(hand):
    """Return (found, quad_rank, kicker)."""
    all4 = uint16(hand[0] & hand[1] & hand[2] & hand[3])
    if all4 == 0:
        return False, -1, -1
    qr = -1
    for r in range(12, -1, -1):
        if (all4 >> r) & 1:
            qr = r
            break
    ranks = uint16(hand[0] | hand[1] | hand[2] | hand[3])
    kicker_mask = uint16(ranks & ~uint16(1 << qr))
    kr, _, _, _, _ = highest_n_from_mask(kicker_mask, 1)
    return True, qr, kr


@njit(cache=True)
def full_house(hand):
    """Return (found, triple_rank, pair_rank). Uses best triple, then best pair."""
    h0, h1, h2, h3 = hand[0], hand[1], hand[2], hand[3]
    _, ge2, ge3, ge4 = ge_masks_from_suits(h0, h1, h2, h3)
    exactly3 = uint16(ge3 & ~ge4)
    pairs = uint16(ge2 & ~ge3)
    tr1 = -1
    tr2 = -1
    for r in range(12, -1, -1):
        if (exactly3 >> r) & 1:
            if tr1 == -1:
                tr1 = r
            elif tr2 == -1:
                tr2 = r
    if tr1 != -1:
        pr = -1
        for r in range(12, -1, -1):
            if (pairs >> r) & 1 and r != tr1:
                pr = r
                break
        if pr != -1:
            return True, tr1, pr
        if tr2 != -1:
            return True, tr1, tr2
    return False, -1, -1


@njit(cache=True)
def flush(hand):
    """Return (found, top5 ranks, suit) — top5 as r0..r4."""
    for s in range(4):
        m = hand[s]
        if popcount16(m) >= 5:
            r0, r1, r2, r3, r4 = highest_n_from_mask(m, 5)
            return True, r0, r1, r2, r3, r4, s
    return False, -1, -1, -1, -1, -1, -1


@njit(cache=True)
def straight(hand):
    """Return (found, end_rank)."""
    ranks = uint16(hand[0] | hand[1] | hand[2] | hand[3])
    end = has_run5(ranks)
    if end >= 0:
        return True, end
    if (ranks & WHEEL_MASK) == WHEEL_MASK:
        return True, 3
    return False, -1


@njit(cache=True)
def three_kind(hand):
    """Return (found, trips_rank, kicker1, kicker2)."""
    h0, h1, h2, h3 = hand[0], hand[1], hand[2], hand[3]
    _, _, ge3, ge4 = ge_masks_from_suits(h0, h1, h2, h3)
    trips = uint16(ge3 & ~ge4)
    if trips == 0:
        return False, -1, -1, -1
    tr = -1
    for r in range(12, -1, -1):
        if (trips >> r) & 1:
            tr = r
            break
    ranks = uint16(h0 | h1 | h2 | h3)
    kick_mask = uint16(ranks & ~uint16(1 << tr))
    k1, k2, _, _, _ = highest_n_from_mask(kick_mask, 2)
    return True, tr, k1, k2


@njit(cache=True)
def two_pair(hand):
    """Return (found, hi_pair, lo_pair, kicker)."""
    h0, h1, h2, h3 = hand[0], hand[1], hand[2], hand[3]
    _, ge2, ge3, _ = ge_masks_from_suits(h0, h1, h2, h3)
    pairs = uint16(ge2 & ~ge3)
    p1 = -1
    p2 = -1
    for r in range(12, -1, -1):
        if (pairs >> r) & 1:
            if p1 == -1:
                p1 = r
            elif p2 == -1:
                p2 = r
    if p2 == -1:
        return False, -1, -1, -1
    ranks = uint16(h0 | h1 | h2 | h3)
    kick_mask = uint16(ranks & ~uint16((1 << p1) | (1 << p2)))
    k, _, _, _, _ = highest_n_from_mask(kick_mask, 1)
    return True, p1, p2, k


@njit(cache=True)
def one_pair(hand):
    """Return (found, pair_rank, kick1, kick2, kick3)."""
    h0, h1, h2, h3 = hand[0], hand[1], hand[2], hand[3]
    _, ge2, ge3, _ = ge_masks_from_suits(h0, h1, h2, h3)
    pairs = uint16(ge2 & ~ge3)
    if pairs == 0:
        return False, -1, -1, -1, -1
    pr = -1
    for r in range(12, -1, -1):
        if (pairs >> r) & 1:
            pr = r
            break
    ranks = uint16(h0 | h1 | h2 | h3)
    kick_mask = uint16(ranks & ~uint16(1 << pr))
    k1, k2, k3, _, _ = highest_n_from_mask(kick_mask, 3)
    return True, pr, k1, k2, k3


@njit(cache=True)
def high_card(hand):
    """Return (r0..r4) top five ranks overall."""
    ranks = uint16(hand[0] | hand[1] | hand[2] | hand[3])
    return highest_n_from_mask(ranks, 5)


@njit(cache=True)
def evaluate(hand):
    """Return a 6-tuple (cat, r0, r1, r2, r3, r4) for tie-breaking."""

    # Straight flush
    found, end = straight_flush(hand)
    if found:
        return 8, end, -1, -1, -1, -1

    # Quads
    found, qr, k = four_kind(hand)
    if found:
        return 7, qr, k, -1, -1, -1

    # Full house
    found, tr, pr = full_house(hand)
    if found:
        return 6, tr, pr, -1, -1, -1

    # Flush (top-5 kickers in-suit)
    found, r0, r1, r2, r3, r4, _ = flush(hand)
    if found:
        return 5, r0, r1, r2, r3, r4

    # Straight (across suits)
    found, end = straight(hand)
    if found:
        return 4, end, -1, -1, -1, -1

    # Trips
    found, tr, k1, k2 = three_kind(hand)
    if found:
        return 3, tr, k1, k2, -1, -1

    # Two pair
    found, p1, p2, k = two_pair(hand)
    if found:
        return 2, p1, p2, k, -1, -1

    # One pair
    found, pr, k1, k2, k3 = one_pair(hand)
    if found:
        return 1, pr, k1, k2, k3, -1

    # High card
    r0, r1, r2, r3, r4 = high_card(hand)
    return 0, r0, r1, r2, r3, r4
