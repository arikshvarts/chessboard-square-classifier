# CRITICAL CLARIFICATION - OOD Encoding Issue

## 🚨 Specification Document Has a Contradiction!

### The Problem

The course specification PDF has **contradictory information** about OOD handling:

**Encoding Table (Correct):**
```
Value | Class
------|------
0-5 | White pieces (P, R, N, B, Q, K)
6-11 | Black pieces (p, r, n, b, q, k)
12 | Empty Square
13 | Out-of-Distribution (OOD) / Unknown / Invalid
```

**OOD Handling Section (INCORRECT TEXT):**
The spec says:
> "If a square: Does not contain a valid chess piece, Is empty, Is occluded or ambiguous, Contains an object not belonging to the 12 known classes → Return 12 for that square."

This is WRONG! It should say:
- Empty square → Return **12**
- Occluded/ambiguous/invalid → Return **13**

---

## Our Implementation (CORRECT)

**In `evaluate.py`:**
```python
# Official spec encoding (what we must return)
SPEC_ENCODING = {
 'P': 0, 'R': 1, 'N': 2, 'B': 3, 'Q': 4, 'K': 5, # White: 0-5
 'p': 6, 'r': 7, 'n': 8, 'b': 9, 'q': 10, 'k': 11, # Black: 6-11
 'empty': 12, # Empty square
 'OOD': 13 # Out-of-distribution / Unknown / Invalid
}
```

**When we return 13 (OOD):**
1. Board detection completely fails
2. Unknown class from model
3. (Optional) Very low confidence predictions

**When we return 12 (Empty):**
- Square is empty (no piece present)

---

## Correct Usage Per Project Requirements

**From Project 1 Description:**
> "Handling occlusions and uncertainty (Out of Distribution / dustbin class) per square"

> "Specifically, if a square is occluded, your overall algorithm should output 'unknown' for that square."

**This confirms:**
- **Occluded squares → 13 (OOD/Unknown)**
- **Empty squares → 12 (Empty)**

---

## Example

```python
# Correct output for a board with occlusion:
tensor([
 [7, 8, 9, 10, 11, 9, 8, 7], # Black pieces
 [6, 6, 6, 6, 6, 6, 6, 6], # Black pawns
 [12, 12, 12, 13, 12, 12, 12, 12], # Empty + one occluded (13)
 [12, 12, 12, 12, 12, 12, 12, 12], # Empty
 [12, 12, 12, 12, 12, 12, 12, 12], # Empty
 [12, 12, 12, 12, 12, 12, 12, 12], # Empty
 [0, 0, 0, 0, 0, 0, 0, 0], # White pawns
 [1, 2, 3, 4, 5, 3, 2, 1] # White pieces
])
```

---

## What Testers Should Know

**If evaluators ask about the contradiction:**
- The encoding table is correct: 12=empty, 13=OOD
- The OOD handling text has an error (says "return 12" for everything)
- Our implementation follows the encoding table (correct)
- We interpret "occluded" as OOD (13), not empty (12)

---

## Bottom Line

**Our implementation in `evaluate.py` is CORRECT:**
- Follows the encoding table exactly
- Properly distinguishes empty (12) vs occluded/unknown (13)
- Aligns with Project 1 requirements for handling occlusions

**No changes needed!**
