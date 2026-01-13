### TODO List
* [ ] Add figures to the intro post
* [ ] Try to follow the same narrative of the slides aka give a historical perspective:
    FP32 -> FP16 -> BF16 -> FP8 -> MX formats -> NVFP4
* [ ] Look for more details on the Random Hadamard Transform (so far I didn't put much emphasis on that)
* [ ] Add some detail/plot on the Stochastic Rounding
* [ ] Ask if we can reuse the figures from NVIDIA or we need to redo them

---

### Suggested Edits for `00-intro.md` (Claude Review)

#### Content Fixes
- [x] **Figure `fp_00.png`**: Fix mismatch with text — change `3.1405` to `3.141` in the figure (text correctly uses 3 fractional digits)
- [x] **Line 116**: Clarify BF16 transition, reword `E5M10 -> E8M7` to explicitly state "reduced mantissa from 10 to 7 bits while expanding exponent from 5 to 8"
- [x] **Lines 195-197**: Verify ArXiv link, ID `2509.25149` is valid (paper published September 2025)

#### Missing Context
- [x] **Line ~142 (FP8 section)**: Add hardware context — mention FP8 was introduced with NVIDIA Hopper (H100) architecture
- [x] **Line 157-158**: Expand acronyms, change "Weights (FWD/BWD)" to "Weights (forward/backward pass)" for clarity
- [x] **Line 177 (MX example)**: Add clarification, note that the 4-element block is a "simplified example; real blocks use 32 elements"

#### Flow & Clarity
- [x] **Line 181 (MX → NVFP4 transition)**: Add bridging sentence, explicitly state NVFP4 builds on MX concepts with NVIDIA-specific refinements
- [x] **Lines 199-201 (2D Block Scaling)**: Explain the "why", updated with chain rule explanation from paper
- [x] **Lines 213-229 (Stochastic Rounding)**: Add intuition, include: "Over many operations, errors cancel out rather than accumulate in one direction"

#### Structural Suggestions
- [x] **Audience consistency**: Review for consistent technical depth — either add brief explanations for ML terms (gradient descent, activations) or assume expert audience throughout
- [x] **Figures 5 & 6**: Differentiate captions more clearly — one shows compute flow, one shows training flow
- [x] **Conclusion (lines 258-261)**: Strengthen with concrete performance numbers, future directions beyond NVFP4, or guidance on when to use which format
