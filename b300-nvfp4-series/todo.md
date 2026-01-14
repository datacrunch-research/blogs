### TODO List

#### Figures
* [x] Redo Figure 1 (fp_summary.png)
* [ ] Crop almost all figures again
* [ ] NVFP4 section: describe content of figures properly

#### FP32/FP16 Section
* [ ] Mention that FP32 and FP16 are defined by IEEE 754 standard
* [ ] Double-check FP16/BF16 examples
* [ ] Add visualization for precision trade-offs (dynamic range vs precision)
* [ ] Use binades narrative (inspired by Paulius presentation) — lightweight version with visualization

#### FP8 Section
* [ ] Add missing crucial reference: [Micikevicius et al. 2022](https://arxiv.org/abs/2209.05433) — FP8 Formats for Deep Learning
* [ ] Expand DeepSeek approach details
* [ ] Highlight how DeepSeek led the way to fine-grained scaling (per-block)
* [ ] Connect to how this influenced hardware design (MX formats)

#### MX Formats Section
* [ ] Add references (OCP spec, relevant papers)

#### Other
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

---

### Paper Cross-Check Corrections (arXiv-2509.25149v1)

- [ ] **2D Block Scaling**: Add that activations/gradients use 1×16 blocks (not 16×16 like weights)
- [ ] **RHT specificity**: Clarify RHT is applied only to Wgrad (weight gradient GEMMs), not Fprop/Dgrad. Uses 16×16 matrices.
- [ ] **Stochastic Rounding**: Specify SR applies only to gradients; weights/activations use round-to-nearest-even
- [ ] **Scale format**: Change "FP8 scale" to "E4M3 scale" for precision
- [ ] **Mixed precision strategy**: Add that ~15% of layers kept in BF16 (first 2 + last 8 blocks)

---

## Plan: Improve 00-intro.md + Design Follow-up Post (Quartet POV)

### Context
- **00-intro.md**: General intro to FP precision evolution (FP32→FP4)
- **post_radicalnumbers_pt1.txt**: Radical Numerics Part 1 - NVFP4 theory (deeper, more technical)
- **post_radicalnumbers_pt2.txt**: Radical Numerics Part 2 - TransformerEngine implementation details
- **Quartet paper**: Alternative MXFP4 approach with scaling laws framework

---

### Part A: Improvements to 00-intro.md

#### 1. Add Binades Section (after line 13)
**Location**: After dynamic range/precision/accuracy definitions, before Figure 1

**Content**:
```markdown
### Measuring Dynamic Range: Binades

A **binade** is one power of 2 of dynamic range—essentially measuring how many "doublings" fit between the smallest and largest representable values:

$$\text{binades} = \log_2\left(\frac{\text{max representable}}{\text{min representable}}\right)$$

| Format | Exponent Bits | Binades | Implication |
|--------|---------------|---------|-------------|
| FP32 | 8 | ~277 | Sufficient for most computations |
| FP16 | 5 | ~40 | Sufficient for most activations |
| BF16 | 8 | ~261 | FP32 range, limited precision |
| FP8 E4M3 | 4 | ~18 | Suitable for forward pass |
| FP8 E5M2 | 5 | ~32 | Needed for backward pass |
| FP4 E2M1 | 2 | ~3.6 | Very constrained |

FP4's 3.6 binades cannot represent typical tensor value distributions, which often span 10-20 binades. This is precisely why block scaling becomes essential at 4-bit precision.
```

#### 2. Add MXFP4 vs NVFP4 Comparison (in Microscaling section ~line 180)
**Location**: After NVFP4 description, before the algorithmic interventions

**Content**:
```markdown
#### NVFP4 vs MXFP4

NVIDIA's NVFP4 and the OCP standard MXFP4 represent two approaches to 4-bit training:

| Feature | MXFP4 (OCP) | NVFP4 (NVIDIA) |
|---------|-------------|----------------|
| Data format | E2M1 | E2M1 |
| Block size | 32 elements | 16 elements |
| Scale format | E8M0 (power-of-2 only) | E4M3 (has mantissa) |
| Scale levels | 1 (block only) | 2 (block + global FP32) |

MXFP4's E8M0 scale has full FP32 dynamic range but no precision (power-of-2 only). NVFP4's E4M3 scale trades some range for precision via its 3 mantissa bits, compensating with a global FP32 scale factor. Recent work (Quartet, 2025) suggests MXFP4 can match NVFP4 accuracy with the right algorithmic recipe.
```

#### 3. Add Scaling Laws Mention (in Conclusions ~line 260)
**Brief addition**:
```markdown
Recent research has also introduced scaling law frameworks for low-precision training, showing that the impact of quantization can be modeled as reduced "parameter efficiency" (forward pass) and "data efficiency" (backward pass). This opens the door to principled trade-off analysis: under a fixed compute budget, lower precision may actually be optimal if the speedup compensates for the efficiency loss.
```

---

### Part B: Follow-up Post Structure (Quartet POV)

**Title**: "MXFP4 Training: Scaling Laws and Optimal Quantization Recipes"

**Target**: Technical depth matching post_radicalnumbers_pt2.txt but from Quartet paper perspective

#### Outline

##### 1. Introduction
- [ ] Bridge from Part 1 (NVFP4 recipe)
- [ ] Motivation: Can we do better than NVFP4? Is MXFP4 viable?
- [ ] Preview: Scaling laws framework + Quartet algorithm

##### 2. Scaling Laws for Low-Precision Training
- [ ] **Ingredient 1**: The scaling law formulation
  - $L(N, D, P_{fwd}, P_{bwd}) = \left(\frac{A}{(N \cdot \text{eff}_N)^\alpha} + \frac{B}{(D \cdot \text{eff}_D)^\beta}\right)^\gamma + E$
  - Parameter efficiency ($\text{eff}_N$) vs Data efficiency ($\text{eff}_D$)
- [ ] **Ingredient 2**: Forward/Backward trade-offs
  - Forward precision → inference cost (33% of training)
  - Backward precision → training cost (67% of training)
  - Optimality regions analysis
- [ ] PyTorch code for scaling law fitting

##### 3. The Error-Bias Trade-off
- [ ] **Forward pass**: Minimize MSE → maximize $\text{eff}_N$
  - Comparison: SR AbsMax, RTN AbsMax, LSQ, QuEST
  - QuEST wins for forward pass
- [ ] **Backward pass**: Minimize bias → maximize $\text{eff}_D$
  - Projection magnitude misalignment metric
  - Stochastic Rounding wins for backward pass
- [ ] Why different strategies for forward vs backward
- [ ] PyTorch implementation of misalignment analysis

##### 4. The Quartet Algorithm
- [ ] **Forward pass**: Hadamard + QuEST (RMSE-optimal clipping)
- [ ] **Backward pass**: Randomized Hadamard + Stochastic Rounding
- [ ] Full algorithm pseudocode
- [ ] Key insight: $\frac{3}{4}$ scaling before SR, $\frac{16}{9}$ rescaling after
- [ ] PyTorch reference implementation

##### 5. MXFP4 vs NVFP4 Deep Comparison
| Aspect | NVFP4 (TransformerEngine) | MXFP4 (Quartet) |
|--------|---------------------------|-----------------|
| Block size | 16 | 32 |
| Scale format | E4M3 + FP32 global | E8M0 |
| Forward strategy | RTN | QuEST |
| Backward strategy | SR for gradients | SR everywhere |
| RHT | Wgrad only | Forward + Backward |
| 2D scaling | Weights only | Not used |

##### 6. GPU Implementation on Blackwell
- [ ] CUTLASS 3.9 templates
- [ ] Stage 1: Fused Hadamard + Quantization + QuEST
- [ ] Stage 2: MXFP4 GEMM with `tcgen05.mma`
- [ ] Key optimizations:
  - Hadamard as 32×32 GEMM in SMEM
  - FP32→FP4 using PTX instructions
  - Scale factor computation in epilogue
- [ ] Performance: 2× over FP8, 4× over BF16 (forward)

##### 7. Custom Experiments: Linear Layer Analysis
**Hands-on experiments demonstrating FP4 training dynamics**

###### 7.1 Setup
- [ ] Single linear layer experiments (varying sizes: 768→2048, 2048→8192, 4096→14336)
- [ ] Compare: BF16 baseline, FP8 (TE), NVFP4 (TE), MXFP4 (Quartet-style)
- [ ] Metrics: forward/backward MSE, gradient alignment, throughput

###### 7.2 Quantization Error Analysis
```python
# Measure quantization error across precisions
def measure_quant_error(x, quant_fn, dequant_fn):
    qx = quant_fn(x)
    x_hat = dequant_fn(qx)
    return (x - x_hat).pow(2).mean().sqrt()  # RMSE
```
- [ ] Compare RTN vs QuEST vs SR on typical activation distributions
- [ ] Visualize error distribution across layers

###### 7.3 Gradient Flow Experiments
- [ ] Forward pass: measure how quantization error propagates
- [ ] Backward pass: measure gradient misalignment over multiple steps
- [ ] Show cumulative bias with RTN vs unbiased SR

###### 7.4 Mini Training Loop
- [ ] Small MLP or single transformer block
- [ ] Train for N steps with different precisions
- [ ] Plot loss curves showing convergence behavior
- [ ] Demonstrate the data efficiency gap mentioned in Quartet paper

##### 8. Paper Results Reference
- [ ] Accuracy comparison table (Quartet vs LUQ, HALO, Jetfire, LSS)
- [ ] Scaling law fits visualization from paper
- [ ] Optimality regions for FP4:FP4 vs FP8:FP8

##### 9. Practical Recommendations
- [ ] When to use NVFP4 vs MXFP4
- [ ] Compute budget considerations
- [ ] Data-to-parameter ratio guidelines

---

### Key Differentiators from Radical Numerics Posts

1. **Scaling laws framework** - Not covered in RN posts
2. **Error-bias trade-off analysis** - Novel contribution from Quartet
3. **QuEST for forward pass** - Different from NVFP4's RTN approach
4. **MXFP4 focus** - RN focuses on NVFP4/TransformerEngine
5. **Optimality regions** - Showing when FP4 beats FP8
6. **QuTLASS implementation** - Different from TransformerEngine

---

### For Experiments
- Need GPU with FP4 support (Blackwell) or emulation in PyTorch
- Can use TransformerEngine for NVFP4 baseline
- Quartet code available at: https://github.com/IST-DASLab/Quartet
- For emulation without hardware: implement quantization in PyTorch, run GEMMs in higher precision

---

### Implementation Order

1. [ ] **First**: Edit 00-intro.md with Part A improvements (binades, MXFP4 comparison, scaling laws mention)
2. [ ] **Second**: Create 01-quartet.md skeleton with outline
3. [ ] **Third**: Write theory sections (scaling laws, error-bias trade-off)
4. [ ] **Fourth**: Implement custom experiments with PyTorch code
5. [ ] **Fifth**: Complete algorithm and implementation sections
6. [ ] **Sixth**: Add paper results and recommendations
