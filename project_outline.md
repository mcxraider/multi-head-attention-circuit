# Mechanistic Interpretability Project: Reverse-Engineering Modular Arithmetic in Transformers

## Executive Summary

This project aims to deeply understand how transformer neural networks learn and execute algorithms by training a small decoder-only transformer on modular addition, then systematically reverse-engineering the computational circuits that emerge within the model. Rather than treating the transformer as a black box, you will dissect its internal mechanisms—identifying which attention heads perform which algorithmic steps, mapping information flow through layers, and discovering how the model internally represents mathematical operations.

The goal is not just to achieve high accuracy on a task, but to achieve **mechanistic understanding**: to know exactly what each component of the trained model is doing and why. This demonstrates mastery of transformer internals beyond surface-level API usage and connects to cutting-edge research in AI interpretability and safety.

---

## Project Motivation and Significance

### Why This Matters for Interviews

Most machine learning practitioners can fine-tune pre-trained models or use APIs like `transformers.AutoModel`. Far fewer can explain what's actually happening inside the model at a mechanistic level. This project showcases:

- **Deep architectural understanding**: You know what attention heads compute, how information flows through layers, and why transformers need residual connections
- **Research methodology**: You can form hypotheses, design experiments, and systematically test them
- **Practical debugging skills**: Understanding internals helps debug when models fail
- **Connection to AI safety**: Mechanistic interpretability is crucial for understanding and controlling powerful AI systems

### Why This Matters for AI Research

Mechanistic interpretability is an active research area at organizations like Anthropic and OpenAI. As AI systems become more powerful, understanding their internal reasoning becomes critical for:

- **Detecting harmful behavior**: Before deployment, not after
- **Improving architectures**: Designing models intentionally rather than accidentally
- **Debugging failures**: Understanding why models make mistakes
- **Building trust**: Explaining AI decisions to stakeholders

Your project, while small-scale, demonstrates the methodology used in cutting-edge interpretability research.

### Why Synthetic Tasks?

Real-world tasks like language modeling involve thousands of simultaneous computations. A model processing "The cat sat on the mat" is doing grammatical parsing, semantic understanding, world knowledge retrieval, and more—all at once.

Synthetic algorithmic tasks like modular addition provide a **controlled laboratory environment** where:
- You know the ground truth algorithm that should work
- The task has clear, discrete steps you can verify
- Success is unambiguous (correct answer or not)
- You can isolate specific computational mechanisms

This is like studying physics with idealized experiments (frictionless surfaces, perfect vacuums) before tackling messy real-world scenarios.

---

## The Task: Modular Addition

### Task Definition

The model will learn to perform modular arithmetic:

**Input**: "234 + 567 mod 113 ="
**Output**: "14"

Where `(234 + 567) mod 113 = 801 mod 113 = 14`

### Why Modular Addition?

This task is ideal for interpretability because:

1. **Multi-step algorithm**: Requires parsing numbers, performing addition, applying modulo—multiple distinct operations you can identify

2. **Not trivially memorizable**: With proper dataset design (explained below), the model must learn the algorithm rather than memorize answers

3. **Clear failure modes**: You can test specific algorithmic steps (can it add? can it apply modulo? does it understand place value?)

4. **Interpretable attention patterns**: You expect attention to focus on specific positions (operands, operators, modulus) in predictable ways

5. **Established difficulty**: Not too easy (single layer solves it) nor too hard (requires 50 layers)

### Task Properties

**Input properties**:
- Numbers range from 0 to 999 (up to 3 digits)
- Fixed modulus: 113 (a prime number)
- Format: "a + b mod m ="
- Maximum input length: 19 tokens

**Output properties**:
- Results range from 0 to 112
- Up to 3 digits
- Maximum output length: 3 tokens

**Algorithmic steps required**:
1. **Parse input**: Identify the two operands (a and b) and the modulus
2. **Compute addition**: Add the two numbers (may require carry operations for multi-digit addition)
3. **Apply modulo**: Determine remainder when dividing by modulus
4. **Format output**: Convert result to string of digits

The model must learn all these steps without explicit programming.

---

## Data Generation Strategy

### Dataset Size and Composition

**Total unique possible examples**: 1,000 × 1,000 = 1,000,000 combinations (numbers 0-999)

**Your dataset**:
- **Training**: 100,000 examples (10% of possible space)
- **Validation**: 10,000 examples
- **Test (In-Distribution)**: 10,000 examples
- **Test (Out-of-Distribution)**: 10,000 examples with numbers 1000-1999

Using only 10% of the possible space ensures the model cannot simply memorize all examples and must generalize by learning the underlying algorithm.

### Stratified Sampling by Difficulty

Rather than uniform random sampling, you'll deliberately balance difficulty levels:

**Bucket 1: No Wraparound (30% of training data)**
- Cases where `a + b < 113` (sum is less than modulus)
- Examples: "5 + 10 mod 113 =" → "15", "50 + 30 mod 113 =" → "80"
- Tests if model can perform basic addition
- Establishes baseline capability

**Bucket 2: Small Wraparound (40% of training data)**
- Cases where `113 ≤ a + b < 226` (sum wraps once)
- Examples: "60 + 80 mod 113 =" → "27", "100 + 100 mod 113 =" → "87"
- Core case requiring modulo operation
- Primary test of algorithmic understanding

**Bucket 3: Large Wraparound (20% of training data)**
- Cases where `a + b ≥ 226` (sum wraps multiple times or is very large)
- Examples: "500 + 600 mod 113 =" → "5", "999 + 999 mod 113 =" → "85"
- Harder generalization test
- Reveals if model truly understands modulo vs using shortcuts

**Bucket 4: Edge Cases (10% of training data)**
- Zero handling: "0 + 5 mod 113 =", "50 + 0 mod 113 ="
- Boundary cases: "113 + 0 mod 113 =" → "0", "56 + 57 mod 113 =" → "0"
- Single-digit numbers: "1 + 1 mod 113 =" → "2"
- Maximum values: "999 + 999 mod 113 ="
- Tests robustness and edge case handling

### Why This Distribution?

This stratification ensures:
- The model sees diverse difficulty levels during training
- You can later analyze which difficulty levels the model handles well
- The model can't "cheat" by only learning simple patterns
- You have clear test cases for each algorithmic component

### Data Quality Requirements

**Format consistency**:
- All examples use identical spacing: "a + b mod m ="
- No leading zeros: "5" not "05" or "005"
- Consistent vocabulary: only digits, "+", "mod", "=", and spaces

**Verification**:
- No duplicate examples in any split
- Train/validation/test sets are completely non-overlapping
- All arithmetic is verified correct: `(a + b) mod m` computed accurately
- All examples are within valid token length limits

**Out-of-Distribution test sets**:
- Numbers outside training range (1000-1999) to test numeric generalization
- Different modulus values (mod 97, mod 127) to test algorithmic generalization
- These OOD tests reveal whether the model learned the algorithm or just pattern-matched the training distribution

---

## Model Architecture

### Architecture Choice: Decoder-Only Transformer

You will implement a **decoder-only (GPT-style) transformer** rather than encoder-decoder or encoder-only.

**Rationale for decoder-only**:

1. **Simplicity**: Single unified architecture with only one type of attention (self-attention), making analysis straightforward

2. **Clear information flow**: Information flows strictly left-to-right due to causal masking, creating a clear computational pipeline to trace

3. **Sequential processing**: The model processes tokens one by one, making algorithmic steps more visible than in bidirectional architectures

4. **Interpretability precedent**: Most mechanistic interpretability research uses decoder-only models because their simpler structure makes circuit discovery more tractable

5. **Task appropriateness**: The task is sequential (read problem left-to-right, generate answer), matching decoder-only's strengths

**Why not encoder-decoder**:
- Adds complexity with three attention types (encoder self-attention, decoder self-attention, cross-attention)
- Information flow becomes unclear: does computation happen in encoder, decoder, or cross-attention?
- Circuit discovery requires analyzing many more pathways
- No clear benefit for this task (input and output are in the same "language"—arithmetic)

**Why not encoder-only**:
- Not designed for generation tasks
- Bidirectional attention obscures sequential reasoning
- Awkward for variable-length outputs

### Model Specifications

**Core parameters**:
- **Vocabulary size**: ~20 tokens (digits 0-9, operators, special tokens)
- **Embedding dimension**: 128
- **Number of layers**: 4
- **Attention heads per layer**: 8
- **Total attention heads**: 32 (across all layers)
- **Feed-forward expansion ratio**: 4× (128 → 512 → 128)
- **Maximum sequence length**: 32 tokens
- **Total parameters**: ~1.5 million

**Architecture structure**:
```
Input tokens
    ↓
Token Embedding + Positional Encoding
    ↓
[Layer 1: Multi-head Self-Attention + FFN]
    ↓
[Layer 2: Multi-head Self-Attention + FFN]
    ↓
[Layer 3: Multi-head Self-Attention + FFN]
    ↓
[Layer 4: Multi-head Self-Attention + FFN]
    ↓
Output projection (vocabulary logits)
    ↓
Generated tokens
```

Each layer contains:
- Multi-head causal self-attention (8 heads)
- Layer normalization (pre-norm configuration for training stability)
- Position-wise feed-forward network
- Residual connections around both sublayers

### Why These Specifications?

**Embedding dimension (128)**:
- Not too small (64 would lack expressiveness)
- Not too large (256+ is overkill and harder to interpret)
- 128 dimensions provide enough "feature space" for representing arithmetic operations

**4 Layers**:
- Expected computational decomposition:
  - **Layer 1**: Token-level processing (identify digits, operators, special characters)
  - **Layer 2**: Number-level processing (combine digits into full numbers, identify operands)
  - **Layer 3**: Operation execution (perform addition and modulo)
  - **Layer 4**: Output construction (format and generate result digits)
- Not 2 layers (insufficient depth for multi-step reasoning)
- Not 6+ layers (unnecessarily deep, spreads computation diffusely, harder to interpret)

**8 heads per layer (32 total)**:
- Enough heads for specialization (different heads can implement different sub-operations)
- Few enough to analyze individually in reasonable time
- Each head has dimension 128/8 = 16, sufficient for representing relevant patterns
- Research shows 8-12 heads per layer is effective for arithmetic tasks

**1.5M parameters**:
- Small enough to train quickly (iterate rapidly during development)
- Small enough to fully analyze (you can examine every component)
- Large enough to solve the task (expected 95%+ accuracy)
- The "interpretability sweet spot": not toy-sized, not production-scale

### Causal Self-Attention Mechanism

Each attention head computes:
1. **Queries (Q)**, **Keys (K)**, and **Values (V)** from input embeddings via learned weight matrices
2. **Attention scores**: Q·K^T / √(head_dimension)
3. **Causal mask**: Forces each position to only attend to current and previous positions (no "looking ahead")
4. **Attention weights**: Softmax of masked scores
5. **Output**: Weighted combination of Values

The causal mask is crucial—it creates the left-to-right information flow that makes sequential computation visible.

### Tokenization Strategy

**Character-level tokenization**: Each character is a token

**Vocabulary**:
- Digits: '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
- Operators: '+', '='
- Keywords: 'm', 'o', 'd' (for "mod")
- Separator: ' ' (space)
- Special: '<pad>' (padding token)
- Optional: '<start>', '<end>' (sequence markers)

**Total vocabulary**: ~15-20 tokens

**Why character-level**:
- Simple to implement (direct character-to-integer mapping)
- No complex tokenization algorithms needed (no BPE, no WordPiece)
- Natural for arithmetic (digits and operators are atomic symbols)
- Model must learn place value and number composition (interesting to observe)

**Encoding example**:
- Input string: "234 + 567 mod 113 ="
- Token sequence: ['2','3','4',' ','+','', '5','6','7',' ','m','o','d',' ','1','1','3',' ','=']
- Integer IDs: [2, 3, 4, 10, 11, 10, 5, 6, 7, 10, 12, 13, 14, 10, 1, 1, 3, 10, 15]

No pre-trained tokenizer needed—you build a simple mapping dictionary yourself.

---

## Training Strategy

### Training Objectives

**Primary objective**: Next-token prediction (standard autoregressive language modeling)

Given input sequence "234 + 567 mod 113 =", the model predicts:
- After seeing "234 + 567 mod 113 =", predict "1"
- After seeing "234 + 567 mod 113 = 1", predict "4"
- Sequence complete

**Loss function**: Cross-entropy loss on predicted tokens vs. ground truth

### Training Configuration

**Optimization**:
- Optimizer: Adam
- Learning rate: 3×10^-4 (standard for small transformers)
- Learning rate schedule: Linear warmup (1000 steps) followed by cosine decay
- Batch size: 64-128 examples
- Gradient clipping: Maximum gradient norm of 1.0 (prevents instability)
- Weight decay: 0.01 (light regularization)

**Regularization**:
- Dropout: 0.1 in attention and feed-forward layers
- No data augmentation (want clean, consistent examples)
- Early stopping based on validation loss

**Training duration**:
- Expected convergence: 10-50 epochs
- Total training examples seen: 1-5 million (10-50 passes through 100K training set)
- Training time: Hours on single GPU, not days

### Success Metrics

**Accuracy targets**:
- Training accuracy: 99%+ (near-perfect on seen examples)
- Validation accuracy: 95%+ (good generalization)
- Test accuracy: 95%+ (robust to new examples)

**What these metrics mean**:
- Small gap between train and validation (2-4%): Model is generalizing, not memorizing
- High absolute accuracy (95%+): Model has learned the algorithm
- Large gap (>10%): Overfitting; need more data or regularization

**Diagnostic metrics**:
- Accuracy by difficulty bucket (no-wrap vs. wrap vs. edge cases)
- Accuracy by digit length (1-digit vs. 2-digit vs. 3-digit numbers)
- Loss curve smoothness (jagged = unstable training)
- Gradient norms per layer (checking for vanishing/exploding gradients)

### Expected Training Behavior

**Early training (epochs 1-5)**:
- Model learns basic token patterns
- Accuracy rises from random (5% for 20 tokens) to ~40-60%
- Learns to output numbers, not random characters

**Middle training (epochs 6-20)**:
- Model learns addition
- Accuracy rises to 70-85%
- May struggle with wraparound cases initially

**Late training (epochs 21-50)**:
- Model learns modulo operation
- Accuracy rises to 95%+
- Error rate drops on hard cases

**Potential issues and solutions**:
- Model plateaus at 85%: Increase model size or adjust learning rate
- Loss spikes periodically: Reduce learning rate or increase gradient clipping
- Validation accuracy decreases while training increases: Overfitting; increase dropout or regularization

---

## Mechanistic Interpretability Analysis

This is the core of your project—understanding what the trained model actually does.

### Phase 1: Attention Pattern Visualization

**What you'll do**:
For each of the 32 attention heads (8 heads × 4 layers), extract and visualize attention weight matrices.

**Attention heatmap structure**:
- Rows: Query positions (tokens in the sequence)
- Columns: Key positions (positions each query attends to)
- Color intensity: Attention weight (0 to 1)
- One heatmap per attention head

**What you're looking for**:

**Pattern 1: Positional patterns**
- **Diagonal pattern**: Head attends to adjacent tokens (local context)
- **Vertical lines**: Head attends to specific fixed positions (like the "=" or "mod" token)
- **Horizontal lines**: Specific query position attends broadly (gathering information)

**Pattern 2: Semantic patterns**
- Heads that focus on digits
- Heads that focus on operators
- Heads that connect operands to the modulus

**Pattern 3: Compositional patterns**
- Early layers: Local, position-based attention
- Middle layers: Number-level attention (attending across full numbers)
- Late layers: Result-copying attention

**Example hypothesized findings**:
- "Head 1.3 creates a diagonal pattern—it's tracking adjacent tokens to identify digit sequences"
- "Head 2.5 shows vertical attention to the '+' symbol—it's identifying the operation type"
- "Head 3.2 attends from the '=' token back to both operands—it's gathering the numbers to add"

### Phase 2: Activation Patching (Causal Intervention)

**Concept**: Surgically modify activations at specific locations to determine their causal role.

**Method**:
1. Run model on Input A: "23 + 45 mod 113 =" → "68"
2. Run model on Input B: "56 + 78 mod 113 =" → "21"
3. **Intervention**: Run Input A again, but replace the output of Head 2.3 with its activation from Input B
4. **Observe**: Does the final answer change? How much?

**Interpretation**:
- **No change**: Head 2.3 doesn't affect this computation (might be unused or redundant)
- **Dramatic change**: Head 2.3 is critical for the computation
- **Specific change pattern**: If output becomes more like Input B's answer, Head 2.3 is carrying information about operands

**Systematic approach**:
- Patch each head individually (32 experiments per input pair)
- Patch combinations of heads (identify which heads work together)
- Patch at different layers (understand information flow through depth)
- Test on diverse inputs (does importance vary by difficulty?)

**What you'll discover**:
- **Critical heads**: Removing them breaks the model
- **Redundant heads**: Removing them has no effect (interesting—why did the model learn them?)
- **Task-specific heads**: Only matter for certain input types (e.g., only activate for wraparound cases)
- **Information pathways**: Which heads feed into which in later layers

**Example findings**:
- "Patching Head 3.1 changes the predicted answer to match the wrong input—this head is computing the sum"
- "Patching Head 4.2 only affects digit formatting, not the actual value—it's handling output structure"
- "Patching Heads 2.3 and 3.5 together completely breaks modulo computation, but patching either alone has partial effect—they're working together"

### Phase 3: Ablation Studies

**Concept**: Permanently remove components and measure performance degradation.

**Types of ablation**:

**1. Head ablation (zero-out)**
- Set all outputs of a specific attention head to zero
- Retrain model without this head (or test frozen model)
- Measure accuracy drop

**2. Mean ablation**
- Replace head's output with its mean activation across the dataset
- Less disruptive than zero ablation
- Better isolates head's contribution

**3. Layer ablation**
- Remove entire layers
- Tests if model can work with fewer layers (depth necessity)

**What you'll measure**:
- Overall accuracy degradation
- Accuracy by difficulty bucket (does removing this head only affect hard cases?)
- Specific error patterns (does it now fail modulo but still add correctly?)

**Expected findings**:
- Some heads are critical: Removing them drops accuracy from 95% to 60%
- Some heads are redundant: Removing them drops accuracy by only 1-2%
- Some heads are specialized: Removing them only affects specific input types

**Example analysis**:
"After ablating Head 3.4, accuracy on no-wraparound cases stays at 98%, but accuracy on wraparound cases drops from 95% to 40%. This head is specifically responsible for the modulo operation, not addition."

### Phase 4: Logit Attribution

**Concept**: Decompose the final output into contributions from each component.

**Mathematical insight**:
Due to transformer's linear structure (residual connections + linear transformations), the final output logit can be written as a sum:

Logit = Contribution(Embedding) + Contribution(Head 1.1) + Contribution(Head 1.2) + ... + Contribution(FFN 4)

**What you'll compute**:
For a specific prediction (e.g., predicting token "6" as first digit of "68"):
- How much did each attention head push toward "6"?
- How much did each FFN layer contribute?
- Were there components that pushed toward wrong answers?

**Visualization**:
- Bar chart showing contribution of each component to the correct prediction
- Identify which heads are "doing the work"
- Identify which heads might be making errors that are corrected later

**Example findings**:
- "Head 3.1 contributes +8.2 to the logit for '6', the largest contribution"
- "Head 2.4 contributes -1.3, slightly pushing toward wrong answer"
- "FFN Layer 3 contributes +2.1, suggesting it does post-processing after attention computes the sum"

### Phase 5: Feed-Forward Network Analysis

**What FFN layers do**:
While attention moves information around, FFN layers compute features and transformations.

**Analysis techniques**:

**1. Neuron activation analysis**
- Which neurons in FFN activate strongly on specific input patterns?
- Identify "feature detectors"

**Example**: "Neuron 47 in FFN Layer 2 activates strongly (value > 5.0) whenever the input contains a two-digit number in the first operand position. It seems to be detecting 'first number is two digits'."

**2. Direction analysis in activation space**
- FFN creates a high-dimensional activation space
- Find interpretable "directions" in this space
- Example: "The direction (0.7, -0.3, 0.5, ...) in FFN Layer 3 activation space correlates with 'sum exceeds modulus'"

**3. Feature visualization**
- For each neuron, collect inputs that maximally activate it
- Identify what pattern it's detecting

**Example findings**:
- "Neuron 113 activates on inputs where first operand > second operand"
- "Neuron 28 activates on edge cases (zero operands, boundary cases)"
- "The direction corresponding to 'needs modulo wrap' is learned in FFN Layer 3"

### Phase 6: Circuit Discovery and Synthesis

**Goal**: Combine all analysis into a coherent **circuit diagram** showing computational flow.

**What a circuit includes**:
- Specific attention heads involved in the computation
- What each head does (high-level function)
- How information flows between heads
- Which FFN layers contribute
- Clear pathway from input to output

**Example circuit for modular addition**:

```
INPUT: "234 + 567 mod 113 ="

Layer 1:
  Head 1.1: "Position Tracker" - tracks token positions
  Head 1.2: "Digit Identifier" - identifies digit tokens
  Head 1.3: "Operator Locator" - finds '+' and 'mod' tokens

Information flows to Layer 2 ↓

Layer 2:
  Head 2.1: "First Number Parser" - combines digits into first operand (234)
  Head 2.3: "Second Number Parser" - combines digits into second operand (567)
  Head 2.4: "Modulus Extractor" - identifies modulus value (113)

Information flows to Layer 3 ↓

Layer 3:
  Head 3.1: "Adder" - computes 234 + 567 = 801
  Head 3.3: "Modulo Computer" - computes 801 mod 113 = 14
  FFN 3: "Wraparound Detector" - determines if wraparound occurred

Information flows to Layer 4 ↓

Layer 4:
  Head 4.1: "Result Copier" - copies result value to output
  Head 4.2: "Digit Formatter" - formats as "14"

OUTPUT: "14"
```

**Verification of circuit**:
- Keep only components in the circuit, remove all others → accuracy should stay high
- Remove any component in the circuit → accuracy should drop
- Manually trace circuit on new examples → should predict model behavior

**The "story"**:
You can now explain: "The model solves modular addition by first parsing the input in Layer 1, extracting operands and modulus in Layer 2, performing arithmetic in Layer 3, and formatting output in Layer 4. Specifically, Head 3.1 does the addition, and Head 3.3 applies the modulo operation."

---

## Visualization Deliverables

### Visualization 1: Attention Heatmap Gallery

**Format**: Grid of heatmaps, one per attention head

**Structure**:
- 4 rows (one per layer)
- 8 columns (one per head)
- Each cell shows attention pattern for that head
- Color scale: white (no attention) to dark red (high attention)
- Axes labeled with actual tokens from an example

**Purpose**: High-level overview of what each head is doing

**Tool**: Matplotlib or Plotly for creating heatmaps

### Visualization 2: Interactive Attention Explorer

**Format**: Web-based interactive tool

**Features**:
- User inputs an example: "234 + 567 mod 113 ="
- Model processes it
- User can:
  - Select which layer to view
  - Select which head to view
  - See attention pattern for that head on this input
  - Click on heads to see their "role" (your interpretation)
  - Step through layers one by one

**Purpose**: Explore how model processes specific examples

**Tool**: Gradio (Python library, creates web interface automatically)

**Example user flow**:
1. Enter "999 + 1 mod 113 ="
2. Click "Layer 3"
3. Click "Head 3.3"
4. See heatmap showing strong attention from '=' back to '1' and '113'
5. Read annotation: "Head 3.3: Modulo computer - identifies when wraparound needed"

### Visualization 3: Circuit Diagram

**Format**: Hand-drawn or tool-generated flowchart

**Content**:
- Boxes representing key components (heads, FFN layers)
- Arrows showing information flow
- Labels explaining what each component computes
- Color-coding by layer or function type

**Purpose**: Communicate your understanding of the full computational pathway

**Tool**: Draw by hand in PowerPoint/Google Slides, or use diagramming tools like Excalidraw, Figma, draw.io

**Quality**: Should be publication-quality—clean, professional, understandable without explanation

### Visualization 4: Activation Flow Animation

**Format**: Video or animated GIF showing step-by-step processing

**Content**:
- Start with input tokens displayed
- Show Layer 1 attention patterns activating
- Highlight which heads are "active"
- Show information flowing to Layer 2
- Continue through all layers
- End with output being generated

**Duration**: 30-60 seconds

**Purpose**: Demonstrate dynamic flow of computation

**Tool**: Video editing software, or export slides as video, or create programmatically with matplotlib animation

### Visualization 5: Ablation Impact Chart

**Format**: Bar chart or heatmap

**Content**:
- X-axis: Each attention head (1.1, 1.2, ... 4.8)
- Y-axis: Accuracy drop when head is ablated
- Color: Red for critical heads, yellow for moderate, green for redundant

**Purpose**: Quickly show which components matter most

**Additional version**: Heatmap showing accuracy drop by (head, difficulty bucket) to see which heads matter for which cases

---

## Expected Findings and Hypotheses

### Hypothesis 1: Layer-wise Specialization

**Prediction**: Each layer will specialize in different aspects of computation:
- Layer 1: Token-level pattern recognition
- Layer 2: Number parsing and operand extraction
- Layer 3: Arithmetic operations
- Layer 4: Output formatting

**How to test**:
- Visualize attention patterns by layer (should see increasing abstraction)
- Ablate entire layers (should see specific failure modes)
- Analyze what FFN neurons in each layer detect

**Expected evidence**: "Neurons in Layer 1 FFN activate on simple patterns like 'is digit'. Neurons in Layer 3 FFN activate on complex patterns like 'sum exceeds modulus'."

### Hypothesis 2: Specialized Addition and Modulo Heads

**Prediction**: Distinct attention heads will implement addition vs. modulo operations.

**How to test**:
- Ablate individual heads, check if accuracy drops only on wraparound cases (modulo) or all cases (addition)
- Activation patching with inputs that differ in whether they need modulo
- Logit attribution showing which heads contribute when

**Expected evidence**: "Head 3.1 is critical for all examples (does addition). Head 3.3 is only critical for wraparound cases (does modulo)."

### Hypothesis 3: Positional Encoding Utilization

**Prediction**: The model will use positional information to identify digit positions (hundreds, tens, ones places).

**How to test**:
- Analyze attention from result tokens back to input
- Check if specific heads consistently attend to specific positions
- Test on numbers with different digit lengths

**Expected evidence**: "Head 2.1 shows strong positional bias—when attending to form the first number, it attends to positions 0,1,2 (where first number's digits are) regardless of actual digit values."

### Hypothesis 4: Redundancy and Backup Circuits

**Prediction**: The model may learn redundant heads or multiple pathways for the same computation (for robustness).

**How to test**:
- Ablate multiple heads simultaneously
- Look for heads with similar attention patterns
- Check if removing one head causes another to "activate more"

**Expected evidence**: "Heads 2.2 and 2.5 both parse the second operand. Ablating either alone drops accuracy by 3%, but ablating both drops it by 40%—they're backups for each other."

### Hypothesis 5: Grokking-style Sudden Learning

**Prediction**: The model may show "grokking" behavior—suddenly learning the algorithm after memorizing for many epochs.

**How to test**:
- Plot training and validation accuracy over time
- Look for sudden jumps in validation accuracy
- Track when attention patterns become interpretable

**Expected evidence**: "For epochs 1-15, validation accuracy stayed at 75% while training accuracy reached 99% (memorization). At epoch 17, validation accuracy suddenly jumped to 92% (algorithm learning). Attention patterns became clearly interpretable only after epoch 17."

### Hypothesis 6: Failure Mode Patterns

**Prediction**: When the model fails, it will fail in systematic ways revealing algorithmic misunderstandings.

**How to test**:
- Collect all examples where model fails
- Categorize error types (wrong addition? wrong modulo? formatting error?)
- Identify which heads are misbehaving in failure cases

**Expected evidence**: "95% of errors occur on large wraparound cases (sum > 226). In these cases, Head 3.3's attention pattern is diffuse rather than focused, suggesting it hasn't learned to handle multiple wraparounds."

---

## Out-of-Distribution Testing

### Purpose

Test whether the model learned the **algorithm** (generalizes to new cases) or **memorized patterns** (fails on distribution shifts).

### Test Set 1: Larger Numbers (OOD-Numeric)

**Design**: Numbers from 1000-1999, still mod 113

**Example**: "1234 + 1567 mod 113 =" → "81"

**What this tests**: Can the model handle 4-digit numbers when trained on up to 3-digit?

**Expected outcomes**:
- **Strong algorithm learning**: Accuracy stays high (90%+) because place-value algorithm generalizes
- **Weak algorithm learning**: Accuracy drops significantly because model is pattern-matching digit lengths

### Test Set 2: Different Modulus (OOD-Operation)

**Design**: Numbers 0-999, but use mod 97 or mod 127 instead of mod 113

**Example**: "234 + 567 mod 97 =" → "24"

**What this tests**: Did model learn general modular arithmetic, or is it specific to mod 113?

**Expected outcomes**:
- **General algorithm**: Accuracy stays high because modulo operation works for any modulus
- **Specific memorization**: Accuracy drops because model "hardcoded" for mod 113

### Test Set 3: Extreme Edge Cases

**Design**: Cases not in training distribution

**Examples**:
- Very large operands: "999 + 999 mod 113 ="
- Zero cases: "0 + 0 mod 113 =", "1000 + 0 mod 113 ="
- Boundary: "10000 + 0 mod 113 ="

**What this tests**: Robustness to unusual inputs

**Expected outcomes**: Model likely fails on cases very different from training (expected and acceptable)

### Test Set 4: Adversarial Inputs

**Design**: Inputs specifically designed to exploit potential shortcuts the model might use

**Example shortcut**: "Model might look only at the last digit to determine modulo result"

**Adversarial test**: Pairs like "123 + 456 mod 113" vs. "223 + 356 mod 113" (same last digits, different answers)

**What this tests**: Whether model is using shortcuts vs. full computation

**Analysis**: These tests inform your circuit understanding—if model fails in specific patterns, you can identify which shortcut it learned.

---

## Project Timeline and Milestones

### Week 1: Setup and Data Generation
**Goals**:
- Implement data generation script with stratified sampling
- Generate 120,000 examples (100K train, 10K val, 10K test)
- Create OOD test sets
- Verify data quality (no duplicates, correct arithmetic, proper formatting)
- Implement character-level tokenizer
- Test tokenization pipeline

**Deliverable**: Dataset files ready for training

### Week 2: Model Implementation and Initial Training
**Goals**:
- Implement decoder-only transformer architecture
- Configure training loop with proper optimization settings
- Run initial training to verify model can learn the task
- Achieve 95%+ accuracy on training and validation sets
- Debug any training issues (learning rate, gradient explosions, etc.)

**Deliverable**: Trained model checkpoint achieving target accuracy

### Week 3: Attention Pattern Analysis
**Goals**:
- Extract attention weights from all 32 heads
- Create attention heatmap visualizations
- Manually examine patterns across many examples
- Form initial hypotheses about what each head does
- Identify heads that show clear, interpretable patterns
- Identify heads that seem unused or redundant

**Deliverable**: Annotated attention heatmap gallery with initial interpretations

### Week 4: Circuit Discovery
**Goals**:
- Implement activation patching infrastructure
- Run systematic patching experiments (32 heads × multiple input pairs)
- Conduct ablation studies (remove heads, measure impact)
- Perform logit attribution analysis
- Synthesize findings into circuit diagram
- Verify circuit by testing predictions on new examples

**Deliverable**: Complete circuit diagram with evidence-backed claims

### Week 5: Visualization and Documentation
**Goals**:
- Build interactive Gradio tool for attention exploration
- Create circuit diagram (polished version)
- Optionally create animated visualization
- Write comprehensive README documenting:
  - Project motivation
  - Methodology
  - Key findings
  - How to use the interactive tool
- Prepare presentation materials for interviews
- Create 5-minute video walkthrough (optional but impressive)

**Deliverable**: Complete project portfolio ready to share

---

## Interview Presentation Strategy

### Opening Statement (30 seconds)

"I wanted to deeply understand how transformers actually work internally, beyond just using APIs. So I trained a decoder-only transformer on modular arithmetic and then reverse-engineered the computational circuits it learned. I discovered that the model decomposes the task across layers: Layer 1 identifies tokens, Layer 2 parses numbers, Layer 3 performs arithmetic, and Layer 4 formats output. I verified this through activation patching and ablation studies."

### Key Points to Emphasize

**1. Systematic methodology**
- Not just training and hoping
- Hypothesis-driven experimentation
- Multiple validation techniques (patching, ablation, logit attribution)

**2. Concrete findings**
- Specific head functions: "Head 3.1 implements the addition operation"
- Evidence-based: "I verified this by patching—swapping its activation changes the computed sum"
- Surprising insights: "The model learned a distributed counting mechanism I didn't expect"

**3. Deep technical understanding**
- Can explain attention mechanisms mathematically
- Can draw computational graph on whiteboard
- Can predict model behavior on new inputs

### Questions You'll Excel At

**Q: "How does multi-head attention work?"**
A: "Let me give you a concrete example from my project. When computing modular addition, Head 2.1 implements number parsing by computing queries from the current position and keys from previous positions. The QK dot product is high when previous positions contain digits of the same number, so attention weights aggregate those digits. The OV circuit then copies and combines them. I can show you the actual attention patterns..."

**Q: "Why do transformers need multiple layers?"**
A: "From my experiments, I found you need at least 3-4 layers for modular addition. Layer 1 does token-level identification, Layer 2 does number-level parsing, Layer 3 executes operations. I verified this is a hard lower bound by training 2-layer models—they plateau at 75% accuracy because they can't do both parsing and computation."

**Q: "How would you debug a transformer that's not learning?"**
A: "I'd use techniques from my interpretability project. First, visualize attention patterns—are heads specializing or staying random? Second, use activation patching to find which components are actually used. Third, check if architecture has enough capacity for the required circuit. Fourth, verify the data allows learning the algorithm vs. memorization."

### Demo During Interview

If possible, bring laptop and show:
1. **Interactive tool**: "Let me show you how the model processes this example..."
2. **Attention visualizations**: "See how Head 3.1 attends from the equals sign back to both operands?"
3. **Circuit diagram**: "Here's the complete computational pathway I discovered..."

This makes your understanding tangible and memorable.

---

## Extensions and Future Directions

### Extension 1: Multi-Task Learning

**Idea**: Train single model on multiple arithmetic operations

**Implementation**:
- Add task tokens: `[ADD] 23 + 45 mod 113`, `[MULT] 23 * 45 mod 113`, `[SUB] 45 - 23 mod 113`
- Train on mixture of tasks

**Research question**: Do different tasks share circuits or learn separate specialized heads?

**Expected finding**: "Heads 1.x and 2.x are shared across all tasks (general number parsing). Heads 3.x are task-specific (Head 3.1 for addition, Head 3.2 for multiplication)."

### Extension 2: Emergence During Training

**Idea**: Track circuit formation over training

**Implementation**:
- Save model checkpoints every 500 steps
- Analyze attention patterns at each checkpoint
- Identify when interpretable circuits emerge

**Research question**: Do circuits form gradually or suddenly? When does "grokking" occur?

**Expected finding**: "Attention patterns remain random until epoch 12, then Head 2.1 suddenly snaps into the number-parsing pattern. This coincides with validation accuracy jump from 70% to 90%."

### Extension 3: Minimal Circuit Discovery

**Idea**: Find smallest model that can solve task

**Implementation**:
- Train models of varying sizes (2 layers vs. 3 vs. 4)
- Train with varying head counts (4 vs. 8 vs. 12 per layer)
- Identify minimum configuration for 95% accuracy

**Research question**: What's the computational complexity lower bound?

**Expected finding**: "You need at least 3 layers and 6 total heads. With 2 layers or 4 heads, maximum achievable accuracy is 82% regardless of training time."

### Extension 4: Comparative Architecture Study

**Idea**: Compare decoder-only to encoder-decoder and LSTM

**Implementation**:
- Train LSTM, encoder-decoder transformer, and decoder-only on same data
- Analyze what each architecture learns

**Research question**: Do different architectures learn same or different algorithms?

**Expected finding**: "LSTM learns sequential algorithm (processes left-to-right only). Encoder-decoder splits work between encoder (parse) and decoder (compute). Decoder-only is most efficient (does both in unified way)."

### Extension 5: Adversarial Circuit Breaking

**Idea**: Design inputs that break specific heads

**Implementation**:
- Identify what each head does
- Design inputs that should confuse that head
- Test if model fails as predicted

**Research question**: How robust is the learned circuit?

**Expected finding**: "Head 2.1 assumes operands are 2-3 digits. Single-digit numbers with extra spaces break its attention pattern, causing 60% accuracy drop."

---

## Deliverables Summary

### Required Deliverables

1. **Trained model checkpoint**
   - Achieves 95%+ accuracy
   - ~1.5M parameters
   - Saved weights and configuration

2. **Dataset files**
   - 100K training examples
   - 10K validation examples
   - 10K test examples
   - 10K OOD test examples
   - All in JSON Lines format

3. **Attention visualization gallery**
   - 32 heatmaps (one per head)
   - Annotated with interpretations
   - High-quality images

4. **Circuit diagram**
   - Professional-quality flowchart
   - Shows information flow
   - Labels each component's function
   - Includes evidence citations

5. **Interactive exploration tool**
   - Gradio web interface
   - User can input examples and see processing
   - Displays attention patterns by layer/head

6. **Comprehensive README**
   - Project motivation
   - Methodology
   - Key findings
   - How to reproduce
   - How to use tools

### Optional High-Impact Deliverables

7. **Research-style write-up**
   - 8-12 pages
   - Introduction, Methods, Results, Discussion
   - Includes all visualizations
   - Formatted like ML conference paper

8. **Video walkthrough**
   - 5-10 minutes
   - Demonstrates model processing an example
   - Explains discovered circuits
   - Shows interactive tool

9. **Public GitHub repository**
   - Clean, documented code
   - Installation instructions
   - Example notebooks
   - Pre-trained model available for download

---

## Success Criteria

### Technical Success
- ✅ Model achieves 95%+ accuracy on test set
- ✅ Attention patterns are clearly interpretable
- ✅ You can explain what each major head does
- ✅ Activation patching results are consistent with hypotheses
- ✅ Circuit diagram accurately predicts model behavior

### Interpretability Success
- ✅ You've identified at least 10-15 heads with clear functions
- ✅ You can trace information flow through the model
- ✅ You've discovered at least one surprising algorithmic detail
- ✅ You can predict model failures based on circuit understanding
- ✅ Visualizations clearly communicate your findings

### Interview Readiness Success
- ✅ You can explain every design decision you made
- ✅ You can demo the interactive tool confidently
- ✅ You can answer "why did you do X instead of Y?" for all choices
- ✅ You have concrete examples ready for common interview questions
- ✅ Your GitHub repository looks professional and complete

### Research Quality Success
- ✅ Your methodology is systematic and reproducible
- ✅ Your claims are backed by evidence (not speculation)
- ✅ You acknowledge limitations and failure modes
- ✅ You connect your work to relevant research papers
- ✅ Your documentation is clear enough for others to replicate

---

## Why This Project Stands Out

### Demonstrates Multiple Skills

**Technical depth**:
- Neural network architecture implementation
- Training pipeline construction
- Model analysis and debugging

**Research methodology**:
- Hypothesis formation and testing
- Systematic experimentation
- Evidence-based reasoning

**Software engineering**:
- Clean code organization
- Data pipeline construction
- Tool building (interactive interfaces)

**Communication**:
- Visualization design
- Technical writing
- Presentation skills

### Shows Genuine Curiosity

Most candidates fine-tune models. You're asking "but how does it actually work?" This demonstrates:
- Intellectual curiosity beyond surface-level understanding
- Initiative to go beyond what's taught in courses
- Passion for deep understanding

### Connects to Important Research

Mechanistic interpretability is:
- Active research area (Anthropic, OpenAI, others)
- Critical for AI safety and alignment
- Growing field with many open questions

Your project shows you're aware of cutting-edge research directions.

### Provides Concrete Talking Points

Instead of saying "I understand transformers," you can say:
- "I discovered that Head 2.3 implements an argmin operation"
- "I found the model uses a distributed representation across three heads for counting"
- "I verified through activation patching that the modulo circuit is separate from the addition circuit"

These specific, evidence-based claims are far more impressive than general statements.

---

## Final Thoughts

This project is ambitious but achievable. The key to success is:

1. **Start simple**: Get basic pipeline working before adding complexity
2. **Iterate quickly**: Train small models first, scale up when confident
3. **Document as you go**: Don't wait until the end to write explanations
4. **Focus on understanding**: The goal is insight, not just accuracy
5. **Be systematic**: Hypothesis → Experiment → Evidence → Conclusion

The most valuable outcome isn't the trained model—it's your deep understanding of how transformers think, which will serve you throughout your career in machine learning.

You'll emerge from this project able to confidently discuss transformer internals at a level that very few practitioners achieve, positioning you as someone who doesn't just use tools but truly understands them.
