# Transformer From Scratch Curriculum

This project is a hands-on journey to build, train, and optimize Transformer models from first principles using PyTorch.

## Goal
1.  Write Transformer layers from scratch.
2.  Implement Decoder-Only (GPT-style) and Encoder-Decoder models.
3.  Train on simple datasets (Generation & Seq2Seq).
4.  Implement inference optimizations (KV-Cache).

## Modernizations
We are deviating from the 2017 paper defaults to align with modern Large Language Models (LLaMA, PaLM, Mistral):
*   **RoPE (Rotary Positional Embeddings):** Instead of adding learned vectors to embeddings, we rotate Q and K vectors.
*   **RMSNorm:** More efficient and stable than LayerNorm.
*   **SwiGLU:** Gated activation function for the FeedForward block (better performance than ReLU/GELU).
*   **Pre-Normalization:** Apply Norm *before* the sublayer (Attention/FFN) for training stability.
    *   *Post-Norm (Original):* `x = Norm(x + Sublayer(x))`. Gradients can explode near output layers, requiring careful warm-up.
    *   *Pre-Norm (Modern):* `x = x + Sublayer(Norm(x))`. Gradients flow directly through the "residual superhighway" (the `x + ...`), significantly stabilizing deep model training.
*   **No Biases:** Linear layers and Norms will have `bias=False` to save compute/memory and improve stability.

## Curriculum & Roadmap

### Phase 1: Foundations & The Transformer Block
**Objective:** Build the atomic components of the Transformer architecture.

*   [x] **Step 1: Embeddings**
    *   **Module:** `src/models/embeddings.py`
    *   **Tasks:**
        *   Implement `EmbeddingLayer` (wrapper around `nn.Embedding` *scaled by sqrt(d_model)*).
    *   **Audit Check:** Verify output shapes and ensure scaling is applied.

*   [x] **Step 2: The Heart - Attention Mechanisms**
    *   **Module:** `src/models/attention.py`
    *   **Tasks:**
        *   Implement `MultiHeadAttention` from scratch using `einops`.
        *   Implement **RoPE** (Rotary Positional Embeddings) to encode position information into Q and K.
        *   Implement `scaled_dot_product_attention` manually.
        *   Ensure all Linear projections use `bias=False`.
    *   **Audit Check:** Verify output shapes and that causal masking prevents "looking ahead".

*   [x] **Step 3: Layers & Normalization**
    *   **Module:** `src/models/layers.py`
    *   **Tasks:**
        *   Implement `RMSNorm` from scratch.
        *   Implement `SwiGLU` FeedForward block (Gated linear units with Swish activation).
        *   Create the `DecoderBlock` using **Pre-Norm** architecture (Norm -> Attention -> Add -> Norm -> FFN -> Add).
    *   **Audit Check:** Pass a dummy tensor through a block and verify gradients propagate.

### Phase 2: Decoder-Only Model (GPT) & Training
**Objective:** Assemble a GPT-style model and train it to generate text.

*   [x] **Step 4: Assembling the Decoder-Only Model**
    *   **Module:** `src/models/transformer.py`
    *   **Tasks:**
        *   Create `DecoderOnlyTransformer` class.
        *   **Skip** the absolute positional encoding layer (since we use RoPE).
        *   Stack `DecoderBlock`s.
        *   Add the final projection head (bias=False) to vocabulary size.
    *   **Audit Check:** Verify parameter count matches expected calculations.

*   [ ] **Step 5: Data Pipeline & Training Loop**
    *   **Modules:** `src/data/dataset.py`, `src/train.py`
    *   **Tasks:**
        *   Download "Tiny Shakespeare".
        *   Implement a simple Character-level Tokenizer.
        *   Create a PyTorch `Dataset` and `DataLoader` for autoregressive tasks (x=tokens[i:i+n], y=tokens[i+1:i+n+1]).
        *   Write the training loop (CrossEntropyLoss, AdamW).
    *   **Audit Check:** Overfit a single batch (loss goes to near 0), then train for real and watch loss curve drop.

### Phase 3: Inference & Optimization
**Objective:** Efficient text generation.

*   [ ] **Step 6: Inference & KV Cache**
    *   **Module:** `src/inference.py` (or methods in `transformer.py`)
    *   **Tasks:**
        *   Implement greedy decoding loop.
        *   Refactor `MultiHeadAttention` to support **KV Caching** (passing past keys/values to avoid re-computation).
    *   **Audit Check:** Compare generation speed with and without KV-Cache. Ensure outputs are identical.

### Phase 4: Encoder-Decoder & Seq2Seq
**Objective:** Full Transformer architecture for translation-style tasks.

*   [ ] **Step 7: The Encoder & Cross-Attention**
    *   **Modules:** `src/models/attention.py`, `src/models/transformer.py`
    *   **Tasks:**
        *   Add `CrossAttention` logic to your blocks (Decoder attending to Encoder outputs).
        *   Implement `EncoderBlock` and `Encoder`.
        *   Assemble `EncoderDecoderTransformer`.
    *   **Audit Check:** Verify shapes when passing source *and* target sequences.

*   [ ] **Step 8: Seq2Seq Training (Toy Task)**
    *   **Module:** `src/train_seq2seq.py`
    *   **Tasks:**
        *   Create a toy dataset (e.g., string reversal: "ABC" -> "CBA", or simple copy task).
        *   Train the Encoder-Decoder model.
    *   **Audit Check:** Model successfully performs the algorithmic task on unseen data.

## Resources & References

### Core Reading
*   **[Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762)**: The original paper.
    *   *Read:* Section 3 (Model Architecture).
*   **[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)**: Essential for visualizing Q, K, V attention flow.
*   **[The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)**: Line-by-line PyTorch implementation guide.

### Specific Concepts
*   **Learned Positional Encodings**: Used in BERT and GPT models. See [BERT paper](https://arxiv.org/abs/1810.04805).
*   **RMSNorm**: [Root Mean Square Layer Normalization (2019)](https://arxiv.org/abs/1910.07467).
*   **KV Cache**: [Efficiently Scaling Transformer Inference](https://arxiv.org/pdf/1911.02150.pdf).
