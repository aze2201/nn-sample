# Transformer-Based Language Model in Lua

This script implements a simplified version of a GPT-like transformer model using Lua. It leverages FFI for efficient memory operations, SQLite for persistent storage of model parameters, and a manual implementation of key neural network components. Although the code is designed for language modeling, its modular design makes it adaptable to other tasks such as text generation, sequence prediction, or even non-text domains with minor adjustments.

## 1. Overview & Use Cases

### Purpose

-   **Language Modeling:** Train a model to predict the next token in a sequence.
-   **Text Generation:** Once trained, the model can generate coherent text given a prompt.
-   **Educational Tool:** Illustrates how transformers, multi-head self-attention, residual connections, layer normalization, and feed-forward networks can be implemented from scratch.
-   **Custom Applications:** The design allows experimentation on different domains (e.g., music sequences, code, or time-series) by modifying the input preprocessing and vocabulary.

### Use Case Example

-   **Chatbot/Dialogue System:** Train on conversational data to generate contextually relevant responses.
-   **Creative Writing Assistance:** Use the trained model to suggest sentence completions or generate paragraphs.
-   **Data Augmentation:** Generate synthetic data for low-resource language tasks.

## 2. Architecture & Components

The network is organized into the following layers:

### Input Layer:

-   **Tokenization & Vocabulary:** The script builds a vocabulary from the training text and assigns each token a unique ID.
-   **Embeddings:** Two sets of embeddings are used:
    -   Word Token Embeddings (wte): Maps each word ID to a high-dimensional vector.
    -   Positional Embeddings (wpe): Adds information about token order in the sequence.

### Transformer Blocks (Repeated `cfg.num_layers` Times):

Each transformer block contains:

-   **Multi-head Self-Attention:**
    -   Projections: Uses matrices for queries (Q), keys (K), values (V), and a final projection.
    -   Attention Formula: `Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V`
        -   where `d_k` is the head dimension.
-   **Residual Connections & Layer Normalization:**
    -   Residual: Adds the input back to the sub-layer’s output.
    -   Layer Norm: Normalizes the activations: `LayerNorm(x) = (x - μ) / sqrt(σ^2 + ε)`
        -   where `μ` and `σ^2` are the mean and variance of `x`.
-   **Feed-Forward (MLP) Sub-layer:**
    -   Consists of two linear transformations with a ReLU activation in between: `FFN(x) = max(0, xW1 + b1)W2 + b2`

### Output Layer:

-   **Projection to Vocabulary Space:** Final activations are projected to logits over the vocabulary. These logits can be passed through a softmax to obtain probabilities.

## 3. Detailed Function Descriptions

### Initialization & Utility Functions

-   `ffi.cdef` Block: Declares standard C functions for memory allocation and deallocation.
-   `init_database()`
    -   Purpose: Opens (or creates) an SQLite database to store model configurations, vocabulary, and parameters.
    -   Flow:
        1.  Opens the database.
        2.  Executes several PRAGMA commands to optimize SQLite behavior.
        3.  Creates tables (config, vocab, layers, head, and embeddings) with error handling.
-   `create_tensor(rows, cols)`
    -   Purpose: Allocates a new tensor structure that holds:
        -   `data`: Parameter values.
        -   `grad`: Gradients computed during backpropagation.
        -   `m` and `v`: First and second moments (for the Adam optimizer).
    -   Utility Methods: `get(i, j)`, `set(i, j, val)`, `add_grad(i, j, val)`, and `zero_grad()` for element access and manipulation.
-   `layer_norm(vec, size, eps)`
    -   Purpose: Applies layer normalization to a vector.
    -   Mathematics: `μ = (1/N)Σ(x_i)`, `σ^2 = (1/N)Σ((x_i - μ)^2)`, `norm_i = (x_i - μ) / sqrt(σ^2 + ε)`
    -   Implementation: Iterates over elements to compute mean, variance, and normalized output.
-   `dropout(vec, size, dropout_rate)`
    -   Purpose: Implements elementwise dropout by randomly zeroing out some elements of the vector based on the dropout probability.

### Vocabulary Building

-   `build_vocabulary(text)`
    -   Process:
        1.  Sanitizes the input text (removes punctuation, converts to lower case).
        2.  Counts word frequencies and selects the top `cfg.vocab_size` words.
        3.  Adds special tokens `<unk>` (for unknown words) and `<pad>` (for padding).
        4.  Inserts the vocabulary into the SQLite database.

### Model Initialization

-   `transformer_block()`
    -   Purpose: Constructs a transformer block consisting of:
        -   Attention Components: Matrices for Q, K, V, and projection.
        -   MLP Components: Two fully connected layers (fc1 and fc2).
    -   Initialization: Uses Kaiming (He) and uniform initialization methods.
-   `init_model()`
    -   Steps:
        1.  Initializes word and positional embeddings.
        2.  Builds a stack of transformer blocks.
        3.  Initializes the final projection layer (head) that maps to vocabulary logits.
    -   Parameter Scaling: Uses scaling factors (e.g., `1 / sqrt(embed_dim)`) for initialization.

### Forward Pass

-   `forward(inputs)`
    -   Process Overview:
        1.  Embedding Lookup:
            -   For each token in the input batch, sum the corresponding word embedding and positional embedding.
        2.  Transformer Layers:
            -   For each transformer block:
                -   Layer Normalization: Normalize the input token vectors.
                -   Multi-head Self-Attention:
                    -   Compute Q, K, and V for each head.
                    -   Calculate attention weights using the scaled dot-product formula.
                    -   Aggregate the head outputs.
                    -   Apply a projection and dropout.
                -   Residual Connection: Add the attention output back to the original token embedding.
                -   Feed-Forward Network:
                    -   Apply a fully connected layer with ReLU activation.
                    -   Apply a second fully connected layer followed by dropout.
                -   Second Residual Connection: Sum the MLP output with the output from the attention sub-layer.
       3.  Logits Computation:
            -   The final activations are projected onto the vocabulary space using the head tensor.
    -   Math Behind Attention:
        -   For a given head: `q = LayerNorm(x)Wq`, `k = LayerNorm(x)Wk`, `v = LayerNorm(x)Wv`
        -   `scores = q k^T / sqrt(d_k)`, `weights = softmax(scores)`, `output = weights * v`

### Backward Pass & Optimization

-   `compute_gradients(logits, targets)`
    -   Purpose: Computes the gradients for the final logits using a form of cross-entropy loss.
    -   Details:
        -   Iterates over each token in the batch.
        -   Uses softmax gradients for updating the head layer’s parameters.
        -   Applies a scaling factor to average the gradients over the batch.
-   `cross_entropy(logits, targets)`
    -   Calculation: `loss = -(1/N)Σ[log(exp(logit_t) / Σ(exp(logit_v)))]`
        -   where `t` is the target token index.
-   `adam_step(param, t)`
    -   Purpose: Updates parameters using the Adam optimization algorithm.
    -   Mathematical Formulas:
        -   `m_t = β1 m_{t-1} + (1-β1) g_t`, `v_t = β2 v_{t-1} + (1-β2) g_t^2`
        -   `m_hat_t = m_t / (1-β1^t)`, `v_hat_t = v_t / (1-β2^t)`
        -   `θ_t = θ_{t-1} - α m_hat_t / (sqrt(v_hat_t) + ε)`
-   `get_batch(text_tokens)`
    -   Purpose: Creates mini-batches by sampling sequences of tokens from the preprocessed text.
-   `save_model()`
    -   Purpose: Saves model parameters (embeddings, transformer layers, and head) into the SQLite database.
    -   Flow:
        -   Begins a transaction.
        -   Deletes old values and inserts current parameter values.
        -   Commits the transaction.

### Training Loop

-   `train(text_path)`
    -   Overall Flow:
        1.  Input Reading: Reads training text from a file.
        2.  Vocabulary Construction: Builds vocabulary from the text.
        3.  Model Initialization: Sets up embeddings and transformer blocks.
        4.  Batch Training:
            -   Iteratively samples batches.
            -   Executes the forward pass to compute logits.
            -   Calculates the cross-entropy loss.
            -   Computes gradients.
            -   Applies Adam updates for each model parameter.
            -   Periodically saves the model.
        5.  Finalization: Closes the database and prints completion status.

## 4. Neural Network Architecture Diagram

Below is a simplified diagram representing the flow through one transformer block. Each “node” represents a vector (or matrix) of size `embed_dim`.
```
                                                 +------------------------+
                                                 |      Input Tokens      |
                                                 +-----------+------------+
                                                             |
                                              +--------------v--------------+
                                              |  Embedding Lookup           |  (wte + wpe)
                                              +--------------+--------------+
                                                             |
                                               +-------------v-------------+
                                               |      Transformer Block    |
                                               +-------------+-------------+
                                                             |
                                       +---------------------+--------------------+
                                       |                                          |
                                 +-----v------+                           +-------v-------+
                                 | Multi-head |                           | Feed Forward  |
                                 | Attention  |                           | (MLP)         |
                                 +-----+------+                           +-------+-------+
                                       |                                          |
                                       |          Residual Connection             |
                                       +-------------------+----------------------+
                                                           |
                                                    +------v------+
                                                    |  LayerNorm  |
                                                    +-------------+
                                                           |
                                                    +------v------+
                                                    |   Output    |
                                                    +-------------+
                                                           |
                                                +----------v-----------+
                                                |  Final Projection    |
                                                |   (to logits)        |
                                                +----------+-----------+
```

### Node Details

-   **Embeddings:**
    -   Word Embedding (wte): Dimensions: `(vocab_size+2) x embed_dim`
    -   Positional Embedding (wpe): Dimensions: `seq_len x embed_dim`
-   **Attention Node:**
    -   Heads: `num_heads` separate attention computations.
    -   Each Head: Processes a split of size `head_dim = embed_dim / num_heads`.
-   **MLP Nodes:**
    -   fc1: Expands the dimension to `4 * embed_dim`.
    -   fc2: Contracts back to `embed_dim`.

## 5. Mathematical Formulas Recap

-   **Scaled Dot-Product Attention:** `Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V`
-   **Layer Normalization:** `LayerNorm(x) = (x - μ) / sqrt(σ^2 + ε)`
-   **Feed-Forward Network:** `FFN(x) = max(0, xW1 + b1)W2 + b2`
-   **Adam Update Equations:**
    -   `m_t = β1 m_{t-1} + (1-β1) g_t`, `v_t = β2 v_{t-1} + (1-β2) g_t^2`
    -   `m_hat_t = m_t / (1-β1^t)`, `v_hat_t = v_t / (1-β2^t)`
    -   `θ_t = θ_{t-1} - α m_hat_t / (sqrt(v_hat_t) + ε)`

## 6. Code Flow Summary

-   **Startup:**
    -   Parse command-line arguments.
    -   Open/create SQLite database.
-   **Preprocessing:**
    -   Read training text.
    -   Build and store vocabulary.
-   **Model Initialization:**
    -   Initialize embeddings.
    -   Build transformer blocks.
    -   Set up final projection (head).
-   **Training Loop:**
    -   Sample a batch of token sequences.
    -   Forward Pass: Compute embeddings, pass through transformer layers, and calculate logits.
    -   Loss Computation: Calculate cross-entropy loss.
    -   Backward Pass: Compute gradients.
    -   Parameter Update: Apply Adam steps for each parameter.
    -   Save model parameters periodically.
-   **Finalization:**
    -   Close the database.
    -   Report training completion.

## 7. Potential Extensions and Adaptations

-   **Other Modalities:**
    -   Adapt the embedding and tokenization layers to work with different types of data (e.g., images, audio).
-   **Fine-tuning:**
    -   Use the saved model parameters as initialization for transfer learning on a related task.
-   **Model Scaling:**
    -   Increase the number of layers, embedding dimensions, or heads to build a larger model if computational resources allow.
-   **Integration with Other Databases:**
    -   While SQLite is used for simplicity and persistence, you might integrate with more sophisticated storage for production use.

## 8. Conclusion

This Lua script is a self-contained demonstration of how to build a transformer-based language model from scratch. It encompasses:

-   Vocabulary creation and embedding initialization.
-   Implementation of transformer blocks with multi-head self-attention, residual connections, and layer normalization.
-   A training loop with forward and backward passes, including gradient computation and parameter updates using Adam.
-   Persistent storage of model parameters using SQLite.

The design and structure are modular enough to serve as a learning tool, a base for experimentation in language modeling, or even a prototype for extending to other domains.

```
          Input: Embedded Tokens [L x E]
                     |
               +-----v-----+
               | Layer Norm|  (applied to each token)
               +-----+-----+
                     |
       +-------------v-------------+
       |  Multi-Head Self-Attention|
       +-------------+-------------+
                     |
           For each Head (h = 1...H):
                     |
      +--------------------------------------+
      | Split embedding into H heads       |
      | Head Dimension: head_dim = E / H     |
      |                                      |
      | For each token i, compute:           |
      |   Q_i = X_i · W_q    (size: head_dim)|
      |   K_i = X_i · W_k    (size: head_dim)|
      |   V_i = X_i · W_v    (size: head_dim)|
      +----------------+---------------------+
                     |
         Compute Attention per head:
                     |
       +-------------v-------------+
       |   Scaled Dot-Product       |
       |  Attention Formula:        |
       |                            |
       |  Attention(Q, K, V) =      |
       |  softmax( QKᵀ / √dₖ )V     |
       |                            |
       +-------------+-------------+
                     |
      +--------------v--------------+
      | Concatenate outputs of all  |
      | heads back into a single    |
      | vector of dimension E       |
      +--------------+--------------+
                     |
            +--------v--------+
            | Linear Projection|
            |  (W_proj: E x E) |
            +--------+--------+
                     |
          Dropout (optional)

```

```
          📜 Input Text
             │
             ▼
    🔠 Tokenization + Embedding
             │
   🔢 (Word Embed + Position Embed)
             │
┌────────────▼─────────────┐
│        GPT Core          │
│  ┌────────────────────┐  │
│  │    Transformer     │  │
│  │  ┌───────────────┐ │  │
│  │  │  Multi-Head   ◀───| 
│  │  │   Attention   │ │ 
│  │  └───────┬───────┘ │ 
│  │      Q │ K │ V     │ 
│  │        ╲ │ ╱       │ 
│  │      ┌───▼───┐     │ 
│  │      │ Scale │     │
│  │      │ &     │     │
│  │      │Softmax│     │
│  │      └───┬───┘     │
│  │          ▼         │
│  │    Context Vector  │
│  └───────┬────────────┘
│          │ Δ Residual
│  ┌───────▼───────┐
│  │ Layer Norm    │
│  └───────┬───────┘
│          │
│  ┌───────▼───────┐
│  │    FFN        │
│  │ (Dense 4x)    │
│  │   ReLU        │
│  └───────┬───────┘
│          │ Δ Residual
│  ┌───────▼───────┐
│  │ Layer Norm    │
│  └───────┬───────┘
└──────────┼──────────┘
           │ (×6 layers)
           ▼
     🔮 Output Head
           │
           ▼
     🎯 Logits (426+2)
           │
           ▼
     🧮 Softmax
           │
           ▼
      📊 Next Token Prediction

Key Components Detail:

🔢 Embedding Layer:
╭──────────────╮
│ Token:  ░▒▓  │
│ Pos:   ▒▓██  │
╰───────┬──────╯
        ⊕
        
🤖 Transformer Block:
╭───────────────────╮
│        Q K V      │
│  ┌───┐ ┌───┐ ┌───┐│
│  │ Wq│ │ Wk│ │ Wv││
│  └─┬─┘ └─┬─┘ └─┬─┘│
│    │     │     │  │
│  ┌─▼─────▼─────▼─┐│
│  │ Attention      │
│  │  Scores        │
│  └──────┬─────────┘│
│         ▼          │
│  ┌───────────────┐ │
│  │  Value Weight │ │
│  └──────┬────────┘ │
╰─────────┼──────────╯
          ⊕
          
📐 Key Formulas:
• Attention(Q,K,V) = softmax(QKᵀ/√d)⋅V
• LayerNorm(x) = (x-μ)/√(σ²+ε)
• FFN(x) = max(0, xW₁+b₁)W₂+b₂
• Loss = -Σ log(exp(s_y)/Σ exp(s_j))


```

