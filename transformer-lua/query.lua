-- query.lua
local sqlite3 = require('lsqlite3')
local ffi = require('ffi')
local math = require('math')
local DEBUG = true      -- Set to false to disable verbose output
local VERBOSE = true    -- Enable detailed logging

local function log(...)
    if VERBOSE then
        print("[DEBUG]", ...)
    end
end

-- Configuration matching training script
local cfg = {
    vocab_size = 29219,  -- MUST MATCH TRAIN.LUA
    embed_dim = 128,
    num_heads = 8,
    num_layers = 6,
    seq_len = 256,
    model_db = 'gpt_model.db',
    temperature = 0.7,
    top_k = 10,
    repetition_penalty = 1.2
}

ffi.cdef[[
    void* malloc(size_t size);
    void free(void *ptr);
]]

local db = sqlite3.open(cfg.model_db)
local GPT = {}
local vocab = {}
local idx_to_word = {}

-- Tensor creation (same as training)
local function create_tensor(rows, cols)
    local size = rows * cols
    local data = ffi.new("double[?]", size)
    return {
        data = data,
        rows = rows,
        cols = cols,
        get = function(self, i, j) return self.data[(i-1)*self.cols + (j-1)] end,
        set = function(self, i, j, val) self.data[(i-1)*self.cols + (j-1)] = val end
    }
end

------------------------------------------------------------
-- Helper functions for Transformer components
------------------------------------------------------------

-- Layer Normalization over a vector of dimension cfg.embed_dim.
local function layer_norm(vec)
    local sum = 0.0
    for d = 0, cfg.embed_dim - 1 do
        sum = sum + vec.data[d]
    end
    local mean = sum / cfg.embed_dim
    local var = 0.0
    for d = 0, cfg.embed_dim - 1 do
        var = var + (vec.data[d] - mean)^2
    end
    var = var / cfg.embed_dim
    local eps = 1e-5
    local normalized = ffi.new("double[?]", cfg.embed_dim)
    for d = 0, cfg.embed_dim - 1 do
        normalized[d] = (vec.data[d] - mean) / math.sqrt(var + eps)
    end
    return { data = normalized, grad = ffi.new("double[?]", cfg.embed_dim) }
end

-- Multi-Head Attention:
-- Splits the input tokens into heads, computes scaled dot-product attention,
-- then concatenates heads and applies a final linear projection.
local function multi_head_attention(tokens, attn_weights)
    local head_dim = cfg.embed_dim / cfg.num_heads
    local seq_len = #tokens
    -- Compute Q, K, V for each token and head.
    local Q = {}
    local K = {}
    local V = {}
    for t = 1, seq_len do
        Q[t] = {}
        K[t] = {}
        V[t] = {}
        for h = 1, cfg.num_heads do
            Q[t][h] = ffi.new("double[?]", head_dim)
            K[t][h] = ffi.new("double[?]", head_dim)
            V[t][h] = ffi.new("double[?]", head_dim)
            for d = 0, head_dim - 1 do
                Q[t][h][d] = 0.0
                K[t][h][d] = 0.0
                V[t][h][d] = 0.0
            end
            -- Compute linear projections for Q, K, V.
            for d = 0, head_dim - 1 do
                for i = 1, cfg.embed_dim do
                    Q[t][h][d] = Q[t][h][d] + tokens[t].data[i-1] * attn_weights.q:get(i, (h-1)*head_dim + d + 1)
                    K[t][h][d] = K[t][h][d] + tokens[t].data[i-1] * attn_weights.k:get(i, (h-1)*head_dim + d + 1)
                    V[t][h][d] = V[t][h][d] + tokens[t].data[i-1] * attn_weights.v:get(i, (h-1)*head_dim + d + 1)
                end
            end
        end
    end

    -- For each head, compute attention outputs.
    local attended = {}
    -- Initialize attended as zeros for each token (size: cfg.embed_dim)
    for t = 1, seq_len do
        attended[t] = ffi.new("double[?]", cfg.embed_dim)
        for i = 0, cfg.embed_dim - 1 do
            attended[t][i] = 0.0
        end
    end

    for h = 1, cfg.num_heads do
        -- Compute scaled dot-product attention for head h.
        local scores = {}
        for t = 1, seq_len do
            scores[t] = {}
            for s = 1, seq_len do
                local dot = 0.0
                for d = 0, head_dim - 1 do
                    dot = dot + Q[t][h][d] * K[s][h][d]
                end
                scores[t][s] = dot / math.sqrt(head_dim)
            end
        end
        -- Softmax over each row of scores.
        local softmax = {}
        for t = 1, seq_len do
            softmax[t] = {}
            local max_score = -math.huge
            for s = 1, seq_len do
                if scores[t][s] > max_score then max_score = scores[t][s] end
            end
            local sum_exp = 0.0
            for s = 1, seq_len do
                softmax[t][s] = math.exp(scores[t][s] - max_score)
                sum_exp = sum_exp + softmax[t][s]
            end
            for s = 1, seq_len do
                softmax[t][s] = softmax[t][s] / sum_exp
            end
        end
        -- For each token, compute the weighted sum of V values.
        for t = 1, seq_len do
            local head_out = ffi.new("double[?]", head_dim)
            for d = 0, head_dim - 1 do head_out[d] = 0.0 end
            for s = 1, seq_len do
                for d = 0, head_dim - 1 do
                    head_out[d] = head_out[d] + softmax[t][s] * V[s][h][d]
                end
            end
            -- Place head_out in the proper slice of the concatenated attended vector.
            for d = 0, head_dim - 1 do
                attended[t][(h-1)*head_dim + d] = head_out[d]
            end
        end
    end

    -- Apply final linear projection with attn_weights.proj (shape: embed_dim x embed_dim)
    local output = {}
    for t = 1, seq_len do
        output[t] = ffi.new("double[?]", cfg.embed_dim)
        for j = 1, cfg.embed_dim do
            output[t][j-1] = 0.0
            for i = 1, cfg.embed_dim do
                output[t][j-1] = output[t][j-1] + attended[t][i-1] * attn_weights.proj:get(i, j)
            end
        end
    end

    return output
end

-- MLP (feed-forward) sublayer: fc1 (with ReLU) then fc2.
local function mlp_layer(tokens, mlp_weights)
    local seq_len = #tokens
    local output = {}
    for t = 1, seq_len do
        -- First linear layer (fc1)
        local fc1_out = ffi.new("double[?]", 4 * cfg.embed_dim)
        for j = 1, 4 * cfg.embed_dim do
            fc1_out[j-1] = 0.0
            for i = 1, cfg.embed_dim do
                fc1_out[j-1] = fc1_out[j-1] + tokens[t].data[i-1] * mlp_weights.fc1:get(i, j)
            end
            -- Apply ReLU activation
            if fc1_out[j-1] < 0 then fc1_out[j-1] = 0 end
        end
        -- Second linear layer (fc2)
        local fc2_out = ffi.new("double[?]", cfg.embed_dim)
        for j = 1, cfg.embed_dim do
            fc2_out[j-1] = 0.0
            for i = 1, 4 * cfg.embed_dim do
                fc2_out[j-1] = fc2_out[j-1] + fc1_out[i-1] * mlp_weights.fc2:get(i, j)
            end
        end
        output[t] = ffi.new("double[?]", cfg.embed_dim)
        for d = 0, cfg.embed_dim - 1 do
            output[t][d] = fc2_out[d]
        end
    end
    return output
end

------------------------------------------------------------
-- Model Loading
------------------------------------------------------------
local function load_model()
    print("[DEBUG] Initializing model structure...")
    GPT = {
        wte = create_tensor(cfg.vocab_size + 2, cfg.embed_dim),
        wpe = create_tensor(cfg.seq_len, cfg.embed_dim),
        blocks = {},
        head = create_tensor(cfg.embed_dim, cfg.vocab_size + 2)
    }

    print("[DEBUG] Loading vocabulary...")
    local stmt, err = db:prepare("SELECT word, id FROM vocab")
    if not stmt then
        error("Vocabulary prepare failed: " .. (err or db:errmsg()))
    end
    
    for row in stmt:nrows() do
        local id = tonumber(row.id)
        vocab[row.word] = id
        idx_to_word[id] = row.word
        print(string.format("[DEBUG] Loaded word: %-15s → ID %d", row.word, id))
    end
    stmt:finalize()

    print("[DEBUG] Loading embeddings...")
    stmt = db:prepare("SELECT type, position, dim, value FROM embeddings")
    for row in stmt:nrows() do
        local tensor = row.type == "wte" and GPT.wte or GPT.wpe
        local pos = tonumber(row.position) + 1
        local dim = tonumber(row.dim) + 1
        tensor:set(pos, dim, row.value)
    end
    stmt:finalize()

    print("[DEBUG] Loading transformer blocks...")
    for layer = 0, cfg.num_layers - 1 do
        print(string.format("[DEBUG] Loading layer %d...", layer))
        local block = {
            attn = {
                q = create_tensor(cfg.embed_dim, cfg.embed_dim),
                k = create_tensor(cfg.embed_dim, cfg.embed_dim),
                v = create_tensor(cfg.embed_dim, cfg.embed_dim),
                proj = create_tensor(cfg.embed_dim, cfg.embed_dim)
            },
            mlp = {
                fc1 = create_tensor(cfg.embed_dim, 4 * cfg.embed_dim),
                fc2 = create_tensor(4 * cfg.embed_dim, cfg.embed_dim)
            }
        }
        
        stmt = db:prepare("SELECT component, i, j, value FROM layers WHERE layer=?")
        stmt:bind_values(layer)
        for row in stmt:nrows() do
            local component = row.component
            local target
            if component == "q" or component == "k" or component == "v" or component == "proj" then
                target = block.attn[component]
            elseif component == "fc1" or component == "fc2" then
                target = block.mlp[component]
            else
                error("Invalid component: " .. component)
            end
            local i = tonumber(row.i) + 1
            local j = tonumber(row.j) + 1
            target:set(i, j, row.value)
        end
        stmt:finalize()
        table.insert(GPT.blocks, block)
    end

    print("[DEBUG] Loading head weights...")
    stmt = db:prepare("SELECT i, j, value FROM head")
    for row in stmt:nrows() do
        local i = tonumber(row.i) + 1
        local j = tonumber(row.j) + 1
        GPT.head:set(i, j, row.value)
    end
    stmt:finalize()
    
    print("[DEBUG] Model loading completed successfully!")
end

------------------------------------------------------------
-- Tokenization and Detokenization
------------------------------------------------------------
local function tokenize(text)
    local tokens = {}
    text = text:gsub("[%c%p]", ""):lower()
    text = text:gsub("%s+", " "):gsub("^%s*", ""):gsub("%s*$", "")
    for word in text:gmatch("%S+") do
        local id = vocab[word] or cfg.vocab_size + 1  -- fallback to <unk>
        table.insert(tokens, id)
    end
    if #tokens == 0 then
        table.insert(tokens, 0)  -- <pad>
    end
    return tokens
end

local function detokenize(tokens)
    local words = {}
    for _, id in ipairs(tokens) do
        local word = idx_to_word[id] or "<unk>"
        table.insert(words, word)
    end
    return table.concat(words, " ")
end

------------------------------------------------------------
-- Forward Pass with Residual Connections and Layer Norm
------------------------------------------------------------
local function forward(inputs)
    local batch_size = #inputs
    local seq_len = #inputs[1]
    local outputs = {}
    -- Initial embedding: add token embedding and positional embedding.
    for b = 1, batch_size do
        outputs[b] = {}
        for t = 1, seq_len do
            local token_id = inputs[b][t]
            local emb = ffi.new("double[?]", cfg.embed_dim)
            for d = 1, cfg.embed_dim do
                emb[d-1] = GPT.wte:get(token_id, d) + GPT.wpe:get(t, d)
            end
            outputs[b][t] = { data = emb, grad = ffi.new("double[?]", cfg.embed_dim) }
        end
    end

    -- Process each transformer layer.
    for layer = 1, cfg.num_layers do
        for b = 1, batch_size do
            local seq = outputs[b]
            -- Save input for residual
            local residual1 = {}
            for t = 1, seq_len do
                local vec = seq[t]
                residual1[t] = { data = ffi.new("double[?]", cfg.embed_dim) }
                for d = 0, cfg.embed_dim - 1 do
                    residual1[t].data[d] = vec.data[d]
                end
            end
            -- Multi-head attention sublayer.
            local attn_out = multi_head_attention(seq, GPT.blocks[layer].attn)
            -- Add residual connection and apply layer norm.
            for t = 1, seq_len do
                for d = 0, cfg.embed_dim - 1 do
                    attn_out[t][d] = attn_out[t][d] + residual1[t].data[d]
                end
                seq[t] = layer_norm({ data = attn_out[t], grad = ffi.new("double[?]", cfg.embed_dim) })
            end

            -- Save post-attention output for residual to MLP.
            local residual2 = {}
            for t = 1, seq_len do
                local vec = seq[t]
                residual2[t] = { data = ffi.new("double[?]", cfg.embed_dim) }
                for d = 0, cfg.embed_dim - 1 do
                    residual2[t].data[d] = vec.data[d]
                end
            end
            -- MLP sublayer.
            local mlp_out = mlp_layer(seq, GPT.blocks[layer].mlp)
            -- Add residual connection and apply layer norm.
            for t = 1, seq_len do
                for d = 0, cfg.embed_dim - 1 do
                    mlp_out[t][d] = mlp_out[t][d] + residual2[t].data[d]
                end
                seq[t] = layer_norm({ data = mlp_out[t], grad = ffi.new("double[?]", cfg.embed_dim) })
            end
        end
    end

    -- Final projection using GPT.head.
    local logits = {}
    for b = 1, batch_size do
        logits[b] = {}
        for t = 1, seq_len do
            logits[b][t] = ffi.new("double[?]", cfg.vocab_size + 2)
            for v = 0, cfg.vocab_size + 1 do
                logits[b][t][v] = 0.0
                for d = 1, cfg.embed_dim do
                    logits[b][t][v] = logits[b][t][v] + outputs[b][t].data[d-1] * GPT.head:get(d, v+1)
                end
            end
        end
    end

    return logits
end

------------------------------------------------------------
-- Text Generation (Sampling)
------------------------------------------------------------
local function generate(input_ids, max_length, temperature)
    temperature = temperature or cfg.temperature
    local generated = {unpack(input_ids)}
    local repetition_penalty = cfg.repetition_penalty
    local last_tokens = {}
    local max_recent = 4

    for i = 1, max_length do
        local start_idx = math.max(1, #generated - cfg.seq_len + 1)
        local context = {unpack(generated, start_idx, #generated)}
        local inputs = {context}
        local logits = forward(inputs)[1][#context]  -- logits for the last token

        local max_logit = -math.huge
        for v = 0, cfg.vocab_size + 1 do
            max_logit = math.max(max_logit, logits[v])
        end

        local exps = ffi.new("double[?]", cfg.vocab_size + 2)
        local sum_exp = 0.0
        for v = 0, cfg.vocab_size + 1 do
            exps[v] = math.exp((logits[v] - max_logit) / temperature)
            sum_exp = sum_exp + exps[v]
        end

        for _, id in ipairs(last_tokens) do
            exps[id] = exps[id] / repetition_penalty
        end

        local threshold = math.random() * sum_exp
        local cum = 0.0
        local selected_id = cfg.vocab_size + 1
        for v = 0, cfg.vocab_size + 1 do
            cum = cum + exps[v]
            if cum >= threshold then
                selected_id = v
                break
            end
        end

        if selected_id < 1 or selected_id > cfg.vocab_size + 1 then
            selected_id = cfg.vocab_size + 1
        end

        table.insert(last_tokens, selected_id)
        if #last_tokens > max_recent then
            table.remove(last_tokens, 1)
        end

        table.insert(generated, selected_id)
        if selected_id == cfg.vocab_size + 1 then
            break
        end
    end

    return generated
end

------------------------------------------------------------
-- Main Execution Loop
------------------------------------------------------------
load_model()

while true do
    print("Enter your question (press Ctrl+D when done):")
    io.write("> ")
    io.flush()
    local question = io.read()
    if not question then break end

    log(string.format("Raw input received: %q", question))
    if #question == 0 then
        error("No input provided! Did you press Ctrl+D correctly?")
    end

    local input_tokens = tokenize(question)
    log(string.format("Tokenized input (%d tokens):", #input_tokens))
    log(table.concat(input_tokens, ", "))

    local output_tokens = generate(input_tokens, 100, cfg.temperature)
    local response = detokenize(output_tokens)

    print("\n=== Final Response ===")
    print(response)
end

db:close()
