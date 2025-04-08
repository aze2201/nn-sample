-- query.lua: Load a trained GPT model and generate text

local sqlite3 = require('lsqlite3')
local math = require('math')
local ffi = require('ffi')
local os = require('os') -- For os.clock() maybe, and os.time() for randomseed

ffi.cdef[[
    void free(void *ptr);
    void* malloc(size_t size);
]]

-- Default Configuration (will be overridden by loaded model config)
local cfg = {
    vocab_size = 256, -- Placeholder
    embed_dim = 256,
    num_heads = 8,
    num_layers = 6,
    seq_len = 256,
    dropout = 0.0, -- Set dropout to 0 for inference
    model_db = 'gpt_model.db',
    relative_pos_bias = true,
    num_relative_pos_buckets = 32,
    max_distance = 128,
    -- Other config values might be loaded but aren't directly used in generation logic here
}

-- Global variables for loaded model and vocab
local GPT = {}
local vocab = {}        -- word -> id
local idx_to_word = {}  -- id -> word
local db -- Database handle

--------------------------------------------------
-- Utility Functions (Mostly Copied from Training Script)
--------------------------------------------------

-- Layer normalization forward (only forward needed)
local function layer_norm_forward(vec, size, eps)
    eps = eps or 1e-5
    local sum = 0
    for i = 0, size - 1 do
        sum = sum + vec[i]
    end
    local mean = sum / size
    local variance = 0
    for i = 0, size - 1 do
        local diff = vec[i] - mean
        variance = variance + diff * diff
    end
    variance = variance / size
    local norm = ffi.new("double[?]", size)
    for i = 0, size - 1 do
        norm[i] = (vec[i] - mean) / math.sqrt(variance + eps)
    end
    -- No cache needed for simple inference if not optimizing heavily
    return norm --, {mean=mean, variance=variance, size=size, eps=eps, vec=vec}
end

-- Dropout forward (modified for inference - does nothing)
local function dropout_forward(vec, size, dropout_rate)
    -- During inference, dropout should be disabled. Return the input directly.
    return vec, {} -- Return original vector and empty mask
end

-- ReLU forward (only forward needed)
local function relu_forward(input, size)
    local out = ffi.new("double[?]", size)
    for i = 0, size - 1 do
        if input[i] > 0 then
            out[i] = input[i]
        else
            out[i] = 0
        end
    end
    -- No mask needed for simple inference
    return out --, mask
end

-- Tensor/Bias Creation (needed to structure the model before loading)
local function create_tensor(rows, cols)
    local size = rows * cols
    local data = ffi.new("double[?]", size)
    ffi.fill(data, ffi.sizeof("double") * size, 0) -- Initialize data to 0
    return {
        data = data,
        rows = rows,
        cols = cols,
        get = function(self, i, j)
            -- 1-based Lua indexing to 0-based C indexing
            if i < 1 or i > self.rows or j < 1 or j > self.cols then
                error(string.format("Tensor get index out of bounds: (%d, %d) accessing [%d, %d]", i, j, self.rows, self.cols))
            end
            return self.data[(i-1)*self.cols + (j-1)]
        end,
        set = function(self, i, j, val)
             -- 1-based Lua indexing to 0-based C indexing
            if i < 1 or i > self.rows or j < 1 or j > self.cols then
                 error(string.format("Tensor set index out of bounds: (%d, %d) accessing [%d, %d]", i, j, self.rows, self.cols))
            end
            self.data[(i-1)*self.cols + (j-1)] = val
        end,
        -- grad, m, v, add_grad, zero_grad are not needed for inference
    }
end

local function create_bias(size)
    local data = ffi.new("double[?]", size)
    ffi.fill(data, ffi.sizeof("double") * size, 0) -- Initialize data to 0
    return {
        data = data,
        size = size,
        get = function(self, i)
             -- 1-based Lua indexing to 0-based C indexing
            if i < 1 or i > self.size then
                 error(string.format("Bias get index out of bounds: %d accessing size %d", i, self.size))
            end
            return self.data[i-1]
        end,
        set = function(self, i, val)
             -- 1-based Lua indexing to 0-based C indexing
             if i < 1 or i > self.size then
                 error(string.format("Bias set index out of bounds: %d accessing size %d", i, self.size))
            end
            self.data[i-1] = val
        end,
         -- grad, m, v, add_grad, zero_grad are not needed for inference
    }
end

local function create_relative_position_bias()
    local num_buckets = cfg.num_relative_pos_buckets
    local num_heads = cfg.num_heads
    local data = ffi.new("double[?]", num_buckets * num_heads)
    ffi.fill(data, ffi.sizeof("double") * (num_buckets * num_heads), 0)
    return {
        data = data,
        num_buckets = num_buckets,
        num_heads = num_heads,
        get = function(self, bucket, head)
             -- 1-based Lua indexing to 0-based C indexing
            if bucket < 1 or bucket > self.num_buckets or head < 1 or head > self.num_heads then
                 error(string.format("RelPosBias get index out of bounds: (bucket %d, head %d) accessing [%d, %d]", bucket, head, self.num_buckets, self.num_heads))
            end
            return self.data[(bucket-1) * self.num_heads + (head-1)]
        end,
        set = function(self, bucket, head, val)
            -- 1-based Lua indexing to 0-based C indexing
            if bucket < 1 or bucket > self.num_buckets or head < 1 or head > self.num_heads then
                 error(string.format("RelPosBias set index out of bounds: (bucket %d, head %d) accessing [%d, %d]", bucket, head, self.num_buckets, self.num_heads))
            end
            self.data[(bucket-1) * self.num_heads + (head-1)] = val
        end,
        -- grad, m, v, add_grad, zero_grad are not needed for inference
    }
end

-- Relative Position Bucketing (needed for forward pass)
local function relative_position_bucket(relative_position, bidirectional, num_buckets, max_distance)
    -- Simplified T5-style bucketing. Note: Original training code used log scale. Keep consistent!
    -- Assuming the training code used the log scale version, keep it:
    local ret = 0
    local n = -relative_position -- T5 uses query_pos - key_pos
    if bidirectional then
        num_buckets = num_buckets / 2
        ret = ret + (n < 0 and num_buckets or 0)
        n = math.abs(n)
    else
        n = math.max(0, n) -- Only consider past positions for causal attention
    end

    local max_exact = num_buckets / 2
    local is_small = (n < max_exact)

    local val_if_large = max_exact + math.floor(
        (num_buckets - max_exact) * math.log(n / max_exact) / math.log(max_distance / max_exact)
    )
    val_if_large = math.min(val_if_large, num_buckets - 1)

    ret = ret + (is_small and n or val_if_large)
    return ret + 1 -- Return 1-based index for Lua tables
end


-- Transformer block structure definition (NO INITIALIZATION needed here)
local function transformer_block_structure()
    return {
        attn = {
            q = create_tensor(cfg.embed_dim, cfg.embed_dim),
            k = create_tensor(cfg.embed_dim, cfg.embed_dim),
            v = create_tensor(cfg.embed_dim, cfg.embed_dim),
            proj = create_tensor(cfg.embed_dim, cfg.embed_dim),
            q_bias = create_bias(cfg.embed_dim),
            k_bias = create_bias(cfg.embed_dim),
            v_bias = create_bias(cfg.embed_dim),
            proj_bias = create_bias(cfg.embed_dim)
        },
        mlp = {
            fc1 = create_tensor(cfg.embed_dim, 4 * cfg.embed_dim),
            fc2 = create_tensor(4 * cfg.embed_dim, cfg.embed_dim),
            fc1_bias = create_bias(4 * cfg.embed_dim),
            fc2_bias = create_bias(cfg.embed_dim)
        }
    }
end

-- Transformer block forward pass (Simplified - No Cache for basic generation)
local function transformer_block_forward(block, norm_tokens, relative_pos_bias)
    local head_dim = cfg.embed_dim / cfg.num_heads
    local seq_len = #norm_tokens
    local attn_outputs = {} -- Stores the output of the attention mechanism for each token

    for t = 1, seq_len do
        attn_outputs[t] = ffi.new("double[?]", cfg.embed_dim)
        for d = 0, cfg.embed_dim-1 do attn_outputs[t][d] = 0 end
    end

    -- Multi-Head Attention Calculation
    for h = 1, cfg.num_heads do
        for i = 1, seq_len do -- For each query token
            -- Compute query (q) for token i, head h
            local q = ffi.new("double[?]", head_dim)
            for d = 1, head_dim do
                local bias_val = block.attn.q_bias:get((h-1)*head_dim + d)
                q[d-1] = bias_val
                for j = 1, cfg.embed_dim do
                    q[d-1] = q[d-1] + norm_tokens[i][j-1] * block.attn.q:get(j, (h-1)*head_dim + d)
                end
            end

            -- Compute keys (k) and values (v) for relevant tokens (j <= i for causal)
            local keys = {}
            local values = {}
            local scores = ffi.new("double[?]", seq_len) -- Scores for token i attending to tokens j
            local max_score = -math.huge

            for j = 1, i do -- Causal: only attend to positions <= i
                 local k = ffi.new("double[?]", head_dim)
                 local v = ffi.new("double[?]", head_dim)
                 for d = 1, head_dim do
                     local k_bias_val = block.attn.k_bias:get((h-1)*head_dim + d)
                     local v_bias_val = block.attn.v_bias:get((h-1)*head_dim + d)
                     k[d-1] = k_bias_val
                     v[d-1] = v_bias_val
                     for r = 1, cfg.embed_dim do
                         k[d-1] = k[d-1] + norm_tokens[j][r-1] * block.attn.k:get(r, (h-1)*head_dim + d)
                         v[d-1] = v[d-1] + norm_tokens[j][r-1] * block.attn.v:get(r, (h-1)*head_dim + d)
                     end
                 end
                 keys[j] = k
                 values[j] = v

                 -- Calculate score
                 local score = 0
                 for d = 0, head_dim-1 do
                     score = score + q[d] * keys[j][d]
                 end
                 score = score / math.sqrt(head_dim)

                 -- Add relative position bias
                 if cfg.relative_pos_bias then
                    local relative_pos = i - j -- query_pos - key_pos
                    local bucket = relative_position_bucket(relative_pos, false, cfg.num_relative_pos_buckets, cfg.max_distance)
                    score = score + relative_pos_bias:get(bucket, h)
                 end
                 scores[j-1] = score
                 if score > max_score then max_score = score end
            end

             -- Mask future tokens explicitly (scores remain 0 initialized or assigned -math.huge)
            for j = i + 1, seq_len do
                 scores[j-1] = -math.huge -- Mask score
            end

            -- Softmax scores
            local exps = ffi.new("double[?]", seq_len)
            local sum_exp = 0
            for j = 1, i do -- Only consider valid scores up to i
                local score_val = scores[j-1]
                if score_val ~= -math.huge then
                    exps[j-1] = math.exp(score_val - max_score) -- Subtract max for numerical stability
                    sum_exp = sum_exp + exps[j-1]
                else
                     exps[j-1] = 0 -- Should not happen if j <= i, but safe
                end
            end
             -- Set future token exps explicitly to 0
            for j = i+1, seq_len do exps[j-1] = 0 end

            local attn_weights = ffi.new("double[?]", seq_len)
            if sum_exp > 0 then
                for j = 1, i do
                    attn_weights[j-1] = exps[j-1] / sum_exp
                end
            else -- Handle case where all scores are -inf or sum_exp is 0
                for j = 1, i do attn_weights[j-1] = 0 end -- or uniform 1/i ? 0 is safer.
            end
            for j = i+1, seq_len do attn_weights[j-1] = 0 end -- Future weights are 0

            -- Compute weighted sum of values
            local head_output = ffi.new("double[?]", head_dim)
            for d = 0, head_dim-1 do head_output[d] = 0 end
            for j = 1, i do -- Only sum values up to position i
                local weight = attn_weights[j-1]
                if weight > 0 then -- Optimization: skip if weight is zero
                    for d = 0, head_dim-1 do
                        head_output[d] = head_output[d] + weight * values[j][d]
                    end
                end
            end

            -- Accumulate head output into the correct slice of attn_outputs
            for d = 0, head_dim-1 do
                attn_outputs[i][(h-1)*head_dim + d] = attn_outputs[i][(h-1)*head_dim + d] + head_output[d]
            end
        end
    end

    -- Apply projection
    local proj_outputs = {}
    for t = 1, seq_len do
        proj_outputs[t] = ffi.new("double[?]", cfg.embed_dim)
        for d = 1, cfg.embed_dim do
            local sum = block.attn.proj_bias:get(d)
            for r = 1, cfg.embed_dim do
                sum = sum + attn_outputs[t][r-1] * block.attn.proj:get(r, d)
            end
            proj_outputs[t][d-1] = sum
        end
        -- Apply dropout (which does nothing in inference mode)
        proj_outputs[t], _ = dropout_forward(proj_outputs[t], cfg.embed_dim, cfg.dropout)
    end

    -- First residual connection
    local res1 = {}
    for t = 1, seq_len do
        res1[t] = ffi.new("double[?]", cfg.embed_dim)
        for d = 0, cfg.embed_dim-1 do
            res1[t][d] = norm_tokens[t][d] + proj_outputs[t][d] -- Add input to proj output
        end
    end

    -- MLP branch
    local mlp_outputs = {}
    for t = 1, seq_len do
        local norm_res1 = layer_norm_forward(res1[t], cfg.embed_dim) -- Normalize before MLP

        -- FC1 + Bias
        local fc1_out = ffi.new("double[?]", 4 * cfg.embed_dim)
        for j = 1, 4 * cfg.embed_dim do
            local sum = block.mlp.fc1_bias:get(j)
            for r = 1, cfg.embed_dim do
                sum = sum + norm_res1[r-1] * block.mlp.fc1:get(r, j)
            end
            fc1_out[j-1] = sum
        end

        -- ReLU Activation
        local relu_out = relu_forward(fc1_out, 4 * cfg.embed_dim)

        -- FC2 + Bias
        local fc2_out = ffi.new("double[?]", cfg.embed_dim)
        for d = 1, cfg.embed_dim do
            local sum = block.mlp.fc2_bias:get(d)
            for j = 1, 4 * cfg.embed_dim do
                 sum = sum + relu_out[j-1] * block.mlp.fc2:get(j, d)
            end
            fc2_out[d-1] = sum
        end

        -- MLP Dropout (does nothing in inference)
        mlp_outputs[t], _ = dropout_forward(fc2_out, cfg.embed_dim, cfg.dropout)
    end

    -- Second residual connection
    local out_tokens_data = {}
    for t = 1, seq_len do
        out_tokens_data[t] = ffi.new("double[?]", cfg.embed_dim)
        for d = 0, cfg.embed_dim-1 do
            out_tokens_data[t][d] = res1[t][d] + mlp_outputs[t][d] -- Add MLP output to first residual
        end
    end

    return out_tokens_data
end

-- Forward pass for the full model (simplified for generation)
local function forward(input_tokens_single_batch)
    local seq_len = #input_tokens_single_batch
    local activations = {} -- Stores activations layer by layer

    -- 1. Embeddings
    activations[1] = {} -- Layer 0 = embeddings
    for t = 1, seq_len do
        local emb = ffi.new("double[?]", cfg.embed_dim)
        local token_id = input_tokens_single_batch[t]
        local pos_id = t -- Position embedding index (1-based)

        -- Check bounds before accessing embeddings
        if token_id < 1 or token_id > GPT.wte.rows then
             error(string.format("Token ID %d out of bounds (Vocab Size: %d)", token_id, GPT.wte.rows))
        end
        if pos_id < 1 or pos_id > GPT.wpe.rows then
             error(string.format("Position ID %d out of bounds (Max Seq Len: %d)", pos_id, GPT.wpe.rows))
        end

        for d = 1, cfg.embed_dim do
            emb[d-1] = GPT.wte:get(token_id, d) + GPT.wpe:get(pos_id, d)
        end
        activations[1][t] = emb
    end

    -- 2. Transformer Blocks
    for layer = 1, cfg.num_layers do
        local current_activations = activations[layer]
        local norm_tokens = {}
        for t = 1, seq_len do
            norm_tokens[t] = layer_norm_forward(current_activations[t], cfg.embed_dim)
        end

        -- Pass through transformer block
        local block_out = transformer_block_forward(GPT.blocks[layer], norm_tokens, GPT.relative_pos_bias)
        activations[layer+1] = block_out -- Store output for next layer
    end

    -- 3. Final Layer Norm (Optional, depends on model architecture - assuming needed before head)
    local final_layer_idx = cfg.num_layers + 1
    local final_activations = activations[final_layer_idx]
    local final_norm_tokens = {}
    for t=1, seq_len do
        final_norm_tokens[t] = layer_norm_forward(final_activations[t], cfg.embed_dim)
    end


    -- 4. Projection Head (Logits)
    local logits = {}
    for t = 1, seq_len do
        logits[t] = ffi.new("double[?]", cfg.vocab_size) -- Vocab size from loaded config
        local token_act = final_norm_tokens[t] -- Use final normalized activations
        for v = 1, cfg.vocab_size do
            local sum = GPT.head_bias:get(v)
            for d = 1, cfg.embed_dim do
                sum = sum + token_act[d-1] * GPT.head:get(d, v)
            end
            logits[t][v-1] = sum
        end
    end

    return logits -- Return logits for the entire sequence
end

-- Softmax function (needed for sampling)
local function softmax(logits_vec, temperature)
    temperature = temperature or 1.0
    local size = ffi.sizeof(logits_vec) / ffi.sizeof("double")
    local probs = ffi.new("double[?]", size)
    local max_logit = -math.huge

    for i = 0, size - 1 do
        if logits_vec[i] > max_logit then
            max_logit = logits_vec[i]
        end
    end

    local sum_exp = 0
    for i = 0, size - 1 do
        local exp_val = math.exp((logits_vec[i] - max_logit) / temperature)
        probs[i] = exp_val
        sum_exp = sum_exp + exp_val
    end

    if sum_exp > 0 then
        for i = 0, size - 1 do
            probs[i] = probs[i] / sum_exp
        end
    else -- Handle underflow or all-negative-infinity cases
         for i = 0, size - 1 do
             probs[i] = 1.0 / size -- Uniform distribution as fallback
         end
    end
    return probs
end

-- Tokenizer (using loaded vocab)
local function tokenize(text)
    local tokens = {}
    local current_text = text
    
    while #current_text > 0 do
        local best_match = ""
        local best_id = nil
        
        -- Look for multi-character matches first
        for len = math.min(4, #current_text), 1, -1 do
            local substr = string.sub(current_text, 1, len)
            if vocab[substr] then
                best_match = substr
                best_id = vocab[substr]
                break
            end
        end
        
        -- Fallback to single-byte encoding
        if not best_id then
            local byte = string.sub(current_text, 1, 1)
            best_id = string.byte(byte) + 1  -- Store as 1-based index
            best_match = byte
        end
        
        -- Validate ID before adding
        if best_id and best_id >= 1 and best_id <= cfg.vocab_size then
            table.insert(tokens, best_id)
            current_text = string.sub(current_text, #best_match + 1)
        else
            -- Error recovery: skip invalid character
            current_text = string.sub(current_text, 2)
        end
    end
    
    return tokens
end


-- Detokenizer (using loaded idx_to_word)
-- In query.lua
local function detokenize(tokens)
    local parts = {} -- Use a table to collect parts, then concat once
    for i, token_id in ipairs(tokens) do
        local word = idx_to_word[token_id]
        if word then
            table.insert(parts, word)
        else
            -- If token ID not found, insert a placeholder
            table.insert(parts, string.format("[UNK_ID_%d]", token_id))
            -- Optional: print warning only once per ID maybe
            -- print(string.format("Warning: Unknown token ID during detokenization: %d", token_id))
        end
    end
    return table.concat(parts) -- Concatenate all parts at the end
end

-- Load Model function (adapted for inference)
-- query.lua function (Already incorporates size detection)

local function load_model(db_path)
    db = sqlite3.open(db_path)
    if not db then error("Failed to open database: " .. db_path) end

    -- Load configuration with type conversion
    local stmt_cfg = db:prepare("SELECT key, value FROM config")
    while stmt_cfg:step() == sqlite3.ROW do
        local key = stmt_cfg:get_value(0)
        local value = stmt_cfg:get_value(1)
        
        -- Handle numeric and boolean values
        if key == "relative_pos_bias" or key == "mixed_precision" then
            cfg[key] = (tonumber(value) == 1)
        else
            cfg[key] = tonumber(value) or value
        end
    end
    stmt_cfg:finalize()

    -- Calculate actual vocabulary size
    local max_vocab_id = 0
    local stmt_vocab = db:prepare("SELECT MAX(id) FROM vocab")
    if stmt_vocab:step() == sqlite3.ROW then
        max_vocab_id = tonumber(stmt_vocab:get_value(0)) or 0
    end
    stmt_vocab:finalize()

    -- Ensure complete byte coverage
    cfg.vocab_size = math.max(256, max_vocab_id)
    
    -- Initialize vocabulary with guaranteed byte coverage
    vocab = {}
    idx_to_word = {}
    for i=0,255 do
        local byte = string.char(i)
        vocab[byte] = i + 1
        idx_to_word[i + 1] = byte
    end

    -- Load merged tokens from DB
    local stmt_merged = db:prepare("SELECT word, id FROM vocab WHERE LENGTH(word) > 1")
    while stmt_merged:step() == sqlite3.ROW do
        local word = stmt_merged:get_value(0)
        local id = stmt_merged:get_value(1)
        if id > 256 then  -- Reserve 1-256 for single bytes
            vocab[word] = id
            idx_to_word[id] = word
        end
    end
    stmt_merged:finalize()

    -- Model structure initialization
    GPT = {
        wte = create_tensor(cfg.vocab_size, cfg.embed_dim),
        wpe = create_tensor(cfg.seq_len, cfg.embed_dim),
        blocks = {},
        head = create_tensor(cfg.embed_dim, cfg.vocab_size),
        head_bias = create_bias(cfg.vocab_size)
    }

    -- Load embeddings with bounds checking
    local stmt_embed = db:prepare("SELECT type, position, dim, value FROM embeddings")
    while stmt_embed:step() == sqlite3.ROW do
        local embed_type = stmt_embed:get_value(0)
        local pos = stmt_embed:get_value(1)
        local dim = stmt_embed:get_value(2)
        local val = stmt_embed:get_value(3)
        
        if embed_type == "wte" and pos <= cfg.vocab_size and dim <= cfg.embed_dim then
            GPT.wte:set(pos, dim, val)
        elseif embed_type == "wpe" and pos <= cfg.seq_len and dim <= cfg.embed_dim then
            GPT.wpe:set(pos, dim, val)
        end
    end
    stmt_embed:finalize()

    -- Load transformer layers
    local stmt_layers = db:prepare([[
        SELECT layer, component, i, j, value 
        FROM layers 
        ORDER BY layer, component, i, j
    ]])
    while stmt_layers:step() == sqlite3.ROW do
        local layer_num = stmt_layers:get_value(0)
        local component = stmt_layers:get_value(1)
        local i = stmt_layers:get_value(2)
        local j = stmt_layers:get_value(3)
        local value = stmt_layers:get_value(4)

        if layer_num == 0 then
            -- Handle projection head
            if component == "head" and i <= cfg.embed_dim and j <= cfg.vocab_size then
                GPT.head:set(i, j, value)
            elseif component == "head_bias" and i <= cfg.vocab_size then
                GPT.head_bias:set(i, value)
            end
        elseif layer_num > 0 and layer_num <= cfg.num_layers then
            -- Initialize layer if not exists
            GPT.blocks[layer_num] = GPT.blocks[layer_num] or {
                attn = {
                    q = create_tensor(cfg.embed_dim, cfg.embed_dim),
                    k = create_tensor(cfg.embed_dim, cfg.embed_dim),
                    v = create_tensor(cfg.embed_dim, cfg.embed_dim),
                    proj = create_tensor(cfg.embed_dim, cfg.embed_dim),
                    q_bias = create_bias(cfg.embed_dim),
                    k_bias = create_bias(cfg.embed_dim),
                    v_bias = create_bias(cfg.embed_dim),
                    proj_bias = create_bias(cfg.embed_dim)
                },
                mlp = {
                    fc1 = create_tensor(cfg.embed_dim, 4 * cfg.embed_dim),
                    fc2 = create_tensor(4 * cfg.embed_dim, cfg.embed_dim),
                    fc1_bias = create_bias(4 * cfg.embed_dim),
                    fc2_bias = create_bias(cfg.embed_dim)
                }
            }

            -- Set layer values
            local block = GPT.blocks[layer_num]
            if block.attn[component] then
                if j then  -- Weight matrix
                    block.attn[component]:set(i, j, value)
                else  -- Bias vector
                    block.attn[component]:set(i, value)
                end
            elseif block.mlp[component] then
                if j then  -- Weight matrix
                    block.mlp[component]:set(i, j, value)
                else  -- Bias vector
                    block.mlp[component]:set(i, value)
                end
            end
        end
    end
    stmt_layers:finalize()

    -- Load relative position bias if enabled
    if cfg.relative_pos_bias then
        GPT.relative_pos_bias = create_relative_position_bias()
        local stmt_relpos = db:prepare("SELECT bucket, head, value FROM relative_pos_bias")
        while stmt_relpos:step() == sqlite3.ROW do
            local bucket = stmt_relpos:get_value(0)
            local head = stmt_relpos:get_value(1)
            local value = stmt_relpos:get_value(2)
            if bucket <= cfg.num_relative_pos_buckets and head <= cfg.num_heads then
                GPT.relative_pos_bias:set(bucket, head, value)
            end
        end
        stmt_relpos:finalize()
    end

    db:close()
    return true
end


local function sample_top_k(probs, k)
    local top_probs = {}
    for i = 0, cfg.vocab_size do
        table.insert(top_probs, {idx = i, prob = probs[i]})
    end
    table.sort(top_probs, function(a, b) return a.prob > b.prob end)
    
    local sum = 0
    for i = 1, math.min(k, #top_probs) do
        sum = sum + top_probs[i].prob
    end
    
    local sampled = math.random() * sum
    local cumulative = 0
    for i = 1, math.min(k, #top_probs) do
        cumulative = cumulative + top_probs[i].prob
        if cumulative >= sampled then
            return top_probs[i].idx
        end
    end
    return top_probs[1].idx
end

local function sample_top_p(probs, p)
    local sorted = {}
    for i = 0, cfg.vocab_size do
        table.insert(sorted, {idx = i, prob = probs[i]})
    end
    table.sort(sorted, function(a, b) return a.prob > b.prob end)
    
    local cumulative = 0
    local cutoff = 0
    for i = 1, #sorted do
        cumulative = cumulative + sorted[i].prob
        if cumulative >= p then
            cutoff = i
            break
        end
    end
    
    local sum = cumulative
    local sampled = math.random() * sum
    cumulative = 0
    for i = 1, cutoff do
        cumulative = cumulative + sorted[i].prob
        if cumulative >= sampled then
            return sorted[i].idx
        end
    end
    return sorted[1].idx
end


-- Generation function
local function generate(prompt, max_new_tokens, temperature)
    temperature = temperature or 1.0
    math.randomseed(os.time()) -- Seed random number generator

    print("Tokenizing prompt...")
    local input_tokens = tokenize(prompt)
    if #input_tokens == 0 then
         print("Warning: Prompt tokenized to empty sequence.")
         return prompt -- Return original prompt if tokenization fails
    end
    print(string.format("Prompt tokens (%d): %s", #input_tokens, table.concat(input_tokens, " ")))

    local generated_tokens = {} -- Store only the newly generated tokens

    print("Generating text...")
    local current_sequence = input_tokens
    for i = 1, max_new_tokens do
        -- Truncate sequence if it exceeds max length
        local sequence_to_process = current_sequence
        if #sequence_to_process > cfg.seq_len then
            sequence_to_process = {}
            for j = #current_sequence - cfg.seq_len + 1, #current_sequence do
                table.insert(sequence_to_process, current_sequence[j])
            end
             -- print("Truncated sequence to:", table.concat(sequence_to_process, " ")) -- Debug
        end

        -- Perform forward pass
        local logits_all_tokens = forward(sequence_to_process)

        -- Get logits for the *last* token in the sequence
        local last_token_logits = logits_all_tokens[#sequence_to_process]

        -- Apply softmax with temperature to get probabilities
        local probs = softmax(last_token_logits, temperature)

        -- Sample the next token
        local rand_val = math.random()
        local cumulative_prob = 0
        local next_token = -1
        for token_idx = 0, cfg.vocab_size - 1 do -- Iterate 0 to vocab_size-1
            cumulative_prob = cumulative_prob + probs[token_idx]
            if rand_val <= cumulative_prob then
                next_token = token_idx + 1 -- Convert 0-based index back to 1-based token ID
                break
            end
        end

        if next_token == -1 then -- Should not happen if probs sum to 1
            print("Warning: Sampling failed, defaulting to token 0")
            next_token = 1 -- Default to first token (or handle as EOS)
        end

        -- Check for End-of-Sequence (if you have a specific EOS token ID)
        -- if next_token == EOS_TOKEN_ID then break end

        -- Add the sampled token to the sequence and generated list
        table.insert(current_sequence, next_token)
        table.insert(generated_tokens, next_token)

        -- Optional: Print generated token in real-time
        -- local next_word = idx_to_word[next_token] or "[UNK]"
        -- io.write(next_word) io.flush()
        if i % 10 == 0 then io.write(".") io.flush() end -- Progress indicator

    end
    print("\nGeneration finished.")

    -- Detokenize the generated part
    local generated_text = detokenize(generated_tokens)
    return prompt .. generated_text
end

--------------------------------------------------
-- Main Execution Logic
--------------------------------------------------
local function run_query()
    -- Get prompt from command line arguments
    if #arg < 1 then
        print("Usage: luajit query.lua \"Your prompt goes here\" [max_new_tokens] [temperature]")
        print("\nExample: luajit query.lua \"The meaning of life is\" 50 0.8")
        return
    end
    local prompt = arg[1]
    local max_new_tokens = tonumber(arg[2]) or 100 -- Default to 100 tokens
    local temperature = tonumber(arg[3]) or 0.8  -- Default temperature

    print("Model: " .. cfg.model_db)
    print("Prompt: \"" .. prompt .. "\"")
    print("Max New Tokens: " .. max_new_tokens)
    print("Temperature: " .. temperature)

    -- Check if model DB exists
    local f = io.open(cfg.model_db, "r")
    if not f then
        print("Error: Model database file not found: " .. cfg.model_db)
        print("Please ensure the model has been trained and saved.")
        return
    end
    f:close()

    local load_start = os.clock()
    load_model(cfg.model_db)
    local load_end = os.clock()
    print(string.format("Model loaded in %.2f seconds.", load_end - load_start))

    local gen_start = os.clock()
    local generated_text = generate(prompt, max_new_tokens, temperature)
    local gen_end = os.clock()
    print(string.format("Text generated in %.2f seconds.", gen_end - gen_start))

    print("\n--- Generated Text ---")
    print(generated_text)
    print("----------------------")

    -- Close the database connection
    if db then
        db:close()
        print("Database closed.")
    end
end

-- Run the main query function
run_query()
