-- query.lua: Load a pre-trained GPT model and generate text based on user prompts.

local sqlite3 = require('lsqlite3')
local math = require('math')
local ffi = require('ffi')
local os = require('os') -- For os.clock, os.time

ffi.cdef[[
    void free(void *ptr);
    void* malloc(size_t size);
]]

-- Configuration (Structural parameters needed before loading, others loaded from DB)
local cfg = {
    -- Essential structural params (defaults, will be overwritten by loaded config if possible)
    embed_dim = 256,
    num_heads = 8,
    num_layers = 6,
    seq_len = 256,
    relative_pos_bias = true,
    num_relative_pos_buckets = 32,
    max_distance = 128,
    vocab_size = 256, -- Placeholder, will be updated after loading vocab

    -- Database file
    model_db = 'gpt_model.db',

    -- Generation defaults (can be changed by user)
    default_max_new_tokens = 100,
    default_temperature = 0.8,

    -- Internal: Ensure dropout is off for inference
    dropout = 0.0 -- Set dropout to 0 for generation
}

local db -- Database connection handle
local GPT = {} -- Model parameters structure
local vocab = {} -- word -> token_id mapping
local idx_to_word = {} -- token_id -> word mapping

--------------------------------------------------
-- Utility functions (Copied/Adapted from training script)
--------------------------------------------------

-- Layer normalization forward returns normalized vector and cache
-- NOTE: Cache is not strictly needed for inference but kept for compatibility
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
    local std_dev = math.sqrt(variance + eps)
    for i = 0, size - 1 do
        -- Avoid division by zero if std_dev is extremely small
        if std_dev > 1e-10 then
             norm[i] = (vec[i] - mean) / std_dev
        else
             norm[i] = 0.0 -- Or handle as appropriate
        end
    end
    -- Cache is returned but not used in typical inference-only paths
    local cache = {mean = mean, variance = variance, size = size, eps = eps, vec = vec}
    return norm, cache
end

-- Dropout forward (Modified for Inference: always pass through)
local function dropout_forward(vec, size, dropout_rate)
    -- During inference (dropout_rate = 0 or cfg.dropout = 0),
    -- dropout should effectively do nothing.
    if dropout_rate > 0 and cfg.dropout > 0 then -- Check cfg.dropout as well
        -- Original dropout logic (kept for completeness, but shouldn't run in query.lua)
        local out = ffi.new("double[?]", size)
        local mask = {} -- Not used in inference
        for i = 0, size - 1 do
            if math.random() < dropout_rate then
                out[i] = 0
                mask[i] = 0
            else
                out[i] = vec[i] / (1 - dropout_rate) -- Inverted dropout
                mask[i] = 1
            end
        end
        return out, mask
    else
        -- Inference mode: pass through directly
        return vec, {} -- Return original vector and empty mask
    end
end


-- ReLU forward returns output
-- NOTE: Mask is not strictly needed for inference
local function relu_forward(input, size)
    local out = ffi.new("double[?]", size)
    local mask = {} -- Not used in inference
    for i = 0, size - 1 do
        if input[i] > 0 then
            out[i] = input[i]
            mask[i] = 1
        else
            out[i] = 0
            mask[i] = 0
        end
    end
    return out, mask
end

--------------------------------------------------
-- Tensor creation helper (Simplified for Inference - no grad/optimizer state)
--------------------------------------------------
local function create_tensor(rows, cols)
    local size = rows * cols
    local data = ffi.new("double[?]", size)
    ffi.fill(data, ffi.sizeof("double") * size, 0)
    return {
        data = data,
        rows = rows,
        cols = cols,
        get = function(self, i, j)
            -- Basic bounds check (optional but safer)
            if i < 1 or i > self.rows or j < 1 or j > self.cols then
                error(string.format("Tensor get out of bounds: accessing (%d, %d) in [%d x %d]", i, j, self.rows, self.cols))
            end
            return self.data[(i-1)*self.cols + (j-1)]
        end,
        set = function(self, i, j, val)
             -- Basic bounds check (optional but safer)
            if i < 1 or i > self.rows or j < 1 or j > self.cols then
                 error(string.format("Tensor set out of bounds: accessing (%d, %d) in [%d x %d]", i, j, self.rows, self.cols))
            end
            self.data[(i-1)*self.cols + (j-1)] = val
        end,
        -- No grad functions needed for inference
        zero_grad = function() end,
        add_grad = function() end
    }
end

local function create_bias(size)
    local data = ffi.new("double[?]", size)
    ffi.fill(data, ffi.sizeof("double") * size, 0)
    return {
        data = data,
        size = size,
        get = function(self, i)
             if i < 1 or i > self.size then
                 error(string.format("Bias get out of bounds: accessing (%d) in size [%d]", i, self.size))
            end
            return self.data[i-1]
        end,
        set = function(self, i, val)
            if i < 1 or i > self.size then
                error(string.format("Bias set out of bounds: accessing (%d) in size [%d]", i, self.size))
            end
            self.data[i-1] = val
        end,
        -- No grad functions needed for inference
        zero_grad = function() end,
        add_grad = function() end
    }
end

--------------------------------------------------
-- Relative Position Bias (Creation and Bucketing)
--------------------------------------------------
local function create_relative_position_bias()
    -- Use config values that *should* be loaded from the DB
    local num_buckets = cfg.num_relative_pos_buckets
    local num_heads = cfg.num_heads
    local size = num_buckets * num_heads
    local data = ffi.new("double[?]", size)
    ffi.fill(data, ffi.sizeof("double") * size, 0)

    return {
        data = data,
        num_buckets = num_buckets,
        num_heads = num_heads,
        get = function(self, bucket, head)
            if bucket < 1 or bucket > self.num_buckets or head < 1 or head > self.num_heads then
                 error(string.format("RelPosBias get out of bounds: accessing bucket %d, head %d in [%d buckets x %d heads]", bucket, head, self.num_buckets, self.num_heads))
            end
            return self.data[(bucket-1) * self.num_heads + (head-1)]
        end,
        set = function(self, bucket, head, val)
             if bucket < 1 or bucket > self.num_buckets or head < 1 or head > self.num_heads then
                 error(string.format("RelPosBias set out of bounds: accessing bucket %d, head %d in [%d buckets x %d heads]", bucket, head, self.num_buckets, self.num_heads))
            end
            self.data[(bucket-1) * self.num_heads + (head-1)] = val
        end,
        zero_grad = function() end,
        add_grad = function() end
    }
end

-- Accurate relative_position_bucket function from training script
local function relative_position_bucket(relative_position, bidirectional, num_buckets, max_distance)
    local ret = 0
    -- Handle zero relative position explicitly if needed, although log(0) is problematic.
    -- Assuming relative_position != 0 for the log calculation.
    -- The logic below seems to assume T5-style bucketing.

    local num_buckets_half = num_buckets / 2
    local rp_abs = math.abs(relative_position)

    local is_small = rp_abs < num_buckets_half

    local rp_log = math.log(rp_abs / num_buckets_half) / math.log(max_distance / num_buckets_half) * (num_buckets - num_buckets_half)
    rp_log = math.floor(rp_log + num_buckets_half) -- Add offset

    if is_small then
        ret = rp_abs
    else
        ret = rp_log
    end

    ret = math.min(ret, num_buckets - 1) -- Clamp to max bucket index

    if not bidirectional then
        if relative_position > 0 then
             -- No offset needed for positive positions in unidirectional case based on common impl.
        else
            -- Shift negative positions if needed by specific unidirectional scheme,
            -- often they are simply ignored or handled differently.
            -- Assuming simple positive offset mapping here if needed:
            -- ret = ret + num_buckets -- Example shift
             ret = ret + num_buckets_half -- If using half for negative like T5 bidirectional
        end
    else -- Bidirectional
         if relative_position < 0 then
              ret = ret + num_buckets -- T5 adds full num_buckets for negative
         end
    end

     -- Ensure final result is within 0 to num_buckets - 1 range, then add 1 for 1-based Lua indexing
     ret = math.max(0, math.min(num_buckets -1, math.floor(ret)))

    return ret + 1 -- Return 1-based index
end


--------------------------------------------------
-- Transformer block constructor (Creates structure for loading)
--------------------------------------------------
local function transformer_block()
    local embed_dim = cfg.embed_dim -- Use potentially loaded config
    local head_dim = embed_dim / cfg.num_heads
    if embed_dim % cfg.num_heads ~= 0 then
        error("embed_dim must be divisible by num_heads")
    end

    -- Attention components
    local attn = {
        q = create_tensor(embed_dim, embed_dim),
        k = create_tensor(embed_dim, embed_dim),
        v = create_tensor(embed_dim, embed_dim),
        proj = create_tensor(embed_dim, embed_dim),
        q_bias = create_bias(embed_dim),
        k_bias = create_bias(embed_dim),
        v_bias = create_bias(embed_dim),
        proj_bias = create_bias(embed_dim)
    }

    -- MLP components
    local mlp = {
        fc1 = create_tensor(embed_dim, 4 * embed_dim),
        fc2 = create_tensor(4 * embed_dim, embed_dim),
        fc1_bias = create_bias(4 * embed_dim),
        fc2_bias = create_bias(embed_dim)
    }

    -- No random initialization here, weights will be loaded
    return { attn = attn, mlp = mlp }
end

--------------------------------------------------
-- Transformer block forward (Inference optimized - no cache needed)
--------------------------------------------------
local function transformer_block_forward(block, norm_tokens_input, relative_pos_bias_tensor)
    local embed_dim = cfg.embed_dim
    local num_heads = cfg.num_heads
    local head_dim = embed_dim / num_heads
    local seq_len = #norm_tokens_input
    local final_outputs = {} -- Stores the output for each token position

    -- Prepare input (copy or reference?) Let's assume norm_tokens_input is read-only
    local current_tokens = {}
    for t=1, seq_len do
        current_tokens[t] = norm_tokens_input[t] -- Use the layer-normed input from previous step/layer
    end

    -- Layer Norm 1 (Applied before attention in this structure)
    local norm1_tokens = {}
    for t = 1, seq_len do
        -- Note: The training script applies norm *before* the block.
        -- If loading a model trained that way, this norm might be redundant
        -- or applied differently (e.g., after residual). Assuming input `norm_tokens_input`
        -- is the output of the *previous* layer or embedding, and norm is applied first here.
        -- Let's follow the training script structure: norm happens *before* calling this.
        -- So, norm_tokens_input *is* already normalized.
        norm1_tokens[t] = norm_tokens_input[t] -- Directly use the pre-normalized input
    end


    -- === Multi-Head Self-Attention ===
    local attn_outputs_concat = {} -- Store concatenated outputs for each token
    for t = 1, seq_len do
        attn_outputs_concat[t] = ffi.new("double[?]", embed_dim)
        ffi.fill(attn_outputs_concat[t], ffi.sizeof("double")*embed_dim, 0.0)
    end

    for h = 1, num_heads do
        local q_proj = block.attn.q
        local k_proj = block.attn.k
        local v_proj = block.attn.v
        local q_bias = block.attn.q_bias
        local k_bias = block.attn.k_bias
        local v_bias = block.attn.v_bias

        local queries = {}
        local keys = {}
        local values = {}

        -- Project Q, K, V for all tokens for this head
        for t = 1, seq_len do
            local token_input = norm1_tokens[t] -- Use normalized input

            local q_h = ffi.new("double[?]", head_dim)
            local k_h = ffi.new("double[?]", head_dim)
            local v_h = ffi.new("double[?]", head_dim)

            for d = 1, head_dim do
                local head_offset = (h-1) * head_dim
                local q_sum, k_sum, v_sum = 0.0, 0.0, 0.0
                q_sum = q_bias:get(head_offset + d) -- Add bias first
                k_sum = k_bias:get(head_offset + d)
                v_sum = v_bias:get(head_offset + d)

                for i = 1, embed_dim do
                    q_sum = q_sum + token_input[i-1] * q_proj:get(i, head_offset + d)
                    k_sum = k_sum + token_input[i-1] * k_proj:get(i, head_offset + d)
                    v_sum = v_sum + token_input[i-1] * v_proj:get(i, head_offset + d)
                end
                q_h[d-1] = q_sum
                k_h[d-1] = k_sum
                v_h[d-1] = v_sum
            end
            queries[t] = q_h
            keys[t] = k_h
            values[t] = v_h
        end

        -- Calculate attention scores and apply softmax for each query token
        for i = 1, seq_len do -- For each query token 'i'
            local q_i = queries[i]
            local scores = ffi.new("double[?]", seq_len)
            local max_score = -math.huge

            -- Calculate raw scores (dot product + relative pos bias)
            for j = 1, seq_len do -- For each key token 'j'
                 if j <= i then -- Apply causal mask
                    local score = 0.0
                    for d = 0, head_dim - 1 do
                        score = score + q_i[d] * keys[j][d]
                    end
                    score = score / math.sqrt(head_dim) -- Scale

                    -- Add relative position bias if enabled
                    if cfg.relative_pos_bias and relative_pos_bias_tensor then
                        local relative_pos = i - j -- Query pos - Key pos
                        local bucket = relative_position_bucket(relative_pos, false, cfg.num_relative_pos_buckets, cfg.max_distance)
                        score = score + relative_pos_bias_tensor:get(bucket, h)
                    end
                    scores[j-1] = score
                    if score > max_score then max_score = score end
                else
                     scores[j-1] = -math.huge -- Mask future tokens
                 end
            end

            -- Softmax
            local sum_exp = 0.0
            local attn_weights = ffi.new("double[?]", seq_len)
            for j = 0, seq_len - 1 do
                if scores[j] > -math.huge then -- Avoid exp(-inf) -> 0
                    local exp_val = math.exp(scores[j] - max_score) -- Stable softmax
                    attn_weights[j] = exp_val
                    sum_exp = sum_exp + exp_val
                else
                    attn_weights[j] = 0.0
                end
            end

            -- Normalize weights
             if sum_exp > 1e-9 then -- Avoid division by zero
                for j = 0, seq_len - 1 do
                    attn_weights[j] = attn_weights[j] / sum_exp
                end
            -- else: weights remain 0 if all scores were -inf or sum_exp is tiny
            end

            -- Calculate weighted sum of values
            local head_output = ffi.new("double[?]", head_dim)
            ffi.fill(head_output, ffi.sizeof("double")*head_dim, 0.0)
            for j = 1, seq_len do -- For each value token 'j'
                local weight = attn_weights[j-1]
                 if weight > 0 then -- Optimization: skip if weight is zero
                    local v_j = values[j]
                    for d = 0, head_dim - 1 do
                        head_output[d] = head_output[d] + weight * v_j[d]
                    end
                end
            end

            -- Add this head's output to the correct slice of the concatenated output
             local head_offset = (h - 1) * head_dim
             for d = 0, head_dim - 1 do
                 attn_outputs_concat[i][head_offset + d] = head_output[d]
             end
        end -- End loop over query tokens 'i'
    end -- End loop over heads 'h'

    -- Attention Projection
    local proj_outputs = {}
    for t = 1, seq_len do
        local concat_output = attn_outputs_concat[t]
        local proj_out_t = ffi.new("double[?]", embed_dim)
        local proj_matrix = block.attn.proj
        local proj_bias = block.attn.proj_bias

        for d = 1, embed_dim do
            local sum = proj_bias:get(d) -- Add bias
            for i = 1, embed_dim do
                sum = sum + concat_output[i-1] * proj_matrix:get(i, d)
            end
            proj_out_t[d-1] = sum
        end
         -- Apply dropout (should be identity in inference)
        local dropped_proj, _ = dropout_forward(proj_out_t, embed_dim, cfg.dropout)
        proj_outputs[t] = dropped_proj
    end

     -- Residual Connection 1
     local res1_outputs = {}
     for t=1, seq_len do
         res1_outputs[t] = ffi.new("double[?]", embed_dim)
         local original_input = norm_tokens_input[t] -- Input before the first norm
         local proj_output = proj_outputs[t]
         for d=0, embed_dim-1 do
             res1_outputs[t][d] = original_input[d] + proj_output[d]
         end
     end

    -- === Feed-Forward Network ===
    -- Layer Norm 2 (Applied before MLP)
    local norm2_tokens = {}
    for t = 1, seq_len do
         norm2_tokens[t], _ = layer_norm_forward(res1_outputs[t], embed_dim)
    end

    local mlp_outputs = {}
    for t = 1, seq_len do
         local norm2_token = norm2_tokens[t]
         local mlp_block = block.mlp

         -- FC1 + Bias + ReLU
         local fc1_out_raw = ffi.new("double[?]", 4 * embed_dim)
         for j=1, 4 * embed_dim do
             local sum = mlp_block.fc1_bias:get(j) -- Add bias
             for i=1, embed_dim do
                 sum = sum + norm2_token[i-1] * mlp_block.fc1:get(i,j)
             end
             fc1_out_raw[j-1] = sum
         end
         local relu_out, _ = relu_forward(fc1_out_raw, 4 * embed_dim)

         -- FC2 + Bias
         local fc2_out_raw = ffi.new("double[?]", embed_dim)
         for j=1, embed_dim do
             local sum = mlp_block.fc2_bias:get(j) -- Add bias
             for i=1, 4 * embed_dim do
                 sum = sum + relu_out[i-1] * mlp_block.fc2:get(i,j)
             end
             fc2_out_raw[j-1] = sum
         end

         -- Apply dropout (should be identity in inference)
        local dropped_mlp, _ = dropout_forward(fc2_out_raw, embed_dim, cfg.dropout)
        mlp_outputs[t] = dropped_mlp
    end

    -- Residual Connection 2
    local final_block_outputs = {}
     for t=1, seq_len do
         final_block_outputs[t] = ffi.new("double[?]", embed_dim)
         local res1_output = res1_outputs[t] -- Output of the first residual connection
         local mlp_output = mlp_outputs[t]
         for d=0, embed_dim-1 do
             final_block_outputs[t][d] = res1_output[d] + mlp_output[d]
         end
         -- Store the final output for this token position
         final_outputs[t] = { data = final_block_outputs[t] } -- Match expected structure
     end

    -- Return the final outputs for all token positions for this block
    -- The cache is not needed for inference, return empty table
    return final_outputs, {}
end


--------------------------------------------------
-- Forward pass for the full model (Simplified for single sequence inference)
--------------------------------------------------
local function forward_single(input_token_ids)
    local seq_len = #input_token_ids
    if seq_len > cfg.seq_len then
        -- If input is longer than model's trained seq_len, truncate (keep most recent)
        local start_idx = seq_len - cfg.seq_len + 1
        input_token_ids = { table.unpack(input_token_ids, start_idx, seq_len) }
        seq_len = cfg.seq_len -- Update seq_len to the truncated length
    end

    -- 1. Embeddings (Token + Position)
    local current_activations = {} -- Stores {data = ffi_array} for each position
    for t = 1, seq_len do
        local token_id = input_token_ids[t]
        local pos_id = t -- Position embedding index (1-based)

        -- Check token_id bounds
        if token_id < 1 or token_id > GPT.wte.rows then
             error(string.format("Invalid token ID %d encountered (vocab size: %d). Check tokenizer or input.", token_id, GPT.wte.rows))
        end
        -- Check pos_id bounds
        if pos_id < 1 or pos_id > GPT.wpe.rows then
             -- This happens if input_seq is longer than cfg.seq_len and not truncated
             -- Or if generation exceeds cfg.seq_len without truncation logic in generate()
             error(string.format("Position ID %d out of bounds for positional embeddings (max: %d). Input sequence likely too long.", pos_id, GPT.wpe.rows))
        end


        local emb = ffi.new("double[?]", cfg.embed_dim)
        for d = 1, cfg.embed_dim do
            local token_emb_val = GPT.wte:get(token_id, d)
            local pos_emb_val = GPT.wpe:get(pos_id, d) -- Use pos_id (t)
            emb[d-1] = token_emb_val + pos_emb_val
        end
         current_activations[t] = { data = emb }
    end

    -- 2. Transformer Layers
    for layer_idx = 1, cfg.num_layers do
         local block = GPT.blocks[layer_idx]
         local block_input_norm = {}
         local norm_caches = {} -- Not used, but returned by norm function

          -- Layer Norm before the block
         for t = 1, seq_len do
             block_input_norm[t], norm_caches[t] = layer_norm_forward(current_activations[t].data, cfg.embed_dim)
         end

         -- Pass through transformer block
         local block_output, _ = transformer_block_forward(block, block_input_norm, GPT.relative_pos_bias)
         current_activations = block_output -- Output of this layer becomes input for the next
    end

    -- 3. Final Layer Norm (Optional, depends on architecture, e.g., pre-LN vs post-LN)
    -- Assuming final layer norm before projection based on typical GPT-2 style
    local final_norm_tokens = {}
    for t = 1, seq_len do
        final_norm_tokens[t], _ = layer_norm_forward(current_activations[t].data, cfg.embed_dim)
    end

    -- 4. Projection to Logits
    local logits = {} -- logits[t] will hold the logits for token at position t
    for t = 1, seq_len do
        local token_act = final_norm_tokens[t] -- Use final normalized activation
        local logit_t = ffi.new("double[?]", cfg.vocab_size) -- Size matches loaded vocab
        local head_matrix = GPT.head
        local head_bias = GPT.head_bias

        for v = 1, cfg.vocab_size do
            local sum = head_bias:get(v) -- Add bias
            for d = 1, cfg.embed_dim do
                 -- Ensure head_matrix dimensions match vocab_size
                 if v > head_matrix.cols then
                      error(string.format("Vocab index %d exceeds head matrix columns %d.", v, head_matrix.cols))
                 end
                 sum = sum + token_act[d-1] * head_matrix:get(d, v)
            end
            logit_t[v-1] = sum
        end
        logits[t] = logit_t
    end

    -- Return logits for all positions
    return logits
end

--------------------------------------------------
-- Softmax (Simplified for single logit vector)
--------------------------------------------------
local function softmax_single(logit_vector)
    local size = cfg.vocab_size
    local probs = ffi.new("double[?]", size)
    local max_logit = -math.huge

    -- Find max logit for numerical stability
    for i = 0, size - 1 do
        if logit_vector[i] > max_logit then
            max_logit = logit_vector[i]
        end
    end

    -- Calculate exponentials and sum
    local sum_exp = 0.0
    for i = 0, size - 1 do
        local exp_val = math.exp(logit_vector[i] - max_logit)
        probs[i] = exp_val
        sum_exp = sum_exp + exp_val
    end

    -- Normalize to get probabilities
    if sum_exp > 1e-9 then -- Avoid division by zero
        for i = 0, size - 1 do
            probs[i] = probs[i] / sum_exp
        end
    else
         -- Handle case where all logits were extremely small or negative infinity
         -- Uniform distribution or error? Let's assign uniform for now.
         local uniform_prob = 1.0 / size
         for i = 0, size - 1 do
              probs[i] = uniform_prob
         end
    end

    return probs
end

--------------------------------------------------
-- Tokenization (Using loaded BPE vocab)
--------------------------------------------------
local function tokenize(text, vocab_map)
     if not vocab or not next(vocab_map) then
         error("Vocabulary is not loaded. Cannot tokenize.")
     end
     if not idx_to_word or not next(idx_to_word) then
         error("Index-to-word map is not loaded. Cannot tokenize/detokenize.")
     end

    -- 1. Initial breakdown into known tokens (longest match)
    local current_tokens = {}
    local i = 1
    while i <= #text do
        local best_match_len = 0
        local best_match_id = -1
        -- Iterate potential lengths from long to short (greedy match)
        for len = math.min(20, #text - i + 1), 1, -1 do -- Limit max token length check
            local sub = string.sub(text, i, i + len - 1)
            if vocab_map[sub] then
                best_match_len = len
                best_match_id = vocab_map[sub]
                break -- Found longest match starting at i
            end
        end

        if best_match_id ~= -1 then
            table.insert(current_tokens, best_match_id)
            i = i + best_match_len
        else
            -- Handle unknown character/byte: Find its byte value if possible
            local byte_char = string.sub(text, i, i)
            local byte_val = string.byte(byte_char)
            if vocab_map[byte_char] then -- Check if single byte is in vocab (should be from BPE init)
                 table.insert(current_tokens, vocab_map[byte_char])
            else
                -- This case should ideally not happen if vocab includes all bytes
                -- Fallback: Skip character or assign an <UNK> token if defined
                print("Warning: Skipping unknown character/byte during tokenization: " .. byte_char)
                -- Alternatively: error("Unknown character and not in byte vocab: " .. byte_char)
            end
            i = i + 1 -- Move to the next character
        end
    end

    -- BPE merging logic (adapted from training - needed if vocab contains merges)
    -- The initial tokenization above might be sufficient if vocab is flat,
    -- but BPE relies on merging bytes. Let's use the merge approach.

    -- Initial Byte Tokenization (Necessary for BPE)
    local byte_tokens = {}
    for k=1, #text do
        local byte_char = string.sub(text, k, k)
         if vocab_map[byte_char] then -- Assumes bytes 0-255 chars are in vocab
             table.insert(byte_tokens, vocab_map[byte_char])
         else
             -- This should not happen if BPE was trained correctly
             print("Warning: Character '"..byte_char.."' not found in byte vocab during initial tokenization.")
             -- Fallback: maybe assign a default/UNK token ID if you have one
         end
    end

    if #byte_tokens == 0 then return {} end -- Handle empty input

    -- Iterative Merging based on learned vocab
    local merged_tokens = byte_tokens
    while true do
        local best_pair_idx = -1
        local best_pair_id = -1
        local min_rank = math.huge -- Assuming merges have ranks; using vocab existence as rank 1

        for k = 1, #merged_tokens - 1 do
             local first_word = idx_to_word[merged_tokens[k]]
             local second_word = idx_to_word[merged_tokens[k+1]]
             if first_word and second_word then
                 local combined_word = first_word .. second_word
                 if vocab_map[combined_word] then
                     -- Found a valid merge based on the loaded vocab
                     -- In a real BPE impl, you'd check rank here. We just take the first found.
                     best_pair_idx = k
                     best_pair_id = vocab_map[combined_word]
                     break -- Perform one merge per pass (simplification)
                 end
            else
                -- This indicates an invalid token ID was somehow generated/present
                error("Invalid token ID found during BPE merge step: " .. tostring(merged_tokens[k]) .. " or " .. tostring(merged_tokens[k+1]))
            end
        end

        if best_pair_idx ~= -1 then
            -- Perform the merge
            local new_merged_tokens = {}
            for k = 1, best_pair_idx - 1 do table.insert(new_merged_tokens, merged_tokens[k]) end
            table.insert(new_merged_tokens, best_pair_id)
            for k = best_pair_idx + 2, #merged_tokens do table.insert(new_merged_tokens, merged_tokens[k]) end
            merged_tokens = new_merged_tokens
        else
            -- No more merges found in this pass
            break
        end
    end

    return merged_tokens
end


--------------------------------------------------
-- Generate text
--------------------------------------------------
local function generate(prompt, max_new_tokens, temperature)
    temperature = temperature or 1.0 -- Default temperature
    if temperature <= 0 then
        print("Warning: Temperature must be positive. Using 1.0.")
        temperature = 1.0
    end

    -- Ensure dropout is off globally in cfg for generation
    local original_dropout = cfg.dropout
    cfg.dropout = 0.0

    local input_tokens = tokenize(prompt, vocab)
    if #input_tokens == 0 and #prompt > 0 then
        print("Warning: Tokenization resulted in empty sequence for non-empty prompt.")
        -- Maybe return empty string or handle differently
    end
    local generated_tokens = {} -- Store only the newly generated tokens

    print("Generating...") -- Feedback

    local current_seq = input_tokens
    for i = 1, max_new_tokens do
        -- Get logits for the *last* token position
        local logits_all_pos = forward_single(current_seq)
        local last_logits = logits_all_pos[#logits_all_pos] -- Get logits for the final position

        -- Apply temperature scaling
        if temperature ~= 1.0 then
            for v = 0, cfg.vocab_size - 1 do
                last_logits[v] = last_logits[v] / temperature
            end
        end

        -- Get probabilities via softmax
        local probs = softmax_single(last_logits)

        -- Sample from the probability distribution
        local rand_val = math.random()
        local cumulative_prob = 0.0
        local next_token_id = -1

        -- Iterate through vocab probabilities to find the sampled token
        for v = 0, cfg.vocab_size - 1 do
             -- Ensure probability is valid
             local p = probs[v]
             if p >= 0 and p <= 1.0001 then -- Allow for slight floating point inaccuracies
                cumulative_prob = cumulative_prob + p
                if rand_val <= cumulative_prob then
                    next_token_id = v + 1 -- Convert 0-based index to 1-based token ID
                    break
                end
             else
                -- This shouldn't happen with correct softmax, indicates potential NaN/Inf issue
                print(string.format("Warning: Invalid probability p=%.5f for token index %d. Skipping.", p, v))
             end
        end


         if next_token_id == -1 then
             -- Fallback: If sampling failed (e.g., due to NaN/Inf or cumulative prob rounding issues),
             -- maybe pick the most likely token (argmax) or stop generation.
             print("Warning: Sampling failed. Picking argmax or stopping.")
             -- Example: Find argmax
             local max_p = -1
             local argmax_id = 1
             for v=0, cfg.vocab_size-1 do
                 if probs[v] > max_p then
                     max_p = probs[v]
                     argmax_id = v + 1
                 end
             end
             next_token_id = argmax_id
             -- Or just break:
             -- break
         end

        -- TODO: Add EOS (End Of Sentence) token handling if your tokenizer/model uses one.
        -- Example: if next_token_id == eos_token_id then break end

        -- Add the new token to the sequence and the generated list
        table.insert(current_seq, next_token_id)
        table.insert(generated_tokens, next_token_id)

        -- Keep the sequence length within the model's limit by removing the oldest token
        if #current_seq > cfg.seq_len then
            table.remove(current_seq, 1)
        end

        -- Provide some progress feedback
        io.write(".") io.flush()
        if i % 50 == 0 then io.write("\n") io.flush() end

    end
    print("\nGeneration complete.")

    -- Convert generated tokens back to text
    local generated_text = ""
    for _, token_id in ipairs(generated_tokens) do
        local word = idx_to_word[token_id]
        if word then
            generated_text = generated_text .. word
        else
            generated_text = generated_text .. "<UNK:" .. token_id .. ">" -- Handle unknown tokens
        end
    end

    -- Restore original dropout setting if necessary (though unlikely needed)
    cfg.dropout = original_dropout

    return generated_text
end

--------------------------------------------------
-- Parameter initialization structure (creates tensors to be filled by load)
--------------------------------------------------
local function init_parameter_structure()
    print("Initializing parameter structures...")
    -- Ensure vocab size is known before creating tensors dependent on it
    if not cfg.vocab_size or cfg.vocab_size <= 256 then -- Use loaded or default if needed
         print("Warning: Vocab size not fully loaded or seems small. Using cfg.vocab_size = " .. cfg.vocab_size)
    end

    GPT.wte = create_tensor(cfg.vocab_size, cfg.embed_dim)
    GPT.wpe = create_tensor(cfg.seq_len, cfg.embed_dim)
    GPT.blocks = {}
    for i = 1, cfg.num_layers do
        GPT.blocks[i] = transformer_block()
    end
    GPT.head = create_tensor(cfg.embed_dim, cfg.vocab_size) -- Size matches vocab
    GPT.head_bias = create_bias(cfg.vocab_size) -- Size matches vocab

    if cfg.relative_pos_bias then
        GPT.relative_pos_bias = create_relative_position_bias()
    end
    print("Parameter structures created.")
end

--------------------------------------------------
-- Load Model from Database
--------------------------------------------------
--------------------------------------------------
-- Load Model from Database (Updated)
--------------------------------------------------
local function load_model_from_db()
    print("Attempting to load model from " .. cfg.model_db .. "...")

    -- Check if DB exists and is readable
    local f = io.open(cfg.model_db, "rb")
    if not f then
        error("Model database file not found or not readable: " .. cfg.model_db)
    end
    f:close()

    -- Open database connection
    local err
    db, err = sqlite3.open(cfg.model_db)
    if not db then
        error("Failed to open database: " .. (err or "unknown error"))
    end
    print("Database opened successfully.")

    -- 1. Load Configuration
    print("Loading configuration...")
    local config_count = 0
    local stmt_cfg = db:prepare("SELECT key, value FROM config")
    if not stmt_cfg then error("Failed to prepare config statement: " .. db:errmsg()) end
    while stmt_cfg:step() == sqlite3.ROW do
        local key = stmt_cfg:get_value(0)
        local value = stmt_cfg:get_value(1) -- Value stored as REAL (float)
        -- Try conversion, handle boolean/integer cases if stored differently
        local num_value = tonumber(value)
        if key == "relative_pos_bias" or key == "learning_rate_decay" or key == "mixed_precision" then
             cfg[key] = (num_value == 1.0) -- Treat 1.0 as true for boolean flags
        elseif num_value then
             -- Check if it's likely an integer
             if math.floor(num_value) == num_value then
                 cfg[key] = math.floor(num_value)
             else
                 cfg[key] = num_value
             end
        else
             cfg[key] = value -- Keep as string if not convertible (shouldn't happen)
        end
        -- print(string.format("Loaded config: %s = %s (type: %s)", key, tostring(cfg[key]), type(cfg[key])))
        config_count = config_count + 1
    end
    stmt_cfg:finalize()
    if config_count == 0 then
        print("Warning: No configuration found in the database.")
        -- Rely on defaults in cfg table (less safe, better to require config)
        -- Consider adding an error here if config is essential
    else
        print(string.format("Loaded %d configuration parameters.", config_count))
    end

    -- *** IMPORTANT CHECK ***: Ensure vocab_size was loaded from config and is valid
    if not cfg.vocab_size or type(cfg.vocab_size) ~= 'number' or cfg.vocab_size <= 0 then
         error("Configuration 'vocab_size' not found in database or is invalid. Cannot proceed.")
    else
         -- This is the vocab size we will trust and use for building model structures
         print(string.format("Using vocab_size from config: %d", cfg.vocab_size))
    end

    -- 2. Load Vocabulary
    print("Loading vocabulary...")
    vocab = {}
    idx_to_word = {}
    local vocab_count = 0
    local stmt_vocab = db:prepare("SELECT word, id FROM vocab")
     if not stmt_vocab then error("Failed to prepare vocab statement: " .. db:errmsg()) end
    while stmt_vocab:step() == sqlite3.ROW do
        local word = stmt_vocab:get_value(0)
        local id = stmt_vocab:get_value(1) -- Stored as INTEGER
        vocab[word] = id
        idx_to_word[id] = word
        vocab_count = vocab_count + 1
    end
    stmt_vocab:finalize()
    if vocab_count == 0 then
        error("Vocabulary table is empty in the database. Cannot proceed.")
    end
    print(string.format("Loaded %d vocabulary entries.", vocab_count))

    -- *** IMPORTANT CHANGE ***:
    -- DO NOT update cfg.vocab_size based on vocab_count.
    -- Trust the value loaded from the config table earlier, as the model weights
    -- depend on the vocab_size used during TRAINING.
    if vocab_count ~= cfg.vocab_size then
        print(string.format("Warning: Actual vocabulary entry count (%d) in DB differs from config vocab_size (%d). Proceeding with config value.", vocab_count, cfg.vocab_size))
        -- We intentionally DO NOT update cfg.vocab_size here.
    end
    -- The value in cfg.vocab_size loaded from the config table will be used below.

    -- 3. Initialize Parameter Structures *AFTER* loading config (esp. sizes)
    -- This will now use the trusted cfg.vocab_size from the config table.
    init_parameter_structure()

    -- 4. Load Embeddings
    print("Loading embeddings (wte, wpe)...")
    local embed_count = 0
    local stmt_embed = db:prepare("SELECT type, position, dim, value FROM embeddings")
     if not stmt_embed then error("Failed to prepare embeddings statement: " .. db:errmsg()) end
    while stmt_embed:step() == sqlite3.ROW do
        local type = stmt_embed:get_value(0)
        local pos = stmt_embed:get_value(1) -- row index (token_id or position)
        local dim = stmt_embed:get_value(2) -- column index (dimension)
        local val = stmt_embed:get_value(3) -- Stored as REAL

        if type == 'wte' then
            -- Only load if within the bounds defined by the trusted cfg.vocab_size
            if pos >= 1 and pos <= GPT.wte.rows and dim >= 1 and dim <= GPT.wte.cols then
                 GPT.wte:set(pos, dim, val)
                 embed_count = embed_count + 1
            else
                -- This warning might be noisy if vocab_count > cfg.vocab_size, consider removing if expected
                -- print(string.format("Debug: Skipping wte embedding at (%d, %d) - potentially outside trained vocab size %d", pos, dim, GPT.wte.rows))
            end
        elseif type == 'wpe' then
             if pos >= 1 and pos <= GPT.wpe.rows and dim >= 1 and dim <= GPT.wpe.cols then
                 GPT.wpe:set(pos, dim, val)
                 embed_count = embed_count + 1
             else print(string.format("Warning: Skipping out-of-bounds wpe embedding at (%d, %d)", pos, dim)) end
        end
    end
    stmt_embed:finalize()
    print(string.format("Loaded %d embedding values.", embed_count))
    -- Basic check: did we load roughly the expected number based on trusted config?
    local expected_embed_vals = (cfg.vocab_size * cfg.embed_dim) + (cfg.seq_len * cfg.embed_dim)
    if math.abs(embed_count - expected_embed_vals) > (cfg.embed_dim * 5) then -- Allow some tolerance
         print(string.format("Warning: Loaded %d embedding values, but expected around %d based on config dimensions. Check DB integrity or dimensions.", embed_count, expected_embed_vals))
    end

    -- 5. Load Layers (Transformer Blocks weights/biases) and Head/Head Bias
    print("Loading transformer layers and projection head...")
    local layer_vals_count = 0
    local stmt_layers = db:prepare("SELECT layer, component, i, j, value FROM layers")
    if not stmt_layers then error("Failed to prepare layers statement: " .. db:errmsg()) end

    while stmt_layers:step() == sqlite3.ROW do
        local layer_num = stmt_layers:get_value(0)      -- INTEGER
        local component_name = stmt_layers:get_value(1) -- TEXT
        local i = stmt_layers:get_value(2)              -- INTEGER (row or bias index)
        local j = stmt_layers:get_value(3)              -- INTEGER (col or 1 for bias)
        local value = stmt_layers:get_value(4)          -- REAL

        if layer_num == 0 then -- Special case for Head layer
            if component_name == 'head' then
                -- Load only if within bounds of head sized by trusted cfg.vocab_size
                if i >= 1 and i <= GPT.head.rows and j >= 1 and j <= GPT.head.cols then
                     GPT.head:set(i, j, value)
                     layer_vals_count = layer_vals_count + 1
                else
                    -- Might indicate mismatch if j > cfg.vocab_size, potential noise
                    -- print(string.format("Debug: Skipping head weight at (%d, %d) - potentially outside trained vocab size %d", i, j, GPT.head.cols))
                end
            elseif component_name == 'head_bias' then
                 -- Bias stored with j=1 in save logic, access using 'i'
                 -- Load only if within bounds of head bias sized by trusted cfg.vocab_size
                 if i >= 1 and i <= GPT.head_bias.size then
                      GPT.head_bias:set(i, value)
                      layer_vals_count = layer_vals_count + 1
                 else
                    -- Might indicate mismatch if i > cfg.vocab_size, potential noise
                    -- print(string.format("Debug: Skipping head bias at index %d - potentially outside trained vocab size %d", i, GPT.head_bias.size))
                end
            end
        elseif layer_num >= 1 and layer_num <= cfg.num_layers then -- Transformer Blocks
            local block = GPT.blocks[layer_num]
            if not block then
                print(string.format("Warning: Found data for layer %d, but block structure not initialized. Skipping.", layer_num))
                goto continue_layer_loop -- Skip to next row
            end

            local target_component = nil
            local is_bias = false
             if block.attn[component_name] then
                 target_component = block.attn[component_name]
                 is_bias = component_name:find("_bias$") ~= nil
             elseif block.mlp[component_name] then
                  target_component = block.mlp[component_name]
                  is_bias = component_name:find("_bias$") ~= nil
             end

            if target_component then
                if is_bias then
                     -- Saved bias with j=1, use i for index
                     if i >= 1 and i <= target_component.size then
                         target_component:set(i, value)
                         layer_vals_count = layer_vals_count + 1
                     else print(string.format("Warning: Skipping out-of-bounds bias for layer %d, comp %s at index %d", layer_num, component_name, i)) end
                else -- It's a weight tensor
                     if i >= 1 and i <= target_component.rows and j >= 1 and j <= target_component.cols then
                         target_component:set(i, j, value)
                         layer_vals_count = layer_vals_count + 1
                     else print(string.format("Warning: Skipping out-of-bounds weight for layer %d, comp %s at (%d, %d)", layer_num, component_name, i, j)) end
                end
            else
                print(string.format("Warning: Component '%s' not found in structure for layer %d. Skipping.", component_name, layer_num))
            end
        else
             print(string.format("Warning: Found data for unexpected layer number %d. Skipping.", layer_num))
        end
        ::continue_layer_loop::
    end
    stmt_layers:finalize()
    print(string.format("Loaded %d layer weight/bias values.", layer_vals_count))

    -- 6. Load Relative Position Bias (if enabled in loaded config)
    if cfg.relative_pos_bias then
         print("Loading relative position biases...")
         if not GPT.relative_pos_bias then
             error("Config enables relative_pos_bias, but structure not initialized.")
         end
         local relpos_count = 0
         local stmt_relpos = db:prepare("SELECT bucket, head, value FROM relative_pos_bias")
          if not stmt_relpos then error("Failed to prepare relative_pos_bias statement: " .. db:errmsg()) end
         while stmt_relpos:step() == sqlite3.ROW do
             local bucket = stmt_relpos:get_value(0) -- INTEGER
             local head_idx = stmt_relpos:get_value(1) -- INTEGER
             local value = stmt_relpos:get_value(2) -- REAL

              if bucket >= 1 and bucket <= GPT.relative_pos_bias.num_buckets and head_idx >= 1 and head_idx <= GPT.relative_pos_bias.num_heads then
                   GPT.relative_pos_bias:set(bucket, head_idx, value)
                   relpos_count = relpos_count + 1
              else print(string.format("Warning: Skipping out-of-bounds rel pos bias at bucket %d, head %d", bucket, head_idx)) end
         end
         stmt_relpos:finalize()
         print(string.format("Loaded %d relative position bias values.", relpos_count))
         local expected_relpos = cfg.num_relative_pos_buckets * cfg.num_heads
         if relpos_count ~= expected_relpos then
             print(string.format("Warning: Loaded %d rel pos bias values, expected %d.", relpos_count, expected_relpos))
         end
    else
         print("Relative position bias not enabled in loaded config.")
    end

    print("Model loading complete.")
end

--------------------------------------------------
-- Main Query Loop
--------------------------------------------------
local function run_query_interface()
    -- Load the model first
    local load_status, load_error = pcall(load_model_from_db)
    if not load_status then
        print("Error loading model: " .. tostring(load_error))
        if db then db:close() end
        return -- Cannot continue without model
    end

    print("\n=== Lua GPT Query Interface ===")
    print("Model loaded successfully from:", cfg.model_db)
    print("Vocab size:", cfg.vocab_size)
    print("Type 'quit' or 'exit' to end.")
    print("Commands: ")
    print("  set temp <value>  (e.g., set temp 0.7)")
    print("  set max <value>   (e.g., set max 150)")
    print("--------------------------------")

    local current_temp = cfg.default_temperature
    local current_max_new = cfg.default_max_new_tokens

    math.randomseed(os.time()) -- Seed random generator for sampling

    while true do
        io.write(string.format("\nPrompt (temp=%.2f, max_new=%d)> ", current_temp, current_max_new))
        local input = io.read()

        if not input then break end -- EOF
        input = input:match("^%s*(.-)%s*$") -- Trim whitespace

        if input == "quit" or input == "exit" then
            break
        elseif input:match("^set%s+temp%s+(%d*%.?%d+)$") then
            local val = tonumber(input:match("^set%s+temp%s+(%d*%.?%d+)$"))
            if val and val > 0 then
                current_temp = val
                print("Temperature set to:", current_temp)
            else
                print("Invalid temperature value. Must be positive number.")
            end
        elseif input:match("^set%s+max%s+(%d+)$") then
             local val = tonumber(input:match("^set%s+max%s+(%d+)$"))
             if val and val > 0 then
                 current_max_new = val
                 print("Max new tokens set to:", current_max_new)
             else
                 print("Invalid max new tokens value. Must be positive integer.")
             end
        elseif #input > 0 then
            -- Generate text
            local gen_status, result = pcall(generate, input, current_max_new, current_temp)

            if gen_status then
                print("\n--- Generated Text ---")
                print(result)
                print("----------------------")
            else
                print("\nError during generation: " .. tostring(result))
                -- Optionally break or continue
            end
        else
            -- Empty input, just loop back
        end
    end

    print("\nExiting...")
    if db then
        db:close()
        print("Database connection closed.")
    end
end

-- Seed math.random for dropout (though disabled) and sampling
math.randomseed(os.time())
math.randomGaussian = function() -- Keep for compatibility if any part expects it
    local u1 = math.random()
    local u2 = math.random()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
end


-- Run the interface
run_query_interface()
