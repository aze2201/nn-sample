-- query.lua
local sqlite3 = require('lsqlite3')
local math = require('math')
local ffi = require('ffi')

ffi.cdef[[
    void free(void *ptr);
    void* malloc(size_t size);
    double* copy_array_to_double_ptr(double* src, int size);
]]

-- Helper function (in C, for speed) to copy Lua table to double*
ffi.cdef [[
    double* copy_array_to_double_ptr(double* src, int size);
]]
local c_lib = ffi.load(ffi.os == 'Windows' and 'msvcrt' or 'c')

-- Copy a Lua table (of numbers) to a new FFI double array.
local function copy_to_double_array(lua_table)
    local size = #lua_table
    local ffi_array = ffi.new("double[?]", size)
    for i = 1, size do
        ffi_array[i - 1] = lua_table[i] or 0  -- Handle potential nil
    end
    return ffi_array
end

-- Configuration (must match the training configuration)
local cfg = {
    vocab_size = 4189,
    embed_dim = 128,
    num_heads = 8,
    num_layers = 6,
    seq_len = 128,
    dropout = 0.2,
    model_db = 'gpt_model.db',
    temperature = 0.7,
    top_k = 50,
    num_generate = 50,
    attention_threshold = 0.5, -- Threshold for cosine similarity
}

-- Load the model
local function load_model()
    local db, err = sqlite3.open(cfg.model_db)
    if not db then
        error("Failed to open database: " .. (err or "unknown error"))
    end
    local function exec_sql(sql)
        local rc = db:exec(sql)
        if rc ~= sqlite3.OK then
            error("SQL execution failed ("..rc.."): "..db:errmsg().."
SQL: "..sql)
        end
    end
    exec_sql("PRAGMA journal_mode = WAL;")
    exec_sql("PRAGMA synchronous = NORMAL;")
    local GPT = {}
    local function load_tensor(layer, component, rows, cols)
        local size = rows * cols
        local data = ffi.new("double[?]", size)
        ffi.fill(data, ffi.sizeof("double") * size, 0)
        local stmt
        if layer then
            stmt = db:prepare("SELECT i, j, value FROM layers WHERE layer = ? AND component = ? ORDER BY i, j")
            stmt:bind_values(layer, component)
        else
            stmt = db:prepare("SELECT position, dim, value FROM embeddings WHERE type = ? ORDER BY position, dim")
            stmt:bind_values(component)
        end
        while stmt:step() == sqlite3.ROW do
            local i, j, value
            if layer then
                i = stmt:get_value(0) + 1
                j = stmt:get_value(1) + 1
            else
                i = stmt:get_value(0) + 1
                j = stmt:get_value(1) + 1
            end
            value = stmt:get_value(2)
            if value == nil then
                error(string.format("Nil value found for layer=%s, component=%s, i=%d, j=%d", tostring(layer), component, i - 1, j - 1))
            end
            data[(i - 1) * cols + (j - 1)] = value
        end
        stmt:finalize()
        return {
            data = data,
            rows = rows,
            cols = cols,
            get = function(self, i, j) return self.data[(i - 1) * self.cols + (j - 1)] end
        }
    end
    local function load_head_tensor(rows, cols)
        local size = rows * cols
        local data = ffi.new("double[?]", size)
        ffi.fill(data, ffi.sizeof("double") * size, 0)
        local stmt = db:prepare("SELECT i, j, value FROM head ORDER BY i, j")
        while stmt:step() == sqlite3.ROW do
            local i = stmt:get_value(0) + 1
            local j = stmt:get_value(1) + 1
            local value = stmt:get_value(2)
            if value == nil then
                error(string.format("Nil value found in head, i=%d, j=%d", i - 1, j - 1))
            end
            data[(i - 1) * cols + (j - 1)] = value
        end
        stmt:finalize()
        return {
            data = data,
            rows = rows,
            cols = cols,
            get = function(self, i, j) return self.data[(i - 1) * self.cols + (j - 1)] end
        }
    end
    GPT.wte = load_tensor(nil, 'wte', cfg.vocab_size + 2, cfg.embed_dim)
    GPT.wpe = load_tensor(nil, 'wpe', cfg.seq_len, cfg.embed_dim)
    GPT.blocks = {}
    for layer = 0, cfg.num_layers - 1 do
        local block = {
            attn = {
                q = load_tensor(layer, 'q', cfg.embed_dim, cfg.embed_dim),
                k = load_tensor(layer, 'k', cfg.embed_dim, cfg.embed_dim),
                v = load_tensor(layer, 'v', cfg.embed_dim, cfg.embed_dim),
                proj = load_tensor(layer, 'proj', cfg.embed_dim, cfg.embed_dim),
            },
            mlp = {
                fc1 = load_tensor(layer, 'fc1', cfg.embed_dim, 4 * cfg.embed_dim),
                fc2 = load_tensor(layer, 'fc2', 4 * cfg.embed_dim, cfg.embed_dim),
            }
        }
        table.insert(GPT.blocks, block)
    end
    GPT.head = load_head_tensor(cfg.embed_dim, cfg.vocab_size + 2)
    local vocab = {}
    local idx_to_word = {}
    local stmt = db:prepare("SELECT word, id FROM vocab")
    while stmt:step() == sqlite3.ROW do
        local word = stmt:get_value(0)
        local id = stmt:get_value(1)
        vocab[word] = id
        idx_to_word[id] = word
    end
    stmt:finalize()
    db:close()
    return GPT, vocab, idx_to_word
end

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
    return norm
end

local function dropout_forward_inference(vec, size, dropout_rate)
    local out = ffi.new("double[?]", size)
    local scale = 1.0 / (1.0 - dropout_rate)
    for i = 0, size - 1 do
        out[i] = vec[i] * scale
    end
    return out
end

local function linear_forward(input, tensor)
    local in_features = tensor.rows
    local out_features = tensor.cols
    local output = ffi.new("double[?]", out_features)
    for j = 1, out_features do
        local sum = 0
        for i = 1, in_features do
            sum = sum + input[i - 1] * tensor:get(i, j)
        end
        output[j - 1] = sum
    end
    return output
end

local function relu_forward(input, size)
    local out = ffi.new("double[?]", size)
    for i = 0, size - 1 do
        out[i] = math.max(0, input[i])
    end
    return out
end

-- Modified transformer_block_forward to return attention weights
local function transformer_block_forward(block, norm_tokens)
    local head_dim = cfg.embed_dim / cfg.num_heads
    local seq_len = #norm_tokens
    local attn_outputs = {}
    for t = 1, seq_len do
        attn_outputs[t] = ffi.new("double[?]", cfg.embed_dim)
        for d = 0, cfg.embed_dim-1 do attn_outputs[t][d] = 0 end
    end
     -- This will store the attention weights.  attn_weights_all[h][i][j] is
    -- the attention weight from token i to token j in head h.
    local attn_weights_all = {}
    for h = 1, cfg.num_heads do
        attn_weights_all[h] = {}  -- Initialize the table for this head
        for i = 1, seq_len do
            attn_weights_all[h][i] = {} --Initialize
           -- Compute query for token i
            local q = ffi.new("double[?]", head_dim)
            for d = 1, head_dim do
                q[d-1] = 0
                for j = 1, cfg.embed_dim do
                    q[d-1] = q[d-1] + norm_tokens[i][j-1] * block.attn.q:get(j, (h-1)*head_dim + d)
                end
            end
            -- Compute keys and values
            local keys = {}
            local values = {}
            for j = 1, seq_len do
                local k = ffi.new("double[?]", head_dim)
                local v = ffi.new("double[?]", head_dim)
                for d = 1, head_dim do
                  k[d-1] = 0
                  v[d-1] = 0
                  for r = 1, cfg.embed_dim do
                    k[d-1] = k[d-1] + norm_tokens[j][r-1] * block.attn.k:get(r, (h-1)*head_dim + d)
                    v[d-1] = v[d-1] + norm_tokens[j][r-1] * block.attn.v:get(r, (h-1)*head_dim + d)
                  end
                end
                keys[j] = k
                values[j] = v
            end
            -- Compute attention scores (with causal masking)
            local scores = ffi.new("double[?]", seq_len)
            local max_score = -math.huge
            for j = 1, seq_len do
              local score = 0
                if j <= i then
                  for d = 0, head_dim-1 do
                    score = score + q[d] * keys[j][d]
                  end
                  score = score / math.sqrt(head_dim)
                else
                  score = -math.huge  -- Apply causal mask
                end
                scores[j-1] = score
              if score > max_score then max_score = score end
            end
            local exps = ffi.new("double[?]", seq_len)
            local sum_exp = 0
            for j = 1, seq_len do
              if scores[j-1] == -math.huge then
                exps[j-1] = 0  --  Handle -inf from masking
              else
                exps[j-1] = math.exp(scores[j-1] - max_score)
              end
              sum_exp = sum_exp + exps[j-1]
            end
            local attn_weights = ffi.new("double[?]", seq_len)
            for j = 1, seq_len do
              attn_weights[j-1] = exps[j-1] / sum_exp
              attn_weights_all[h][i][j] = attn_weights[j-1]
            end
            local head_output = ffi.new("double[?]", head_dim)
            for d = 0, head_dim-1 do head_output[d] = 0 end
            for j = 1, seq_len do
              local weight = attn_weights[j-1]
              for d = 0, head_dim-1 do
                head_output[d] = head_output[d] + weight * values[j][d]
              end
            end
            for d = 0, head_dim-1 do
              attn_outputs[i][(h-1)*head_dim + d] = attn_outputs[i][(h-1)*head_dim + d] + head_output[d]
            end
        end
    end
    -- Project attention outputs
    local proj_outputs = {}
    for t = 1, seq_len do
      proj_outputs[t] = ffi.new("double[?]", cfg.embed_dim)
      for d = 1, cfg.embed_dim do
        local sum = 0
        for i = 1, cfg.embed_dim do
          sum = sum + attn_outputs[t][i-1] * block.attn.proj:get(i, d)
        end
        proj_outputs[t][d-1] = sum
      end
      proj_outputs[t] = dropout_forward_inference(proj_outputs[t], cfg.embed_dim, cfg.dropout)
    end
    -- Residual connection 1
    local res1 = {}
    for t = 1, seq_len do
        res1[t] = ffi.new("double[?]", cfg.embed_dim)
        for d = 0, cfg.embed_dim-1 do
            res1[t][d] = norm_tokens[t][d] + proj_outputs[t][d]
        end
    end
    -- MLP
    local mlp_outputs = {}
    for t = 1, seq_len do
      local norm_res1 = layer_norm_forward(res1[t], cfg.embed_dim)
      local fc1_out = ffi.new("double[?]", 4 * cfg.embed_dim)
      for j = 1, 4 * cfg.embed_dim do
        local sum = 0
        for i = 1, cfg.embed_dim do
            sum = sum + norm_res1[i-1] * block.mlp.fc1:get(i, j)
        end
        fc1_out[j-1] = sum
      end
      local relu_out = relu_forward(fc1_out, 4 * cfg.embed_dim)
      local fc2_out = ffi.new("double[?]", cfg.embed_dim)
      for d = 1, cfg.embed_dim do
        local sum = 0
        for j = 1, 4 * cfg.embed_dim do
          sum = sum + relu_out[j-1] * block.mlp.fc2:get(j, d)
        end
        fc2_out[d-1] = sum
      end
       mlp_outputs[t] = dropout_forward_inference(fc2_out, cfg.embed_dim, cfg.dropout)
    end
    -- Residual connection 2 and output
    local out_tokens = {}
    for t = 1, seq_len do
        out_tokens[t] = { data = ffi.new("double[?]", cfg.embed_dim) }
        for d = 0, cfg.embed_dim-1 do
            out_tokens[t].data[d] = res1[t][d] + mlp_outputs[t][d]
        end
    end
    return out_tokens, attn_weights_all  -- Return both outputs and attention
end

-- Modified forward function to return attention weights
local function forward(GPT, input_tokens)
    local seq_len = #input_tokens
    local activations = {}
    activations[1] = {}
    -- Store all attention weights from all layers
    local all_layer_attn_weights = {}
    for t = 1, seq_len do
local emb = ffi.new("double[?]", cfg.embed_dim)
        for d = 1, cfg.embed_dim do
            emb[d - 1] = GPT.wte:get(input_tokens[t], d) + GPT.wpe:get(t, d)
        end
        activations[1][t] = { data = emb }
    end
    for layer = 1, cfg.num_layers do
        activations[layer + 1] = {}
        local norm_tokens = {}
        for t = 1, seq_len do
            norm_tokens[t] = layer_norm_forward(activations[layer][t].data, cfg.embed_dim)
        end
        local block_out, attn_weights = transformer_block_forward(GPT.blocks[layer], norm_tokens)
        activations[layer + 1] = block_out
        all_layer_attn_weights[layer] = attn_weights -- Store attention weights
    end
    local logits = {}
    local final_layer = #activations
    for t = 1, seq_len do
        local token_act = activations[final_layer][t].data
        local logit = ffi.new("double[?]", cfg.vocab_size + 2)
        for v = 0, cfg.vocab_size + 1 do
            local sum = 0
            for d = 1, cfg.embed_dim do
                sum = sum + token_act[d - 1] * GPT.head:get(d, v + 1)
            end
            logit[v] = sum
        end
        logits[t] = logit
    end
    return logits, all_layer_attn_weights  -- Return both logits and attention
end

-- Sample from logits (no change)
local function sample_next_token(logits, temperature, top_k)
    local vocab_size = cfg.vocab_size + 1
    -- Apply temperature
    for i = 0, vocab_size do
        logits[i] = logits[i] / temperature
    end
    -- Find max logit
    local max_logit = -math.huge
    for i = 0, vocab_size do
        if logits[i] > max_logit then
            max_logit = logits[i]
        end
    end
    -- Softmax
    local exps = ffi.new("double[?]", vocab_size + 1)
    local sum_exp = 0
    for i = 0, vocab_size do
        exps[i] = math.exp(logits[i] - max_logit)
        sum_exp = sum_exp + exps[i]
    end
    local probs = ffi.new("double[?]", vocab_size + 1)
    for i = 0, vocab_size do
        probs[i] = exps[i] / sum_exp
    end
    -- Top-k
    local candidates = {}
    for i = 0, vocab_size do
        table.insert(candidates, {prob = probs[i], index = i})
    end
    table.sort(candidates, function(a, b) return a.prob > b.prob end)
    local top_k_probs = ffi.new("double[?]", top_k)
    local top_k_indices = ffi.new("int[?]", top_k)
    local top_k_sum = 0
    for i = 1, math.min(top_k, #candidates) do
        top_k_probs[i-1] = candidates[i].prob
        top_k_indices[i-1] = candidates[i].index
        top_k_sum = top_k_sum + top_k_probs[i-1]
    end
    if top_k_sum > 0 then
        for i = 0, math.min(top_k, #candidates) - 1 do
            top_k_probs[i] = top_k_probs[i] / top_k_sum
        end
    else
        top_k_indices[0] = 0
    end
    local rand_val = math.random()
    local cumulative_prob = 0
    local sampled_index = top_k_indices[0]
    for i = 0, math.min(top_k, #candidates) - 1 do
        cumulative_prob = cumulative_prob + top_k_probs[i]
        if rand_val <= cumulative_prob then
            sampled_index = top_k_indices[i]
            break
        end
    end
    return sampled_index
end

-- Cosine similarity between two FFI arrays
local function cosine_similarity(vec1, vec2, size)
    local dot_product = 0.0
    local norm1 = 0.0
    local norm2 = 0.0
    for i = 0, size - 1 do
        dot_product = dot_product + vec1[i] * vec2[i]
        norm1 = norm1 + vec1[i] * vec1[i]
        norm2 = norm2 + vec2[i] * vec2[i]
    end
    if norm1 == 0 or norm2 == 0 then
        return 0  -- Handle cases where one or both vectors are zero.
    end
    return dot_product / (math.sqrt(norm1) * math.sqrt(norm2))
end

-- Modified generation function
local function generate(GPT, vocab, idx_to_word, prompt)
    local input_tokens = {}
    local prompt_words = {} -- Store original prompt words, preserving case
    -- Tokenize the prompt and store original words
    for word in prompt:gmatch("%S+") do
        local token = vocab[word:lower()] or vocab["<unk>"]
        table.insert(input_tokens, token)
        table.insert(prompt_words, word)  -- Store the *original* word
    end
    -- Pad or truncate input tokens
    if #input_tokens > cfg.seq_len then
        input_tokens = {table.unpack(input_tokens, #input_tokens - cfg.seq_len + 1, #input_tokens)}
    elseif #input_tokens < cfg.seq_len then
        local padding_length = cfg.seq_len - #input_tokens
        for _ = 1, padding_length do
            table.insert(input_tokens, 1, vocab["<pad>"])
        end
    end
    print("Generating text:")
    -- Get attention weights for the *initial* prompt
    local _, initial_attn_weights = forward(GPT, input_tokens)
    for i = 1, cfg.num_generate do
        local logits_table, current_attn_weights = forward(GPT, input_tokens)
        local next_token_id = sample_next_token(logits_table[#input_tokens], cfg.temperature, cfg.top_k)
        local next_word = idx_to_word[next_token_id]
        -- Handle nil words (replace with <unk>)
        if next_word == nil then
            print(string.format("  Step %d: Predicted token ID: %d, Word: 'nil' (Replacing with <unk>)", i, next_token_id))
            next_word = "<unk>"
            next_token_id = vocab["<unk>"]
        else
            print(string.format("  Step %d: Predicted token ID: %d, Word: '%s'", i, next_token_id, next_word))
        end
        -- *Before* attention comparison, update the input_tokens
        table.insert(input_tokens, next_token_id)
        table.remove(input_tokens, 1)
        -- Compare attention weights (only after the first generated token)
        if i > 1 then
            local layer_to_compare = cfg.num_layers
            local prompt_len = #prompt_words  -- Length of the *original* prompt
            -- Pre-allocate attention tables outside the head loop
            local prompt_attention = {}
            local generated_attention = {}
            for h = 1, cfg.num_heads do
                prompt_attention[h] = {}
                generated_attention[h] = {}
            end
            -- Populate the attention tables *correctly*
            for h = 1, cfg.num_heads do
                -- Prompt attention:  initial_attn_weights, last token of prompt, all prompt tokens
                for k = 1, prompt_len do
                    table.insert(prompt_attention[h], initial_attn_weights[layer_to_compare][h][prompt_len][k] or 0)
                end
                -- Generated attention: current_attn_weights, *last generated token*, all prompt tokens
                for k = 1, prompt_len do
                    -- prompt_len + i - 1  is incorrect.  It should be cfg.seq_len
                    table.insert(generated_attention[h], current_attn_weights[layer_to_compare][h][cfg.seq_len][k] or 0)
                end
            end
            -- Use pcall for the *entire* similarity calculation and memory management
            local success, avg_similarity = pcall(function()
                local total_similarity = 0
                for h = 1, cfg.num_heads do
                    local prompt_attn_ffi = nil
                    local generated_attn_ffi = nil
                    -- Allocate only if needed
                    if #prompt_attention[h] > 0 then
                        prompt_attn_ffi = copy_to_double_array(prompt_attention[h])
                    end
                    if #generated_attention[h] > 0 then
                        generated_attn_ffi = copy_to_double_array(generated_attention[h])
                    end
                    local sim = 0
                    if prompt_attn_ffi and generated_attn_ffi then
                        -- Cosine similarity calculation (correct length)
                        sim = cosine_similarity(prompt_attn_ffi, generated_attn_ffi, prompt_len)
                    end
                    total_similarity = total_similarity + sim
                    -- No need to free memory allocated with ffi.new
                end
                return total_similarity / cfg.num_heads  -- Return the *average* similarity
            end)
            if success then
                print(string.format("    Average Attention Similarity (Layer %d): %.4f", layer_to_compare, avg_similarity))
            else
                print("Error during attention comparison: " .. avg_similarity) -- avg_similarity now holds error
                avg_similarity = 0  -- Or handle as appropriate
            end
            if avg_similarity < cfg.attention_threshold then
                print("    Attention similarity below threshold. Consider re-prompting or adjusting parameters.")
            end
            print("    Prompt Self-Attention (Last Token, Head 1):")
            for k = 1, prompt_len do
                local attn_weight = initial_attn_weights[layer_to_compare][1][prompt_len][k] or 0
                print(string.format("      '%s': %.4f", prompt_words[k], attn_weight))
            end
            print("    Generated Self-Attention (Last Token, Head 1):")
            for k = 1, prompt_len do  -- Corrected loop: Only iterate through prompt words
                local word = prompt_words[k] or "<pad>"  -- Use prompt_words
                --  current_attn_weights, *cfg.seq_len*, and prompt words
                local attn_weight = current_attn_weights[layer_to_compare][1][cfg.seq_len][k] or 0
                print(string.format("      '%s': %.4f", word, attn_weight))
            end
        end
        -- Update initial attention weights for the next iteration
        initial_attn_weights = current_attn_weights
    end
    -- Decode generated tokens to text
    local generated_text = ""
     for i = 1, cfg.seq_len do  -- Corrected loop condition
      local word = idx_to_word[input_tokens[i]]
        if word and word ~= "<pad>" then
                generated_text = generated_text .. " " .. word
        end
    end
    return generated_text
end

-- Load model and vocab
local GPT, vocab, idx_to_word = load_model()

-- Interactive prompt
print("Enter your prompt (or 'quit' to exit):")
while true do
    io.write("> ")
    local prompt = io.read()
    if prompt:lower() == "quit" then
        break
    end
    local generated_text = generate(GPT, vocab, idx_to_word, prompt)
    print("Generated text:
" .. generated_text .. "
")
end
