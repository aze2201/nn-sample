local sqlite3 = require('lsqlite3')
local ffi = require('ffi')
local math = require('math')

local DEBUG = true      -- Set to false to disable verbose output
local TOP_K = 5         -- Show top 5 candidates at each step
local VERBOSE = true    -- Enable detailed logging

local function log(...)
    if VERBOSE then
        print("[DEBUG]", ...)
    end
end

-- Load configuration and model from training script
local cfg = {
    vocab_size = 426,
    embed_dim = 128,
    num_heads = 8,
    num_layers = 6,
    seq_len = 256,
    model_db = 'gpt_model.db'
}

ffi.cdef[[
    void* malloc(size_t size);
    void free(void *ptr);
]]

local db = sqlite3.open(cfg.model_db)
local GPT = {}
local vocab = {}
local idx_to_word = {}

-- Load vocabulary from database
db:exec("SELECT word, id FROM vocab", function(row)
    vocab[row.word] = row.id
    idx_to_word[row.id] = row.word
end)

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

-- Transformer block (same as training)
local function transformer_block()
    return {
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
end

-- Load model weights from database (with error checking)
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
        vocab[row.word] = row.id
        idx_to_word[row.id] = row.word
        print(string.format("[DEBUG] Loaded word: %-15s → ID %d", row.word, row.id))
    end
    stmt:finalize()

    -- Verify special tokens
    assert(vocab["<unk>"] == cfg.vocab_size + 1, "Missing <unk> token in vocabulary")
    assert(vocab["<pad>"] == 0, "Missing <pad> token in vocabulary")

    print("[DEBUG] Loading embeddings...")
    stmt = db:prepare("SELECT type, position, dim, value FROM embeddings")
    for row in stmt:nrows() do
        local tensor = row.type == "wte" and GPT.wte or GPT.wpe
        if not tensor then
            error("Unknown embedding type: " .. row.type)
        end
        
        local pos = row.position + 1  -- SQL is 0-indexed
        local dim = row.dim + 1
        
        if pos > tensor.rows or dim > tensor.cols then
            error(string.format("Embedding position out of bounds: type=%s pos=%d dim=%d", row.type, pos, dim))
        end
        
        tensor:set(pos, dim, row.value)
    end
    stmt:finalize()

    print("[DEBUG] Loading transformer blocks...")
    for layer = 0, cfg.num_layers - 1 do
        print(string.format("[DEBUG] Loading layer %d...", layer))
        local block = transformer_block()
        
        stmt = db:prepare([[SELECT component, i, j, value FROM layers WHERE layer=?]])
        stmt:bind_values(layer)

        for row in stmt:nrows() do
            local component = row.component
            local t
            if component == "q" or component == "k" or component == "v" or component == "proj" then
                t = block.attn[component]
            elseif component == "fc1" or component == "fc2" then
                t = block.mlp[component]
            else
                error(string.format("Invalid component '%s' in layer %d", component, layer))
            end

            local i = row.i + 1
            local j = row.j + 1
            
            if i > t.rows or j > t.cols then
                error(string.format("Index out of bounds for %s: (%d,%d) in (%d,%d tensor)", component, i, j, t.rows, t.cols))
            end
            
            t:set(i, j, row.value)
            print(string.format("[DEBUG]   Set %-4s [%3d,%3d] = %.4f", component, i, j, row.value))
        end
        stmt:finalize()
        table.insert(GPT.blocks, block)
    end

    print("[DEBUG] Loading head weights...")
    stmt = db:prepare("SELECT i, j, value FROM head")
    for row in stmt:nrows() do
        local i = row.i + 1
        local j = row.j + 1
        
        if i > GPT.head.rows or j > GPT.head.cols then
            error(string.format("Head index out of bounds: (%d,%d) in (%d,%d tensor)", i, j, GPT.head.rows, GPT.head.cols))
        end
        
        GPT.head:set(i, j, row.value)
        print(string.format("[DEBUG] Set head [%3d,%3d] = %.4f", i, j, row.value))
    end
    stmt:finalize()

    print("[DEBUG] Model loading completed successfully!")
end

local function detokenize(tokens)
    local words = {}
    for _, id in ipairs(tokens) do
        local word = idx_to_word[id]
        if not word then
            word = "<unk>"
            if DEBUG then
                print(string.format("Unknown token ID: %d", id))
            end
        end
        table.insert(words, word)
    end
    return table.concat(words, " ")
end

-- Text tokenization
local function tokenize(text)
    local tokens = {}
    text = text:gsub("[%c%p]", ""):lower()
    text = text:gsub("%s+", " "):gsub("^%s*", ""):gsub("%s*$", "")
    
    if DEBUG then
        print("[DEBUG] Normalized text:", '"' .. text .. '"')
    end

    for word in text:gmatch("%S+") do
        local original_word = word
        local id = vocab[word] or cfg.vocab_size + 1  -- fallback to <unk>
        if id < 0 or id > cfg.vocab_size + 1 then
            if DEBUG then
                print(string.format("[WARN] Clamping invalid ID %d for word '%s'", id, original_word))
            end
            id = cfg.vocab_size + 1
        end
        table.insert(tokens, id)
        if DEBUG then
            local token_str = idx_to_word[id] or "<unk>"
            print(string.format("[DEBUG] Tokenized: %-15s → %-4d (%s)", "'" .. original_word .. "'", id, token_str))
        end
    end

    if #tokens == 0 then
        if DEBUG then
            print("[WARN] Empty input after tokenization, using <pad>")
        end
        table.insert(tokens, 0)  -- <pad>
    end

    return tokens
end

-- Text generation function
local function forward(inputs)
    local batch_size = #inputs
    local seq_len = #inputs[1]

    local emb = {}
    for b = 1, batch_size do
        emb[b] = {}
        for t = 1, seq_len do
            local token_id = inputs[b][t]
            if token_id < 1 or token_id > cfg.vocab_size + 1 then
                token_id = cfg.vocab_size + 1  -- Clamp to <unk>
            end
            emb[b][t] = {
                data = ffi.new("double[?]", cfg.embed_dim),
                grad = ffi.new("double[?]", cfg.embed_dim)
            }
            for d = 1, cfg.embed_dim do
                emb[b][t].data[d-1] = GPT.wte:get(token_id, d) + GPT.wpe:get(t, d)
            end
        end
    end

    local activations = {emb}
    for layer = 1, cfg.num_layers do
        local block = GPT.blocks[layer]
        local new_activations = {}
        for b = 1, batch_size do
            new_activations[b] = {}
            local q = ffi.new("double[?]", seq_len * cfg.embed_dim)
            local k = ffi.new("double[?]", seq_len * cfg.embed_dim)
            local v = ffi.new("double[?]", seq_len * cfg.embed_dim)
            for t = 1, seq_len do
                for d = 1, cfg.embed_dim do
                    local idx = (t-1)*cfg.embed_dim + (d-1)
                    q[idx] = 0
                    k[idx] = 0
                    v[idx] = 0
                    for h = 1, cfg.embed_dim do
                        q[idx] = q[idx] + activations[layer][b][t].data[h-1] * block.attn.q:get(h, d)
                        k[idx] = k[idx] + activations[layer][b][t].data[h-1] * block.attn.k:get(h, d)
                        v[idx] = v[idx] + activations[layer][b][t].data[h-1] * block.attn.v:get(h, d)
                    end
                end
            end

            local scores = ffi.new("double[?]", seq_len * seq_len)
            for i = 1, seq_len do
                for j = 1, seq_len do
                    local score = 0.0
                    if j <= i then
                        for d = 1, cfg.embed_dim do
                            score = score + q[(i-1)*cfg.embed_dim + d-1] * k[(j-1)*cfg.embed_dim + d-1]
                        end
                    else
                        score = -math.huge
                    end
                    scores[(i-1)*seq_len + (j-1)] = score / math.sqrt(cfg.embed_dim)
                end
            end

            local probs = ffi.new("double[?]", seq_len * seq_len)
            for i = 1, seq_len do
                local max_score = -math.huge
                for j = 1, seq_len do
                    max_score = math.max(max_score, scores[(i-1)*seq_len + (j-1)])
                end
                local sum_exp = 0.0
                for j = 1, seq_len do
                    probs[(i-1)*seq_len + (j-1)] = math.exp(scores[(i-1)*seq_len + (j-1)] - max_score)
                    sum_exp = sum_exp + probs[(i-1)*seq_len + (j-1)]
                end
                for j = 1, seq_len do
                    probs[(i-1)*seq_len + (j-1)] = probs[(i-1)*seq_len + (j-1)] / sum_exp
                end
            end

            for t = 1, seq_len do
                new_activations[b][t] = {
                    data = ffi.new("double[?]", cfg.embed_dim),
                    grad = ffi.new("double[?]", cfg.embed_dim)
                }
                for d = 1, cfg.embed_dim do
                    local sum = 0.0
                    for j = 1, seq_len do
                        sum = sum + probs[(t-1)*seq_len + (j-1)] * v[(j-1)*cfg.embed_dim + d-1]
                    end
                    new_activations[b][t].data[d-1] = sum
                end
            end
        end
        activations[layer+1] = new_activations
    end

    local logits = {}
    for b = 1, batch_size do
        logits[b] = {}
        for t = 1, seq_len do
            logits[b][t] = ffi.new("double[?]", cfg.vocab_size + 2)
            for v = 0, cfg.vocab_size + 1 do
                logits[b][t][v] = 0
                for d = 1, cfg.embed_dim do
                    logits[b][t][v] = logits[b][t][v] + activations[#activations][b][t].data[d-1] * GPT.head:get(d, v+1)
                end
            end
        end
    end

    return logits
end

local function generate(input_ids, max_length, temperature)
    temperature = temperature or 0.7
    local generated = {unpack(input_ids)}
    local repetition_penalty = 5
    local last_tokens = {}
    local max_recent = 4

    for i = 1, max_length do
        local start_idx = math.max(1, #generated - cfg.seq_len + 1)
        local context = {unpack(generated, start_idx, #generated)}
        local inputs = {context}
        local logits = forward(inputs)[1][#context]

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

-- Main execution
load_model()


while true do

   print("Enter your question (press Ctrl+D when done):")
   --local question = "Who is Fariz"
   io.write("> ")
   io.flush()
   local question = io.read()
   log(string.format("Raw input received: %q", question))

   if #question == 0 then
       error("No input provided! Did you press Ctrl+D correctly?")
   end

   local input_tokens = tokenize(question)
   log(string.format("Tokenized input (%d tokens):", #input_tokens))
   log(table.concat(input_tokens, ", "))

   -- Only capture the generated tokens
   local output_tokens = generate(input_tokens, 100, 0.7)
   local response = detokenize(output_tokens)

   print("\n=== Final Response ===")
   print(response)
end

db:close()
