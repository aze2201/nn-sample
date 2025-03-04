local sqlite3 = require('lsqlite3')
local math = require('math')
local ffi = require('ffi')
local os = require('os')

ffi.cdef[[
    void free(void *ptr);
    void* malloc(size_t size);
]]

-- Configuration
local cfg = {
    vocab_size = 426,
    embed_dim = 128,
    num_heads = 8,        -- must divide embed_dim evenly
    num_layers = 6,
    seq_len = 256,
    lr = 3e-4,
    batch_size = 32,
    max_iters = 1,     -- increased for actual training
    dropout = 0.2,
    model_db = 'gpt_model.db',
    beta1 = 0.9,
    beta2 = 0.999,
    eps = 1e-8
}

-- Initialize database with error handling
local function init_database()
    local db, err = sqlite3.open(cfg.model_db)
    if not db then
        error("Failed to open database: " .. (err or "unknown error"))
    end

    local function exec_sql(sql)
        local rc = db:exec(sql)
        if rc ~= sqlite3.OK then
            error("SQL execution failed ("..rc.."): "..db:errmsg().."\nSQL: "..sql)
        end
    end

    exec_sql("PRAGMA journal_mode = WAL;")
    exec_sql("PRAGMA synchronous = NORMAL;")

    local tables = {
        [[CREATE TABLE IF NOT EXISTS config(
            key TEXT PRIMARY KEY, 
            value REAL
        );]],
        [[CREATE TABLE IF NOT EXISTS vocab(
            word TEXT PRIMARY KEY, 
            id INTEGER NOT NULL
        );]],
        [[CREATE TABLE IF NOT EXISTS layers(
            layer INTEGER,
            component TEXT,
            i INTEGER,
            j INTEGER,
            value REAL,
            PRIMARY KEY (layer, component, i, j)
        );]],
        [[CREATE TABLE IF NOT EXISTS head(
            i INTEGER,
            j INTEGER,
            value REAL,
            PRIMARY KEY (i, j)
        );]],
        [[CREATE TABLE IF NOT EXISTS embeddings(
            type TEXT,
            position INTEGER,
            dim INTEGER,
            value REAL,
            PRIMARY KEY (type, position, dim)
        );]]
    }

    for _, sql in ipairs(tables) do
        exec_sql(sql)
    end

    return db
end

local db = init_database()
local GPT = {}
local vocab = {}
local idx_to_word = {}

-- Utility: Simple Layer Normalization (no learnable parameters)
local function layer_norm(vec, size, eps)
    print("layer normalization function ....")
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

-- Utility: Dropout (applied elementwise)
local function dropout(vec, size, dropout_rate)
    local out = ffi.new("double[?]", size)
    for i = 0, size - 1 do
        if math.random() < dropout_rate then
            out[i] = 0
        else
            out[i] = vec[i]
        end
    end
    return out
end

-- Vocabulary building remains the same
local function build_vocabulary(text)
    local word_counts = {}
    local valid_chars = "[^%w%s]"
    text = text:gsub(valid_chars, ""):lower()
    for word in text:gmatch("%S+") do
        word_counts[word] = (word_counts[word] or 0) + 1
    end

    local words = {}
    for word in pairs(word_counts) do table.insert(words, word) end
    table.sort(words, function(a,b) return word_counts[a] > word_counts[b] end)

    vocab = {}
    idx_to_word = {}
    local max_id = math.min(#words, cfg.vocab_size)
    for id = 1, max_id do
        local word = words[id]
        vocab[word] = id
        idx_to_word[id] = word
    end

    vocab["<unk>"] = cfg.vocab_size + 1
    idx_to_word[cfg.vocab_size + 1] = "<unk>"
    vocab["<pad>"] = 0
    idx_to_word[0] = "<pad>"

    db:exec("BEGIN IMMEDIATE TRANSACTION")
    local stmt = db:prepare("INSERT OR REPLACE INTO vocab (word, id) VALUES (?, ?)")
    for word, id in pairs(vocab) do
        stmt:bind_values(word, id)
        if stmt:step() ~= sqlite3.DONE then
            db:exec("ROLLBACK")
            error("Failed to insert word: "..word)
        end
        stmt:reset()
    end
    db:exec("COMMIT")
    stmt:finalize()
end

-- Tensor creation helper remains unchanged
local function create_tensor(rows, cols)
    local size = rows * cols
    local data = ffi.new("double[?]", size)
    local grad = ffi.new("double[?]", size)
    local m = ffi.new("double[?]", size)  -- First moment
    local v = ffi.new("double[?]", size)  -- Second moment
    ffi.fill(data, ffi.sizeof("double") * size, 0)
    ffi.fill(grad, ffi.sizeof("double") * size, 0)
    ffi.fill(m, ffi.sizeof("double") * size, 0)
    ffi.fill(v, ffi.sizeof("double") * size, 0)
    return {
        data = data,
        grad = grad,
        m = m,
        v = v,
        rows = rows,
        cols = cols,
        get = function(self, i, j)
            return self.data[(i-1)*self.cols + (j-1)]
        end,
        set = function(self, i, j, val)
            self.data[(i-1)*self.cols + (j-1)] = val
        end,
        add_grad = function(self, i, j, val)
            self.grad[(i-1)*self.cols + (j-1)] = self.grad[(i-1)*self.cols + (j-1)] + val
        end,
        zero_grad = function(self)
            ffi.fill(self.grad, ffi.sizeof("double") * self.rows * self.cols, 0)
        end
    }
end

-- Transformer block: now includes multi-head attention, residual connections,
-- layer norm, feed-forward (MLP), and dropout.
local function transformer_block()
    local attn = {
        -- Full projection matrices (embed_dim x embed_dim)
        q = create_tensor(cfg.embed_dim, cfg.embed_dim),
        k = create_tensor(cfg.embed_dim, cfg.embed_dim),
        v = create_tensor(cfg.embed_dim, cfg.embed_dim),
        proj = create_tensor(cfg.embed_dim, cfg.embed_dim)
    }
    local mlp = {
        fc1 = create_tensor(cfg.embed_dim, 4 * cfg.embed_dim),  -- expansion
        fc2 = create_tensor(4 * cfg.embed_dim, cfg.embed_dim)     -- contraction
    }
    -- Kaiming initialization for attention parameters
    local sqrt_k = math.sqrt(1.0 / cfg.embed_dim)
    for _, component in pairs(attn) do
        for i = 1, component.rows do
            for j = 1, component.cols do
                component:set(i, j, (math.random() - 0.5) * sqrt_k)
            end
        end
    end
    -- Initialization for MLP layers (using a simple uniform distribution)
    for _, component in pairs(mlp) do
        local fan_in = component.rows
        local bound = math.sqrt(3.0 / fan_in)
        for i = 1, component.rows do
            for j = 1, component.cols do
                component:set(i, j, (math.random() - 0.5) * 2 * bound)
            end
        end
    end
    return {
        attn = attn,
        mlp = mlp
    }
end

local function init_model()
    GPT = {
        wte = create_tensor(cfg.vocab_size + 2, cfg.embed_dim),  -- word embeddings (+2 for pad/unk)
        wpe = create_tensor(cfg.seq_len, cfg.embed_dim),         -- positional embeddings
        blocks = {},
        head = create_tensor(cfg.embed_dim, cfg.vocab_size + 2)    -- projection to vocab logits
    }
    local emb_scale = 1 / math.sqrt(cfg.embed_dim)
    for i = 0, (cfg.vocab_size + 1) * cfg.embed_dim - 1 do
        GPT.wte.data[i] = (math.random() - 0.5) * emb_scale
    end
    for i = 0, cfg.seq_len * cfg.embed_dim - 1 do
        GPT.wpe.data[i] = (math.random() - 0.5) * 0.01
    end
    for _ = 1, cfg.num_layers do
        local block = transformer_block()
        table.insert(GPT.blocks, block)
    end
    local head_scale = 1 / math.sqrt(cfg.embed_dim)
    for i = 0, cfg.embed_dim * (cfg.vocab_size + 1) - 1 do
        GPT.head.data[i] = (math.random() - 0.5) * head_scale
    end
end

-- Revised forward pass with multi-head attention, residual connections, layer norm, and MLP.
local function forward(inputs)
    print ("forward function started ...")
    local batch_size = #inputs
    local seq_len = #inputs[1]
    local head_dim = cfg.embed_dim / cfg.num_heads

    -- Build embeddings (sum of token and positional embeddings)
    local activations = {}
    activations[1] = {}
    for b = 1, batch_size do
        activations[1][b] = {}
        for t = 1, seq_len do
            local emb = ffi.new("double[?]", cfg.embed_dim)
            for d = 1, cfg.embed_dim do
                emb[d-1] = GPT.wte:get(inputs[b][t], d) + GPT.wpe:get(t, d)
            end
            activations[1][b][t] = { data = emb, grad = ffi.new("double[?]", cfg.embed_dim) }
        end
    end

    -- Process each transformer layer
    for layer = 1, cfg.num_layers do
        local block = GPT.blocks[layer]
        activations[layer+1] = {}
        for b = 1, batch_size do
            local new_tokens = {}  -- will hold output tokens for batch b
            -- Pre-LayerNorm: normalize each token in the sequence
            local norm_tokens = {}
            for t = 1, seq_len do
                norm_tokens[t] = layer_norm(activations[layer][b][t].data, cfg.embed_dim)
            end

            -- Multi-head Self-Attention
            local attn_outputs = {}
            for t = 1, seq_len do
                attn_outputs[t] = ffi.new("double[?]", cfg.embed_dim)
                for d = 0, cfg.embed_dim-1 do attn_outputs[t][d] = 0 end
            end

            -- For each head, compute Q, K, V projections on normalized tokens
            for h = 1, cfg.num_heads do
                for i = 1, seq_len do
                    -- Compute query for token i for head h
                    local q = ffi.new("double[?]", head_dim)
                    for d = 1, head_dim do
                        q[d-1] = 0
                        for j = 1, cfg.embed_dim do
                            q[d-1] = q[d-1] + norm_tokens[i][j-1] * block.attn.q:get(j, (h-1)*head_dim + d)
                        end
                    end
                    -- Similarly compute key and value for all tokens for head h
                    local Q = q
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
                    -- Compute attention weights for token i over all tokens for head h
                    local scores = ffi.new("double[?]", seq_len)
                    local max_score = -math.huge
                    for j = 1, seq_len do
                        local score = 0
                        for d = 0, head_dim-1 do
                            score = score + Q[d] * keys[j][d]
                        end
                        score = score / math.sqrt(head_dim)
                        scores[j-1] = score
                        if score > max_score then max_score = score end
                    end
                    local sum_exp = 0
                    local exps = ffi.new("double[?]", seq_len)
                    for j = 1, seq_len do
                        exps[j-1] = math.exp(scores[j-1] - max_score)
                        sum_exp = sum_exp + exps[j-1]
                    end
                    -- Weighted sum of values
                    local head_output = ffi.new("double[?]", head_dim)
                    for d = 0, head_dim-1 do head_output[d] = 0 end
                    for j = 1, seq_len do
                        local weight = exps[j-1] / sum_exp
                        for d = 0, head_dim-1 do
                            head_output[d] = head_output[d] + weight * values[j][d]
                        end
                    end
                    -- Accumulate head outputs into the full attention output for token i
                    for d = 0, head_dim-1 do
                        attn_outputs[i][(h-1)*head_dim + d] = attn_outputs[i][(h-1)*head_dim + d] + head_output[d]
                    end
                end
            end

            -- Apply projection of attention outputs
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
                -- Apply dropout on attention projection
                proj_outputs[t] = dropout(proj_outputs[t], cfg.embed_dim, cfg.dropout)
            end

            -- First Residual Connection: add projection output to original activations
            local res1 = {}
            for t = 1, seq_len do
                res1[t] = ffi.new("double[?]", cfg.embed_dim)
                for d = 0, cfg.embed_dim-1 do
                    res1[t][d] = activations[layer][b][t].data[d] + proj_outputs[t][d]
                end
            end

            -- Second sub-layer: Feed-Forward (MLP)
            -- Pre-LayerNorm on res1
            local res1_norm = {}
            for t = 1, seq_len do
                res1_norm[t] = layer_norm(res1[t], cfg.embed_dim)
            end
            local mlp_outputs = {}
            for t = 1, seq_len do
                -- fc1
                local fc1_out = ffi.new("double[?]", 4 * cfg.embed_dim)
                for j = 1, 4 * cfg.embed_dim do
                    local sum = 0
                    for i = 1, cfg.embed_dim do
                        sum = sum + res1_norm[t][i-1] * block.mlp.fc1:get(i, j)
                    end
                    fc1_out[j-1] = math.max(0, sum)  -- ReLU activation
                end
                -- fc2
                local fc2_out = ffi.new("double[?]", cfg.embed_dim)
                for d = 1, cfg.embed_dim do
                    local sum = 0
                    for j = 1, 4 * cfg.embed_dim do
                        sum = sum + fc1_out[j-1] * block.mlp.fc2:get(j, d)
                    end
                    fc2_out[d-1] = sum
                end
                -- Dropout on MLP output
                mlp_outputs[t] = dropout(fc2_out, cfg.embed_dim, cfg.dropout)
            end

            -- Second Residual Connection: add MLP output to res1
            new_tokens = {}
            for t = 1, seq_len do
                new_tokens[t] = {
                    data = ffi.new("double[?]", cfg.embed_dim),
                    grad = ffi.new("double[?]", cfg.embed_dim)
                }
                for d = 0, cfg.embed_dim-1 do
                    new_tokens[t].data[d] = res1[t][d] + mlp_outputs[t][d]
                end
            end
            activations[layer+1][b] = new_tokens
        end
    end

    -- Final logits computation: project final activations to vocabulary space
    local logits = {}
    for b = 1, batch_size do
        logits[b] = {}
        for t = 1, seq_len do
            logits[b][t] = ffi.new("double[?]", cfg.vocab_size + 1)
            for v = 0, cfg.vocab_size do
                local sum = 0
                for d = 1, cfg.embed_dim do
                    sum = sum + activations[#activations][b][t].data[d-1] * GPT.head:get(d, v+1)
                end
                logits[b][t][v] = sum
            end
        end
    end

    return logits
end

-- compute_gradients, cross_entropy, adam_step, get_batch, save_model remain similar
-- (For brevity, we assume these functions are as in your original script.)

local function compute_gradients(logits, targets)
    print ("compute gradients function started ....")
    local batch_size = #targets
    local seq_len = #targets[1]
    GPT.head:zero_grad()
    GPT.wte:zero_grad()
    GPT.wpe:zero_grad()
    for _, block in ipairs(GPT.blocks) do
        block.attn.q:zero_grad()
        block.attn.k:zero_grad()
        block.attn.v:zero_grad()
        block.attn.proj:zero_grad()
        block.mlp.fc1:zero_grad()
        block.mlp.fc2:zero_grad()
    end
    for b = 1, batch_size do
        for t = 1, seq_len do
            local logits_bt = logits[b][t]
            local target = targets[b][t]
            local max_logit = -math.huge
            for v = 0, cfg.vocab_size do
                if logits_bt[v] > max_logit then
                    max_logit = logits_bt[v]
                end
            end
            local sum_exp = 0.0
            local exps = ffi.new("double[?]", cfg.vocab_size+1)
            for v = 0, cfg.vocab_size do
                exps[v] = math.exp(logits_bt[v] - max_logit)
                sum_exp = sum_exp + exps[v]
            end
            for v = 0, cfg.vocab_size do
                local softmax_grad = exps[v] / sum_exp * ((v == target and 1 or 0) - exps[target] / sum_exp)
                for d = 1, cfg.embed_dim do
                    GPT.head:add_grad(d, v+1, softmax_grad)
                end
            end
        end
    end
    local scale = 1.0 / (batch_size * seq_len)
    for i = 0, GPT.head.rows*GPT.head.cols-1 do
        GPT.head.grad[i] = GPT.head.grad[i] * scale
    end
end

local function cross_entropy(logits, targets)
    print("cross entropy started ...")
    local batch_size = #targets
    local seq_len = #targets[1]
    local loss = 0.0
    for b = 1, batch_size do
        for t = 1, seq_len do
            local logits_bt = logits[b][t]
            local target = targets[b][t]
            local max_logit = -math.huge
            for v = 0, cfg.vocab_size do
                if logits_bt[v] > max_logit then max_logit = logits_bt[v] end
            end
            local sum_exp = 0.0
            for v = 0, cfg.vocab_size do
                sum_exp = sum_exp + math.exp(logits_bt[v] - max_logit)
            end
            loss = loss - (logits_bt[target] - max_logit - math.log(sum_exp))
        end
    end
    return loss / (batch_size * seq_len)
end

local function adam_step(param, t)
    print ("Adam step function started...")
    local lr = cfg.lr
    local beta1 = cfg.beta1
    local beta2 = cfg.beta2
    local eps = cfg.eps
    for i = 0, param.rows * param.cols - 1 do
        param.m[i] = beta1 * param.m[i] + (1 - beta1) * param.grad[i]
        param.v[i] = beta2 * param.v[i] + (1 - beta2) * param.grad[i] * param.grad[i]
        local m_hat = param.m[i] / (1 - math.pow(beta1, t))
        local v_hat = param.v[i] / (1 - math.pow(beta2, t))
        param.data[i] = param.data[i] - lr * m_hat / (math.sqrt(v_hat) + eps)
    end
end

local function get_batch(text_tokens)
    local inputs = {}
    local targets = {}
    for _ = 1, cfg.batch_size do
        local start = math.random(1, #text_tokens - cfg.seq_len)
        local input_seq = {}
        local target_seq = {}
        for i = 1, cfg.seq_len do
            input_seq[i] = text_tokens[start + i - 1]
            target_seq[i] = text_tokens[start + i] or cfg.vocab_size
        end
        table.insert(inputs, input_seq)
        table.insert(targets, target_seq)
    end
    return inputs, targets
end

local function save_model()
    db:exec("BEGIN TRANSACTION")
    db:exec("DELETE FROM embeddings")
    local stmt = db:prepare("INSERT INTO embeddings (type, position, dim, value) VALUES (?, ?, ?, ?)")
    for i = 1, GPT.wte.rows do
        for j = 1, GPT.wte.cols do
            stmt:bind_values('wte', i-1, j-1, GPT.wte:get(i, j))
            stmt:step()
            stmt:reset()
        end
    end
    for i = 1, GPT.wpe.rows do
        for j = 1, GPT.wpe.cols do
            stmt:bind_values('wpe', i-1, j-1, GPT.wpe:get(i, j))
            stmt:step()
            stmt:reset()
        end
    end
    stmt:finalize()
    db:exec("DELETE FROM layers")
    stmt = db:prepare("INSERT INTO layers (layer, component, i, j, value) VALUES (?, ?, ?, ?, ?)")
    for layer_idx, block in ipairs(GPT.blocks) do
        local layer = layer_idx - 1
        local components = {
            {name='q', tensor=block.attn.q},
            {name='k', tensor=block.attn.k},
            {name='v', tensor=block.attn.v},
            {name='proj', tensor=block.attn.proj},
            {name='fc1', tensor=block.mlp.fc1},
            {name='fc2', tensor=block.mlp.fc2}
        }
        for _, comp in ipairs(components) do
            for i = 1, comp.tensor.rows do
                for j = 1, comp.tensor.cols do
                    stmt:bind_values(layer, comp.name, i-1, j-1, comp.tensor:get(i, j))
                    stmt:step()
                    stmt:reset()
                end
            end
        end
    end
    stmt:finalize()
    db:exec("DELETE FROM head")
    stmt = db:prepare("INSERT INTO head (i, j, value) VALUES (?, ?, ?)")
    for i = 1, GPT.head.rows do
        for j = 1, GPT.head.cols do
            stmt:bind_values(i-1, j-1, GPT.head:get(i, j))
            stmt:step()
            stmt:reset()
        end
    end
    stmt:finalize()
    db:exec("COMMIT")
end

local function train(text_path)
    local text = io.open(text_path, "r"):read("*a")
    build_vocabulary(text)
    init_model()
    local text_tokens = {}
    for word in text:gmatch("%S+") do
        table.insert(text_tokens, vocab[word:lower()] or cfg.vocab_size)
    end
    print("Training...")
    for iter = 1, cfg.max_iters do
	print ("iteration started ...")
        local inputs, targets = get_batch(text_tokens)
        local logits = forward(inputs)
        local loss = cross_entropy(logits, targets)
        compute_gradients(logits, targets)
        adam_step(GPT.head, iter)
        adam_step(GPT.wte, iter)
        adam_step(GPT.wpe, iter)
        for _, block in ipairs(GPT.blocks) do
	    print ("block working...")
            adam_step(block.attn.q, iter)
            adam_step(block.attn.k, iter)
            adam_step(block.attn.v, iter)
            adam_step(block.attn.proj, iter)
            adam_step(block.mlp.fc1, iter)
            adam_step(block.mlp.fc2, iter)
        end

        save_model()

        if iter % 100 == 0 then
            collectgarbage()
            print(string.format("Iter %d/%d | Loss: %.4f", iter, cfg.max_iters, loss))
        end
    end
    db:close()
    print("Training complete! Model saved to " .. cfg.model_db)
end

if #arg > 0 then
    train(arg[1])
else
    print("Usage: luajit train.lua <training_text_file>")
    os.exit(1)
end
