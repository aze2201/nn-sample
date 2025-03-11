-- LUA basic GPT with Gradient Clipping, Bias Handling, and Improved Backpropagation
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
    vocab_size = 4189,
    embed_dim = 128,
    num_heads = 8,        -- must divide embed_dim evenly
    num_layers = 6,
    seq_len = 128,
    lr = 3e-4,
    batch_size = 16,
    max_iters = 3,     -- increased for actual training
    dropout = 0.2,
    model_db = 'gpt_model.db',
    beta1 = 0.9,
    beta2 = 0.999,
    eps = 1e-8,
    grad_clip = 1.0  -- Gradient clipping threshold
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
        );]],
          [[CREATE TABLE IF NOT EXISTS biases(
            layer INTEGER,
            component TEXT,
            i INTEGER,
            value REAL,
            PRIMARY KEY (layer, component, i)
        );]],
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

--------------------------------------------------
-- Utility functions for forward and backward ops
--------------------------------------------------

-- Layer normalization forward returns normalized vector and cache
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
    local cache = {mean = mean, variance = variance, size = size, eps = eps, vec = vec}
    return norm, cache
end

-- A simplified layer_norm backward (computing dvec given dnorm)
local function layer_norm_backward(dnorm, cache)
    local size = cache.size
    local x = cache.vec
    local mean = cache.mean
    local var = cache.variance
    local eps = cache.eps
    local std = math.sqrt(var + eps)
    local dvec = ffi.new("double[?]", size)
    local dmean = 0
    local dvar = 0
    for i = 0, size - 1 do
        dvar = dvar + dnorm[i] * (x[i] - mean) * -0.5 * (var + eps)^(-1.5)
    end
    for i = 0, size - 1 do
        dmean = dmean + dnorm[i] * -1 / std
    end
    for i = 0, size - 1 do
        dvec[i] = dnorm[i] / std + dvar * 2 * (x[i] - mean) / size + dmean / size
    end
    return dvec
end

-- Dropout forward returns output and a mask
local function dropout_forward(vec, size, dropout_rate)
    local out = ffi.new("double[?]", size)
    local mask = {}
    for i = 0, size - 1 do
        if math.random() < dropout_rate then
            out[i] = 0
            mask[i] = 0
        else
            out[i] = vec[i]
            mask[i] = 1
        end
    end
    return out, mask
end

-- Dropout backward: multiply gradient by mask
local function dropout_backward(dout, mask, size)
    local dvec = ffi.new("double[?]", size)
    for i = 0, size - 1 do
        dvec[i] = dout[i] * mask[i]
    end
    return dvec
end

-- Linear forward: computes output = input * tensor + bias
local function linear_forward(input, tensor, bias)
    local in_features = tensor.rows
    local out_features = tensor.cols
    local output = ffi.new("double[?]", out_features)
    for j = 1, out_features do
        local sum = bias:get(j)  -- Initialize with bias
        for i = 1, in_features do
            sum = sum + input[i-1] * tensor:get(i, j)
        end
        output[j-1] = sum
    end
    return output
end

-- Linear backward: compute gradients and propagate gradient to input, includes bias gradients
local function linear_backward(input, doutput, tensor, grad_tensor, bias, grad_bias)
    local in_features = tensor.rows
    local out_features = tensor.cols
    local dinput = ffi.new("double[?]", in_features)

    -- Initialize dinput to zeros
    for i = 0, in_features-1 do
        dinput[i] = 0
    end

    for i = 1, in_features do
        for j = 1, out_features do
            -- Gradient for the weights
            grad_tensor[(i-1)*out_features + (j-1)] = grad_tensor[(i-1)*out_features + (j-1)] + input[i-1] * doutput[j-1]
            -- Gradient for the input (accumulated)
            dinput[i-1] = dinput[i-1] + tensor:get(i, j) * doutput[j-1]
        end
    end

    -- Gradient for the biases
    for j = 1, out_features do
        grad_bias[j-1] = grad_bias[j-1] + doutput[j-1]
    end
    return dinput
end

-- ReLU forward returns output and a mask of activated units
local function relu_forward(input, size)
    local out = ffi.new("double[?]", size)
    local mask = {}
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

-- ReLU backward: multiply gradient by mask
local function relu_backward(dout, mask, size)
    local dinput = ffi.new("double[?]", size)
    for i = 0, size-1 do
        dinput[i] = dout[i] * mask[i]
    end
    return dinput
end

--------------------------------------------------
-- Tensor creation helper (Now includes bias)
--------------------------------------------------
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

local function create_bias(size)
    local data = ffi.new("double[?]", size)
    local grad = ffi.new("double[?]", size)
    local m = ffi.new("double[?]", size)
    local v = ffi.new("double[?]", size)
    ffi.fill(data, ffi.sizeof("double") * size, 0)
    ffi.fill(grad, ffi.sizeof("double") * size, 0)
    ffi.fill(m, ffi.sizeof("double") * size, 0)
    ffi.fill(v, ffi.sizeof("double") * size, 0)
    return {
        data = data,
        grad = grad,
        m = m,
        v = v,
        size = size,
        get = function(self, i)
            return self.data[i-1]
        end,
        set = function(self, i, val)
            self.data[i-1] = val
        end,
        add_grad = function(self, i, val)
          self.grad[i-1] = self.grad[i-1] + val
        end,
        zero_grad = function(self)
            ffi.fill(self.grad, ffi.sizeof("double") * self.size, 0)
        end
    }
end

--------------------------------------------------
-- Transformer block constructor (initializes parameters)
--------------------------------------------------
local function transformer_block()
    -- Attention components
    local attn = {
        q = create_tensor(cfg.embed_dim, cfg.embed_dim),
        k = create_tensor(cfg.embed_dim, cfg.embed_dim),
        v = create_tensor(cfg.embed_dim, cfg.embed_dim),
        proj = create_tensor(cfg.embed_dim, cfg.embed_dim),
        q_bias = create_bias(cfg.embed_dim),
        k_bias = create_bias(cfg.embed_dim),
        v_bias = create_bias(cfg.embed_dim),
        proj_bias = create_bias(cfg.embed_dim)
    }

    -- MLP components
    local mlp = {
        fc1 = create_tensor(cfg.embed_dim, 4 * cfg.embed_dim),
        fc2 = create_tensor(4 * cfg.embed_dim, cfg.embed_dim),
        fc1_bias = create_bias(4 * cfg.embed_dim),
        fc2_bias = create_bias(cfg.embed_dim)
    }

    -- Initialize attention weights
    local sqrt_k = math.sqrt(1.0 / cfg.embed_dim)
    for _, component in pairs(attn) do
      if component.rows then -- Check if it's a tensor
        for i = 1, component.rows do
            for j = 1, component.cols do
                component:set(i, j, (math.random() - 0.5) * sqrt_k)
            end
        end
      elseif component.size then -- Check if it's a bias
        for i = 1, component.size do
            component:set(i, (math.random() - 0.5) * sqrt_k)  -- Initialize biases
        end
      end
    end

    -- Initialize MLP weights
   for _, component in pairs(mlp) do
      if component.rows then -- Check if it's a tensor (not bias)
          local fan_in = component.rows
          local bound = math.sqrt(3.0 / fan_in)
          for i = 1, component.rows do
              for j = 1, component.cols do
                  component:set(i, j, (math.random() - 0.5) * 2 * bound)
              end
          end
      elseif component.size then
          for i = 1, component.size do
              component:set(i, 0)
          end
      end
    end

    return {
        attn = attn,
        mlp = mlp  -- Ensure MLP is included in the returned block
    }
end

--------------------------------------------------
-- Transformer block forward with causal masking and cache for backprop
--------------------------------------------------
local function transformer_block_forward(block, norm_tokens)
    local head_dim = cfg.embed_dim / cfg.num_heads
    local seq_len = #norm_tokens
    local cache = {}
    cache.attention = {}
    cache.attention.heads = {} -- will store per-token per-head caches
    local attn_outputs = {}
    for t = 1, seq_len do
        attn_outputs[t] = ffi.new("double[?]", cfg.embed_dim)
        for d = 0, cfg.embed_dim-1 do attn_outputs[t][d] = 0 end
        cache.attention.heads[t] = {}
    end
    for h = 1, cfg.num_heads do
        for i = 1, seq_len do
            local head_cache = {}
            -- Compute query for token i for head h
            local q = ffi.new("double[?]", head_dim)
            for d = 1, head_dim do
                local bias_val = block.attn.q_bias:get((h-1)*head_dim + d)
                q[d-1] = bias_val -- Add bias to q
                for j = 1, cfg.embed_dim do
                    q[d-1] = q[d-1] + norm_tokens[i][j-1] * block.attn.q:get(j, (h-1)*head_dim + d)
                end
            end
            head_cache.q = q
            -- Compute keys and values for all tokens
            local keys = {}
            local values = {}
            for j = 1, seq_len do
                local k = ffi.new("double[?]", head_dim)
                local v = ffi.new("double[?]", head_dim)
                for d = 1, head_dim do
                     local k_bias_val = block.attn.k_bias:get((h-1)*head_dim + d)
                    local v_bias_val = block.attn.v_bias:get((h-1)*head_dim + d)
                    k[d-1] = k_bias_val  -- Add bias
                    v[d-1] = v_bias_val  -- Add bias
                    for r = 1, cfg.embed_dim do
k[d-1] = k[d-1] + norm_tokens[j][r-1] * block.attn.k:get(r, (h-1)*head_dim + d)
                    v[d-1] = v[d-1] + norm_tokens[j][r-1] * block.attn.v:get(r, (h-1)*head_dim + d)
                end
            end
            keys[j] = k
            values[j] = v
        end
        head_cache.keys = keys
        head_cache.values = values
        -- Compute attention scores with causal masking:
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
                score = -math.huge
            end
            scores[j-1] = score
            if score > max_score then max_score = score end
        end
        head_cache.scores = scores
        head_cache.max_score = max_score
        local exps = ffi.new("double[?]", seq_len)
        local sum_exp = 0
        for j = 1, seq_len do
            if scores[j-1] == -math.huge then
                exps[j-1] = 0
            else
                exps[j-1] = math.exp(scores[j-1] - max_score)
            end
            sum_exp = sum_exp + exps[j-1]
        end
        head_cache.exps = exps
        head_cache.sum_exp = sum_exp
        local attn_weights = ffi.new("double[?]", seq_len)
        for j = 1, seq_len do
            attn_weights[j-1] = exps[j-1] / sum_exp
        end
        head_cache.attn_weights = attn_weights
        local head_output = ffi.new("double[?]", head_dim)
        for d = 0, head_dim-1 do head_output[d] = 0 end
        for j = 1, seq_len do
            local weight = attn_weights[j-1]
            for d = 0, head_dim-1 do
                head_output[d] = head_output[d] + weight * values[j][d]
            end
        end
        head_cache.head_output = head_output
        for d = 0, head_dim-1 do
            attn_outputs[i][(h-1)*head_dim + d] = attn_outputs[i][(h-1)*head_dim + d] + head_output[d]
        end
        cache.attention.heads[i][h] = head_cache
    end
    end

    -- Apply projection of attention outputs (including bias)
    cache.attention.proj = {}
    local proj_outputs = {}
    for t = 1, seq_len do
        proj_outputs[t] = ffi.new("double[?]", cfg.embed_dim)
        local proj_cache = {input = attn_outputs[t]}
        for d = 1, cfg.embed_dim do
            local sum = block.attn.proj_bias:get(d)  -- Initialize with bias
            for i = 1, cfg.embed_dim do
                sum = sum + attn_outputs[t][i-1] * block.attn.proj:get(i, d)
            end
            proj_outputs[t][d-1] = sum
        end
        local dropped, dropout_mask = dropout_forward(proj_outputs[t], cfg.embed_dim, cfg.dropout)
        proj_outputs[t] = dropped
        proj_cache.dropout_mask = dropout_mask
        cache.attention.proj[t] = proj_cache
    end
    -- First residual connection: add projection output to the (non-normalized) input.
    cache.attention.res1 = {}
    local res1 = {}
    for t = 1, seq_len do
        res1[t] = ffi.new("double[?]", cfg.embed_dim)
        for d = 0, cfg.embed_dim-1 do
            res1[t][d] = norm_tokens[t][d] + proj_outputs[t][d]
        end
        cache.attention.res1[t] = res1[t]
    end
    -- Second sub-layer: MLP branch.
    cache.mlp = {}
    cache.mlp.fc1 = {}
    cache.mlp.fc2 = {}
    cache.mlp.relu_mask = {}
    local mlp_outputs = {}
    for t = 1, seq_len do
        local norm_res1, norm_cache = layer_norm_forward(res1[t], cfg.embed_dim)
        if not cache.mlp[t] then cache.mlp[t] = {} end
        cache.mlp[t].norm_cache = norm_cache
        local fc1_out = ffi.new("double[?]", 4 * cfg.embed_dim)
        local fc1_cache = {input = norm_res1}
        for j = 1, 4 * cfg.embed_dim do
            local sum = block.mlp.fc1_bias:get(j)  -- Initialize with bias
            for i = 1, cfg.embed_dim do
                sum = sum + norm_res1[i-1] * block.mlp.fc1:get(i, j)
            end
            fc1_out[j-1] = sum
        end
        local relu_out, relu_mask = relu_forward(fc1_out, 4 * cfg.embed_dim)
        fc1_cache.output = fc1_out
        fc1_cache.relu_mask = relu_mask
        cache.mlp.fc1[t] = fc1_cache
        local fc2_out = ffi.new("double[?]", cfg.embed_dim)
        local fc2_cache = {input = relu_out}
        for d = 1, cfg.embed_dim do
            local sum = block.mlp.fc2_bias:get(d) -- Initialize with bias
            for j = 1, 4 * cfg.embed_dim do
                sum = sum + relu_out[j-1] * block.mlp.fc2:get(j, d)
            end
            fc2_out[d-1] = sum
        end
        fc2_cache.output = fc2_out
        cache.mlp.fc2[t] = fc2_cache
        local mlp_drop, mlp_dropout_mask = dropout_forward(fc2_out, cfg.embed_dim, cfg.dropout)
        mlp_outputs[t] = mlp_drop
        cache.mlp.fc2[t].dropout_mask = mlp_dropout_mask
    end
    cache.residual_final = {}
    local out_tokens = {}
    for t = 1, seq_len do
        out_tokens[t] = { data = ffi.new("double[?]", cfg.embed_dim) }
        for d = 0, cfg.embed_dim-1 do
            out_tokens[t].data[d] = res1[t][d] + mlp_outputs[t][d]
        end
        cache.residual_final[t] = out_tokens[t].data
    end
    return out_tokens, cache
end

--------------------------------------------------
-- Forward pass for the full model (batch version) with cache storage
--------------------------------------------------
local function forward_with_cache(inputs)
    local batch_size = #inputs
    local seq_len = #inputs[1]
    local caches = {}
    caches.embeddings = {}
    local activations = {}
    activations[1] = {}
    for b = 1, batch_size do
        activations[1][b] = {}
        caches.embeddings[b] = {}
        for t = 1, seq_len do
            local emb = ffi.new("double[?]", cfg.embed_dim)
            for d = 1, cfg.embed_dim do
                emb[d-1] = GPT.wte:get(inputs[b][t], d) + GPT.wpe:get(t, d)
            end
            activations[1][b][t] = { data = emb }
            caches.embeddings[b][t] = { input_token = inputs[b][t] }
        end
    end
    caches.transformer = {}
    for layer = 1, cfg.num_layers do
        caches.transformer[layer] = {}
        activations[layer+1] = {}
        for b = 1, batch_size do
            local norm_tokens = {}
            local norm_caches = {}
            for t = 1, seq_len do
                local norm, norm_cache = layer_norm_forward(activations[layer][b][t].data, cfg.embed_dim)
                norm_tokens[t] = norm
                norm_caches[t] = norm_cache
            end
            caches.transformer[layer].norm = norm_caches
            local block_out, block_cache = transformer_block_forward(GPT.blocks[layer], norm_tokens)
            caches.transformer[layer].block = block_cache
            activations[layer+1][b] = block_out
        end
    end
    caches.projection = {}
    local logits = {}
    local final_layer = #activations
    for b = 1, batch_size do
        logits[b] = {}
        caches.projection[b] = {}
        for t = 1, seq_len do
            local token_act = activations[final_layer][b][t].data
            local logit = ffi.new("double[?]", cfg.vocab_size + 1)
            local proj_cache = {input = token_act}
            for v = 0, cfg.vocab_size do
                local sum = GPT.head_bias:get(v+1)  -- Initialize with bias
                for d = 1, cfg.embed_dim do
                    sum = sum + token_act[d-1] * GPT.head:get(d, v+1)
                end
                logit[v] = sum
            end
            logits[b][t] = logit
            caches.projection[b][t] = proj_cache
        end
    end
    return logits, caches, activations
end

--------------------------------------------------
-- Backward pass for final projection layer
--------------------------------------------------
local function backward_projection(dlogits, caches, activations)
    local batch_size = #dlogits
    local seq_len = #dlogits[1]
    for b = 1, batch_size do
        for t = 1, seq_len do
            local proj_cache = caches.projection[b][t]
            local token_act = proj_cache.input
            local dtoken = ffi.new("double[?]", cfg.embed_dim)
            for d = 0, cfg.embed_dim-1 do dtoken[d] = 0 end
            for v = 0, cfg.vocab_size do
                local grad = dlogits[b][t][v]
                for d = 1, cfg.embed_dim do
                    GPT.head:add_grad(d, v+1, token_act[d-1] * grad)
                    dtoken[d-1] = dtoken[d-1] + GPT.head:get(d, v+1) * grad
                end
                 GPT.head_bias:add_grad(v+1, grad)  -- Update head bias
            end
            activations[#activations][b][t].ddata = dtoken
        end
    end
end

--------------------------------------------------
-- A very simplified backward pass for one transformer block. (with bias handling)
--------------------------------------------------

local function backward_transformer_block(block, block_cache, dactivation, norm_cache)
    local seq_len = #dactivation
    local head_dim = cfg.embed_dim / cfg.num_heads

    -- Backprop through residual connection after MLP
    local dres2 = {}
    for t = 1, seq_len do
        dres2[t] = ffi.new("double[?]", cfg.embed_dim)
        local src = dactivation[t]
        if type(src) == "table" then
            for d = 0, cfg.embed_dim-1 do
                dres2[t][d] = src[d+1] or 0  -- Handle potential nil values
            end
        else

            ffi.copy(dres2[t], src, ffi.sizeof("double") * cfg.embed_dim)
        end
    end

    -- Backprop through MLP branch
    local dmlp = {}
    for t = 1, seq_len do
      if not block.mlp or not block.mlp.fc1 or not block.mlp.fc2 then
          error("MLP components missing in transformer block")  -- More robust error
      end

        local fc1_cache = block_cache.mlp.fc1[t]
        local fc2_cache = block_cache.mlp.fc2[t]
        local norm_cache_mlp = block_cache.mlp[t].norm_cache

        -- Backprop through MLP dropout
        local dmlp_dropout = dropout_backward(dres2[t], fc2_cache.dropout_mask, cfg.embed_dim)

        -- Backprop through fc2
        local dfc2_out = dmlp_dropout
        local drelu_out = ffi.new("double[?]", 4 * cfg.embed_dim)
        local grad_fc2_bias = ffi.new("double[?]", cfg.embed_dim)  -- Gradient for fc2 bias
        ffi.fill(grad_fc2_bias, ffi.sizeof("double") * cfg.embed_dim, 0)

        for j = 1, 4 * cfg.embed_dim do
            local sum = 0
            for d = 1, cfg.embed_dim do
                local grad = dfc2_out[d-1]
                local weight = block.mlp.fc2:get(j, d)
                sum = sum + grad * weight
                block.mlp.fc2:add_grad(j, d, fc1_cache.output[j-1] * grad)
                grad_fc2_bias[d-1] = grad_fc2_bias[d-1] + fc1_cache.output[j-1] * grad
            end
            drelu_out[j-1] = sum
        end

        -- Backprop through ReLU
        local dfc1_out = relu_backward(drelu_out, fc1_cache.relu_mask, 4 * cfg.embed_dim)

        -- Backprop through fc1
        local dnorm_mlp = ffi.new("double[?]", cfg.embed_dim)
        local grad_fc1_bias = ffi.new("double[?]", 4 * cfg.embed_dim) -- Gradient for fc1 bias
        ffi.fill(grad_fc1_bias, ffi.sizeof("double") * (4 * cfg.embed_dim), 0)

        for i = 1, cfg.embed_dim do
            local sum = 0
            for j = 1, 4 * cfg.embed_dim do
                local grad = dfc1_out[j-1]
                local weight = block.mlp.fc1:get(i, j)
                sum = sum + grad * weight
                block.mlp.fc1:add_grad(i, j, norm_cache_mlp.vec[i-1] * grad)
                grad_fc1_bias[j-1] = grad_fc1_bias[j-1] + norm_cache_mlp.vec[i-1] * grad  -- Accumulate bias gradient
            end
            dnorm_mlp[i-1] = sum
        end
        -- Update fc1 and fc2 biases
        for j=1, 4 * cfg.embed_dim do
            block.mlp.fc1_bias:add_grad(j, grad_fc1_bias[j-1])
        end
        for j=1, cfg.embed_dim do
            block.mlp.fc2_bias:add_grad(j, grad_fc2_bias[j-1])
        end

        -- Backprop through layer norm in MLP branch
        dmlp[t] = layer_norm_backward(dnorm_mlp, norm_cache_mlp)
    end

    -- Backprop through residual connection after attention
    local dres1 = {}
    for t = 1, seq_len do
        dres1[t] = ffi.new("double[?]", cfg.embed_dim)
        for d = 0, cfg.embed_dim-1 do
dres1[t][d] = dmlp[t][d] + dres2[t][d]
        end
    end

    -- Backprop through attention branch
    local dattn_proj = {}
    for t = 1, seq_len do
        local proj_cache = block_cache.attention.proj[t]
        local dproj_drop = dropout_backward(dres1[t], proj_cache.dropout_mask, cfg.embed_dim)

        -- Backprop through projection layer
        local dproj_in = ffi.new("double[?]", cfg.embed_dim)
        local grad_proj_bias = ffi.new("double[?]", cfg.embed_dim)  -- Gradient for proj bias
        ffi.fill(grad_proj_bias, ffi.sizeof("double") * cfg.embed_dim, 0)
        for i = 1, cfg.embed_dim do
            local sum = 0
            for d = 1, cfg.embed_dim do
                local grad = dproj_drop[d-1]
                local weight = block.attn.proj:get(i, d)
                sum = sum + grad * weight
                block.attn.proj:add_grad(i, d, proj_cache.input[i-1] * grad)
                grad_proj_bias[d-1] = grad_proj_bias[d-1] + proj_cache.input[i-1] * grad
            end
            dproj_in[i-1] = sum
        end
          -- Update proj bias
        for d = 1, cfg.embed_dim do
          block.attn.proj_bias:add_grad(d, grad_proj_bias[d-1])
        end
        dattn_proj[t] = dproj_in
    end

    -- Backprop through multi-head attention
    local dattn = {}
    for t = 1, seq_len do
        dattn[t] = ffi.new("double[?]", cfg.embed_dim)
        for h = 1, cfg.num_heads do
            local head_cache = block_cache.attention.heads[t][h]
            for d = 0, head_dim-1 do
                dattn[t][(h-1)*head_dim + d] = dattn_proj[t][(h-1)*head_dim + d]
            end

            local dhead_out = ffi.new("double[?]", head_dim)
            for d = 0, head_dim-1 do
                dhead_out[d] = dattn_proj[t][(h-1)*head_dim + d]
            end

            local dattn_weights = ffi.new("double[?]", seq_len)
            for j = 1, seq_len do
                local value = head_cache.values[j]
                local grad = 0
                for d = 0, head_dim-1 do
                    grad = grad + dhead_out[d] * value[d]
                end
                dattn_weights[j-1] = grad
            end

            local d_scores = ffi.new("double[?]", seq_len)
            local sum_exp = head_cache.sum_exp
            for j = 1, seq_len do
                local attn_weight = head_cache.attn_weights[j-1]
                d_scores[j-1] = (dattn_weights[j-1] * attn_weight * (1 - attn_weight)) / sum_exp
            end

            local dq = ffi.new("double[?]", head_dim)
            ffi.fill(dq, ffi.sizeof("double") * head_dim, 0) -- Initialize dq

            local dk_list = {}
            local dv_list = {}
            for j = 1, seq_len do
                if j <= t then
                    local k = head_cache.keys[j]
                    local v = head_cache.values[j]
                    for d = 0, head_dim-1 do
                        dq[d] = dq[d] + d_scores[j-1] * k[d] / math.sqrt(head_dim)
                    end

                    local dk = ffi.new("double[?]", head_dim)
                    ffi.fill(dk, ffi.sizeof("double") * head_dim, 0) -- Initialize dk
                    for d = 0, head_dim-1 do
                        dk[d] = d_scores[j-1] * head_cache.q[d] / math.sqrt(head_dim)
                    end
                    dk_list[j] = dk

                    local dv = ffi.new("double[?]", head_dim)
                    ffi.fill(dv, ffi.sizeof("double") * head_dim, 0) -- Initialize dv

                    for d = 0, head_dim-1 do
                        dv[d] = dattn_weights[j-1] * head_cache.attn_weights[j-1]
                    end
                    dv_list[j] = dv
                end
            end
            local grad_q_bias = ffi.new("double[?]", head_dim)
            local grad_k_bias = ffi.new("double[?]", head_dim)
            local grad_v_bias = ffi.new("double[?]", head_dim)

            ffi.fill(grad_q_bias, ffi.sizeof("double") * head_dim, 0)
            ffi.fill(grad_k_bias, ffi.sizeof("double") * head_dim, 0)
            ffi.fill(grad_v_bias, ffi.sizeof("double") * head_dim, 0)

            for d = 0, head_dim-1 do
                for i = 1, cfg.embed_dim do
                    local input = norm_cache[t].vec[i-1]
                    block.attn.q:add_grad(i, (h-1)*head_dim + d + 1, input * dq[d])
                    grad_q_bias[d] = grad_q_bias[d] + dq[d]
                end

                for j = 1, seq_len do
                    if j <= t then
                        local dk = dk_list[j]
                        local dv = dv_list[j]
                        local input = norm_cache[t].vec  -- Corrected variable name

                        for i = 1, cfg.embed_dim do
                            block.attn.k:add_grad(i, (h-1)*head_dim + d + 1, input[i-1] * dk[d])
                            block.attn.v:add_grad(i, (h-1)*head_dim + d + 1, input[i-1] * dv[d])
                            grad_k_bias[d] = grad_k_bias[d] + dk[d]
                            grad_v_bias[d] = grad_v_bias[d] + dv[d]
                        end
                    end
                end
            end
            -- Accumulate bias gradients for q, k, and v biases
            for d = 1, head_dim do
              block.attn.q_bias:add_grad((h-1)*head_dim + d, grad_q_bias[d-1])
              block.attn.k_bias:add_grad((h-1)*head_dim + d, grad_k_bias[d-1])
              block.attn.v_bias:add_grad((h-1)*head_dim + d, grad_v_bias[d-1])
            end
        end
    end

    -- Backprop through layer norm for attention branch
    local dnorm = {}
    for t = 1, seq_len do
        dnorm[t] = layer_norm_backward(dattn[t], norm_cache[t])
    end

    -- Combine all gradients
    local dinput = {}
    for t = 1, seq_len do
        dinput[t] = ffi.new("double[?]", cfg.embed_dim)
        for d = 0, cfg.embed_dim-1 do
            dinput[t][d] = dnorm[t][d] + dmlp[t][d]
        end
    end
    return dinput
end



--------------------------------------------------
-- Full backward pass for the model
--------------------------------------------------
local function backward_full(logits, targets, caches, activations)
    local batch_size = #logits
    local seq_len = #logits[1]
    local dlogits = {}
    for b = 1, batch_size do
        dlogits[b] = {}
        for t = 1, seq_len do
            local logit = logits[b][t]
            local dlogit = ffi.new("double[?]", cfg.vocab_size + 1)
            local max_logit = -math.huge
            for v = 0, cfg.vocab_size do
                if logit[v] > max_logit then max_logit = logit[v] end
            end
            local sum_exp = 0
            local exps = {}
            for v = 0, cfg.vocab_size do
                exps[v] = math.exp(logit[v] - max_logit)
                sum_exp = sum_exp + exps[v]
            end
            for v = 0, cfg.vocab_size do
                local softmax = exps[v] / sum_exp
                dlogit[v] = softmax - (v == targets[b][t] and 1 or 0)
            end
            dlogits[b][t] = dlogit
        end
    end

    backward_projection(dlogits, caches, activations)

    for layer = cfg.num_layers, 1, -1 do
        for b = 1, batch_size do
            local dactivation = {}
            for t = 1, seq_len do
                dactivation[t] = activations[layer+1][b][t].ddata or ffi.new("double[?]", cfg.embed_dim)
            end

            local norm_cache = caches.transformer[layer].norm
            local dnorm = backward_transformer_block(
                GPT.blocks[layer],          -- Actual model parameters
                caches.transformer[layer].block,
                dactivation,
                norm_cache
            )

            for t = 1, seq_len do
                activations[layer][b][t].ddata = dnorm[t]
            end
        end
    end

    for b = 1, batch_size do
        for t = 1, seq_len do
            local d_emb = activations[1][b][t].ddata or ffi.new("double[?]", cfg.embed_dim)
            for d = 1, cfg.embed_dim do
                GPT.wte:add_grad(caches.embeddings[b][t].input_token, d, d_emb[d-1])
                GPT.wpe:add_grad(t, d, d_emb[d-1])
            end
        end
    end
end

--------------------------------------------------
-- Vocabulary building
--------------------------------------------------
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

--------------------------------------------------
-- Get a random batch from the text tokens
--------------------------------------------------
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

--------------------------------------------------
-- Save model parameters to the database
--------------------------------------------------
local function save_model()
    db:exec("BEGIN TRANSACTION")
    -- Save embeddings
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

    -- Save layer weights and biases
    db:exec("DELETE FROM layers")
    db:exec("DELETE FROM biases")
    local stmt_weights = db:prepare("INSERT INTO layers (layer, component, i, j, value) VALUES (?, ?, ?, ?, ?)")
    local stmt_biases = db:prepare("INSERT INTO biases (layer, component, i, value) VALUES (?, ?, ?, ?)")

    for layer_idx, block in ipairs(GPT.blocks) do
        local layer = layer_idx - 1
        local components = {
            {name='q', tensor=block.attn.q, bias = block.attn.q_bias},
            {name='k', tensor=block.attn.k, bias = block.attn.k_bias},
            {name='v', tensor=block.attn.v, bias = block.attn.v_bias},
            {name='proj', tensor=block.attn.proj, bias = block.attn.proj_bias},
            {name='fc1', tensor=block.mlp.fc1, bias = block.mlp.fc1_bias},
            {name='fc2', tensor=block.mlp.fc2, bias = block.mlp.fc2_bias}
        }
        for _, comp in ipairs(components) do
            -- Save weights
            if comp.tensor then
              for i = 1, comp.tensor.rows do
                  for j = 1, comp.tensor.cols do
                      stmt_weights:bind_values(layer, comp.name, i-1, j-1, comp.tensor:get(i, j))
                      stmt_weights:step()
                      stmt_weights:reset()
                  end
              end
            end

            -- Save biases
            if comp.bias then
              for i = 1, comp.bias.size do
                stmt_biases:bind_values(layer, comp.name, i-1, comp.bias:get(i))
                stmt_biases:step()
                stmt_biases:reset()
              end
            end
        end
    end
    stmt_weights:finalize()
    stmt_biases:finalize()

    --Save Head and Head bias
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

    -- Save head bias
    local stmt_head_bias = db:prepare("INSERT INTO biases (layer, component, i, value) VALUES (-1, 'head', ?, ?)")
    for i = 1, GPT.head_bias.size do
        stmt_head_bias:bind_values(i - 1, GPT.head_bias:get(i))
        stmt_head_bias:step()
        stmt_head_bias:reset()
    end
    stmt_head_bias:finalize()


    db:exec("COMMIT")
end

--------------------------------------------------
-- Adam update step for parameters (with gradient clipping)
--------------------------------------------------
--------------------------------------------------
-- Adam update step for parameters (with gradient clipping)
--------------------------------------------------
local function adam_step(param, t)
    local lr = cfg.lr
    local beta1 = cfg.beta1
    local beta2 = cfg.beta2
    local eps = cfg.eps
    local grad_norm = 0

    -- Determine the size for gradient norm calculation based on type
    local size
    if param.rows then -- It's a tensor
        size = param.rows * param.cols
    elseif param.size then --It's bias
        size = param.size
    else
      error("Invalid parameter structure passed to adam_step")
    end

    -- Calculate the total gradient norm
    for i = 0, size - 1 do
        grad_norm = grad_norm + param.grad[i] * param.grad[i]
    end
    grad_norm = math.sqrt(grad_norm)

    -- Clip gradients if necessary
    local clip_coef = 1.0
    if grad_norm > cfg.grad_clip then
        clip_coef = cfg.grad_clip / grad_norm
    end

    for i = 0, size - 1 do
        local grad = param.grad[i] * clip_coef  -- Apply clipping
        param.m[i] = beta1 * param.m[i] + (1 - beta1) * grad
        param.v[i] = beta2 * param.v[i] + (1 - beta2) * grad * grad
        local m_hat = param.m[i] / (1 - math.pow(beta1, t))
        local v_hat = param.v[i] / (1 - math.pow(beta2, t))

        -- Determine which data to update based on param type
        if param.rows then -- update tensor data
          param.data[i] = param.data[i] - lr * m_hat / (math.sqrt(v_hat) + eps)
        elseif param.size then --update bias data
          param.data[i] = param.data[i] - lr * m_hat / (math.sqrt(v_hat) + eps)
        end

        param.grad[i] = 0 -- Reset Gradient
    end
end

--------------------------------------------------
-- Initialize the model parameters (including biases)
--------------------------------------------------
local function init_model()
    GPT = {
        wte = create_tensor(cfg.vocab_size + 2, cfg.embed_dim),  -- word embeddings (+2 for pad/unk)
        wpe = create_tensor(cfg.seq_len, cfg.embed_dim),         -- positional embeddings
        blocks = {},
        head = create_tensor(cfg.embed_dim, cfg.vocab_size + 2),    -- projection to vocab logits
        head_bias = create_bias(cfg.vocab_size + 2) -- Bias for the head
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
    -- Initialize head bias to zero
    for i = 1, GPT.head_bias.size do
        GPT.head_bias:set(i, 0)
    end
end

--------------------------------------------------
-- Training loop: forward, loss, backward, update, and save
--------------------------------------------------
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
        local inputs, targets = get_batch(text_tokens)
        local logits, caches, activations = forward_with_cache(inputs)
        local loss = 0.0
        for b = 1, cfg.batch_size do
            for t = 1, cfg.seq_len do
                local logit = logits[b][t]
                local target = targets[b][t]
                local max_logit = -math.huge
                for v = 0, cfg.vocab_size do
                    if logit[v] > max_logit then max_logit = logit[v] end
                end
                local sum_exp = 0.0
                for v = 0, cfg.vocab_size do
                    sum_exp = sum_exp + math.exp(logit[v] - max_logit)
                end
                loss = loss - (logit[target] - max_logit - math.log(sum_exp))
            end
        end
        loss = loss / (cfg.batch_size * cfg.seq_len)
        print(string.format("Iter %d/%d | Loss: %.4f", iter, cfg.max_iters, loss))
        backward_full(logits, targets, caches, activations)

        -- Apply Adam update with gradient clipping to all parameters
        adam_step(GPT.head, iter)
        adam_step(GPT.wte, iter)
        adam_step(GPT.wpe, iter)
        adam_step(GPT.head_bias, iter)

        for _, block in ipairs(GPT.blocks) do
            adam_step(block.attn.q, iter)
            adam_step(block.attn.k, iter)
            adam_step(block.attn.v, iter)
            adam_step(block.attn.proj, iter)
            adam_step(block.attn.q_bias, iter)
            adam_step(block.attn.k_bias, iter)
            adam_step(block.attn.v_bias, iter)
            adam_step(block.attn.proj_bias, iter)
            adam_step(block.mlp.fc1, iter)
            adam_step(block.mlp.fc2, iter)
            adam_step(block.mlp.fc1_bias, iter)
            adam_step(block.mlp.fc2_bias, iter)

        end
        save_model()
	    collectgarbage()
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
