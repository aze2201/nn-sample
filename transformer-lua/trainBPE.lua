-- LUA basic GPT with Gradient Clipping, Bias Handling, Improved Backpropagation, and Byte-Level BPE
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
    vocab_size = 256,  -- Initial vocab size for byte-level
    embed_dim = 128,
    num_heads = 8,      -- must divide embed_dim evenly
    num_layers = 6,
    seq_len = 128,
    lr = 3e-4,
    batch_size = 16,
    max_iters = 1,    -- Increased for more training
    dropout = 0.2,
    model_db = 'gpt_model.db',
    beta1 = 0.9,
    beta2 = 0.999,
    eps = 1e-8,
    grad_clip = 1.0,  -- Gradient clipping threshold
    bpe_merges = 50, -- Number of BPE merges to perform
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
        zero_grad = function(self)  -- Correctly define zero_grad for biases
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
                if j <= i then  -- Causal mask
                    for d = 0, head_dim-1 do
                        score = score + q[d] * keys[j][d]
                    end
                    score = score / math.sqrt(head_dim)
                else
                    score = -math.huge  -- Mask out future tokens
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

            -- Accumulate into attn_outputs for this token
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
        local proj_cache = {input = attn_outputs[t]}  -- Input to projection
        for d = 1, cfg.embed_dim do
            local sum = block.attn.proj_bias:get(d)  -- Initialize with bias
            for i = 1, cfg.embed_dim do
                sum = sum + attn_outputs[t][i-1] * block.attn.proj:get(i, d)
            end
            proj_outputs[t][d-1] = sum
        end

        -- Dropout after projection
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

        -- Dropout in MLP
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
            dres1[t][d] = dres2[t][d] + dmlp[t][d] -- Combine gradients
        end
    end

    -- Backprop through attention projection
    local datt_proj = {}
    for t = 1, seq_len do
        local proj_cache = block_cache.attention.proj[t]

        -- Backprop through dropout after projection
        local dproj_dropout = dropout_backward(dres1[t], proj_cache.dropout_mask, cfg.embed_dim)

        datt_proj[t] = ffi.new("double[?]", cfg.embed_dim)
      local grad_proj_bias = ffi.new("double[?]", cfg.embed_dim)
      ffi.fill(grad_proj_bias, ffi.sizeof("double") * cfg.embed_dim, 0)

        for i = 1, cfg.embed_dim do
            local sum = 0
            for j = 1, cfg.embed_dim do
                local grad = dproj_dropout[j-1]
                local weight = block.attn.proj:get(i, j)
                sum = sum + grad * weight
                block.attn.proj:add_grad(i, j, proj_cache.input[i-1] * grad)
              grad_proj_bias[j-1] = grad_proj_bias[j-1] + proj_cache.input[i-1] * grad -- Accumulate bias gradient
            end
            datt_proj[t][i-1] = sum
        end

        -- Update proj bias
        for j = 1, cfg.embed_dim do
            block.attn.proj_bias:add_grad(j, grad_proj_bias[j-1])
        end
    end

    -- Backprop through multi-head attention
    local dnorm = ffi.new("double[?]", seq_len * cfg.embed_dim)  -- Gradient for normalized input

    for h = 1, cfg.num_heads do
        for i = 1, seq_len do
            local head_cache = block_cache.attention.heads[i][h]

            -- Backprop through the head's output aggregation
            local dhead_output = ffi.new("double[?]", head_dim)
            for d = 0, head_dim-1 do
                dhead_output[d] = datt_proj[i][(h-1)*head_dim + d]
            end

            -- Backprop through value multiplication
            local dattn_weights = ffi.new("double[?]", seq_len)
            local dvalues = {}
            for j = 1, seq_len do
                dvalues[j] = ffi.new("double[?]", head_dim)
                local weight = head_cache.attn_weights[j-1]
                local dweight_sum = 0
                for d = 0, head_dim-1 do
                    dvalues[j][d] = weight * dhead_output[d]
                    dweight_sum = dweight_sum + dhead_output[d] * head_cache.values[j][d]
                end
                dattn_weights[j-1] = dweight_sum
            end

            -- Backprop through softmax
            local dexps = ffi.new("double[?]", seq_len)
            local dsum_exp = 0
            for j = 1, seq_len do
                dexps[j-1] = dattn_weights[j-1] / head_cache.sum_exp
                dsum_exp = dsum_exp + dexps[j-1] * (-head_cache.exps[j-1] / head_cache.sum_exp)
            end
            for j = 1, seq_len do
                dexps[j-1] = dexps[j-1] + head_cache.exps[j-1] * dsum_exp
            end
            local dscores = ffi.new("double[?]", seq_len)
            for j = 1, seq_len do
              dscores[j-1] = dexps[j-1] -- no change when score = -huge
            end

            -- Backprop through score calculation
            local dq = ffi.new("double[?]", head_dim)
            ffi.fill(dq, ffi.sizeof("double") * head_dim, 0)
            local dkeys = {}

          local grad_q_bias = ffi.new("double[?]", head_dim) -- Gradient for Q bias
          local grad_k_bias = ffi.new("double[?]", head_dim) -- Gradient for K bias
          local grad_v_bias = ffi.new("double[?]", head_dim) -- Gradient for V bias
          ffi.fill(grad_q_bias, ffi.sizeof("double") * head_dim, 0)
          ffi.fill(grad_k_bias, ffi.sizeof("double") * head_dim, 0)
          ffi.fill(grad_v_bias, ffi.sizeof("double") * head_dim, 0)

            for j = 1, seq_len do
                dkeys[j] = ffi.new("double[?]", head_dim)
                if j <= i then
                    for d = 0, head_dim-1 do
                      local scaled_dscore = dscores[j-1] / math.sqrt(head_dim)
                      dq[d] = dq[d] + head_cache.keys[j][d] * scaled_dscore
                      dkeys[j][d] = head_cache.q[d] * scaled_dscore
                    end
                else
                  for d = 0, head_dim-1 do
                    dkeys[j][d] = 0 -- masked
                  end
                end
            end

            -- Backprop through q, k, v projections
            for d = 1, head_dim do
                for r = 1, cfg.embed_dim do
                    local q_grad = dq[d-1]
                  block.attn.q:add_grad(r, (h-1)*head_dim + d, norm_cache[i].vec[r-1] * q_grad)
                  grad_q_bias[d-1] = grad_q_bias[d-1] + q_grad
                  for j = 1, seq_len do
                        local k_grad = dkeys[j][d-1]
                        block.attn.k:add_grad(r, (h-1)*head_dim + d, norm_cache[j].vec[r-1] * k_grad)
                    grad_k_bias[d-1] = grad_k_bias[d-1] + k_grad  -- Accumulate bias gradient

                        local v_grad = dvalues[j][d-1]
                        block.attn.v:add_grad(r, (h-1)*head_dim + d, norm_cache[j].vec[r-1] * v_grad)
                    grad_v_bias[d-1] = grad_v_bias[d-1] + v_grad
                  end
                end
            end

            -- Update Q, K and V biases
            for d = 1, head_dim do
                block.attn.q_bias:add_grad((h-1)*head_dim + d, grad_q_bias[d-1])
                block.attn.k_bias:add_grad((h-1)*head_dim + d, grad_k_bias[d-1])
                block.attn.v_bias:add_grad((h-1)*head_dim + d, grad_v_bias[d-1])
            end


            -- Accumulate gradients for the normalized input
            for r = 0, cfg.embed_dim-1 do
              local dnorm_idx = (i - 1) * cfg.embed_dim + r
              for d = 1, head_dim do
                  local q_grad = dq[d - 1]
                  dnorm[dnorm_idx] = dnorm[dnorm_idx] + block.attn.q:get(r + 1, (h - 1) * head_dim + d) * q_grad

                  for j = 1, seq_len do
                      local k_grad = dkeys[j][d - 1]
                      local dnorm_j_idx = (j - 1) * cfg.embed_dim + r
                      dnorm[dnorm_j_idx] = dnorm[dnorm_j_idx] + block.attn.k:get(r + 1, (h - 1) * head_dim + d) * k_grad

                      local v_grad = dvalues[j][d - 1]
                      dnorm[dnorm_j_idx] = dnorm[dnorm_j_idx] + block.attn.v:get(r + 1, (h - 1) * head_dim + d) * v_grad
                  end
              end
          end
        end
    end

    -- Backprop through initial layer normalization
    local dinput = {}
    local dnorm_reshaped = {}

    -- Reshape dnorm
    for t = 1, seq_len do
        dnorm_reshaped[t] = ffi.new("double[?]", cfg.embed_dim)
        for d = 0, cfg.embed_dim-1 do
            dnorm_reshaped[t][d] = dnorm[(t-1)*cfg.embed_dim + d]
        end
    end

    for t = 1, seq_len do
        dinput[t] = layer_norm_backward(dnorm_reshaped[t], norm_cache[t])
    end

    return dinput
end
--------------------------------------------------
-- Backward pass for the full model (batch version)
--------------------------------------------------
local function backward(dlogits, caches, activations)
    local batch_size = #dlogits
    local seq_len = #dlogits[1]

    backward_projection(dlogits, caches, activations)

    for layer = cfg.num_layers, 1, -1 do
        local dblock_out_all_b = {}
        for b = 1, batch_size do
          dblock_out_all_b[b] = {}
          for t = 1, seq_len do
            dblock_out_all_b[b][t] = activations[layer+1][b][t].ddata
          end
        end

        for b = 1, batch_size do
          local dblock_in = backward_transformer_block(
                GPT.blocks[layer],
                caches.transformer[layer].block,
                dblock_out_all_b[b],
                caches.transformer[layer].norm
            )
          for t=1, seq_len do
            activations[layer][b][t].ddata = dblock_in[t] -- update gradient
          end
        end
    end

     -- Backprop through embeddings
    for b = 1, batch_size do
        for t = 1, seq_len do
            local token_id = caches.embeddings[b][t].input_token
            local demb = activations[1][b][t].ddata
            for d = 1, cfg.embed_dim do
                GPT.wte:add_grad(token_id, d, demb[d-1])
                GPT.wpe:add_grad(t, d, demb[d-1])
            end
        end
    end
end

--------------------------------------------------
-- Simplified Softmax
--------------------------------------------------
local function softmax(logits)
    local batch_size = #logits
    local seq_len = #logits[1]
    local probs = {}
    local dlogits = {}

    for b = 1, batch_size do
        probs[b] = {}
        dlogits[b] = {}
        for t = 1, seq_len do
            local max_logit = -math.huge
            for i = 0, cfg.vocab_size do
                if logits[b][t][i] > max_logit then
                    max_logit = logits[b][t][i]
                end
            end

            local exps = ffi.new("double[?]", cfg.vocab_size + 1)
            local sum_exp = 0
            for i = 0, cfg.vocab_size do
                exps[i] = math.exp(logits[b][t][i] - max_logit)
                sum_exp = sum_exp + exps[i]
            end

            probs[b][t] = ffi.new("double[?]", cfg.vocab_size + 1)
            for i = 0, cfg.vocab_size do
                probs[b][t][i] = exps[i] / sum_exp
            end

            dlogits[b][t] = ffi.new("double[?]", cfg.vocab_size + 1)
             ffi.fill(dlogits[b][t], ffi.sizeof("double") * (cfg.vocab_size + 1), 0)

        end
    end
    return probs, dlogits
end

--------------------------------------------------
-- Cross-entropy loss
--------------------------------------------------
local function cross_entropy_loss(probs, targets)
    local batch_size = #probs
    local seq_len = #probs[1]
    local loss = 0
    local count = 0
    local dlogits = {}

    for b = 1, batch_size do
      dlogits[b] = {}
        for t = 1, seq_len-1 do -- Iterate only up to seq_len-1
            local target_idx = targets[b][t+1] -- Corrected target indexing
            local prob = probs[b][t][target_idx-1]

            if prob > 0 then
              loss = loss - math.log(prob)
            else
              loss = loss - math.log(1e-15) -- Clip for numerical stability
            end

            count = count + 1
            dlogits[b][t] = ffi.new("double[?]", cfg.vocab_size+1)
            ffi.copy(dlogits[b][t], probs[b][t], ffi.sizeof("double") * (cfg.vocab_size+1))
            dlogits[b][t][target_idx - 1] = dlogits[b][t][target_idx - 1] - 1
        end
    end
    loss = loss / count
    return loss, dlogits
end

--------------------------------------------------
-- Adam Optimizer Update
--------------------------------------------------
local function adam_update(tensor, step)
    local beta1 = cfg.beta1
    local beta2 = cfg.beta2
    local eps = cfg.eps
    local lr = cfg.lr

    local size
    if tensor.rows then
        size = tensor.rows * tensor.cols
    elseif tensor.size then
        size = tensor.size
    else
       error("Invalid tensor structure for Adam update.")
    end

    for i = 0, size - 1 do
        -- Clip gradient
        local grad_val = tensor.grad[i]
        if grad_val > cfg.grad_clip then
            grad_val = cfg.grad_clip
        elseif grad_val < -cfg.grad_clip then
            grad_val = -cfg.grad_clip
        end

        -- Update biased first moment estimate
        tensor.m[i] = beta1 * tensor.m[i] + (1 - beta1) * grad_val

        -- Update biased second raw moment estimate
        tensor.v[i] = beta2 * tensor.v[i] + (1 - beta2) * (grad_val * grad_val)

        -- Compute bias-corrected first moment estimate
        local m_hat = tensor.m[i] / (1 - beta1^step)

        -- Compute bias-corrected second raw moment estimate
        local v_hat = tensor.v[i] / (1 - beta2^step)

        -- Update parameters
        local update = lr * m_hat / (math.sqrt(v_hat) + eps)

      if tensor.rows then -- matrix
          tensor.data[i] = tensor.data[i] - update
      elseif tensor.size then -- bias
          tensor.data[i] = tensor.data[i] - update
      end
    end
end

-- Update all parameters
local function update_parameters(step)
    -- Update embeddings
    adam_update(GPT.wte, step)
    adam_update(GPT.wpe, step)

    -- Update transformer blocks
    for layer = 1, cfg.num_layers do
        for _, component in pairs(GPT.blocks[layer].attn) do
            adam_update(component, step)
        end
        for _, component in pairs(GPT.blocks[layer].mlp) do
           adam_update(component, step)
        end
    end

    -- Update projection head
adam_update(GPT.head, step)
    adam_update(GPT.head_bias, step)
end

--------------------------------------------------
-- Zero gradients (used before backpropagation)
--------------------------------------------------
local function zero_gradients()
    GPT.wte:zero_grad()  -- Use colon operator here
    GPT.wpe:zero_grad()  -- Use colon operator here
    for i = 1, cfg.num_layers do
        for _, component in pairs(GPT.blocks[i].attn) do
            if component.zero_grad then
                component:zero_grad()  -- Use colon operator here
            end
        end
        for _, component in pairs(GPT.blocks[i].mlp) do
             if component.zero_grad then
                component:zero_grad()  -- Use colon operator here
            end
        end
    end
    GPT.head:zero_grad()       -- Use colon operator here
    GPT.head_bias:zero_grad()  -- Use colon operator here
end

--------------------------------------------------
-- Byte-Level BPE Tokenizer Training
--------------------------------------------------
local function train_bpe(text, num_merges)
    -- 1. Initialize vocabulary with individual bytes
    local vocab = {}
    local idx_to_word = {}
    for i = 0, 255 do
        local char = string.char(i)
        vocab[char] = i
        idx_to_word[i] = char
    end

    -- 2. Tokenize text using initial byte-level vocabulary
    local function byte_tokenize(text)
        local tokens = {}
        for i = 1, #text do
            local byte = string.sub(text, i, i)
            table.insert(tokens, vocab[byte])
        end
        return tokens
    end
    local tokenized_text = byte_tokenize(text)

    -- 3. Calculate byte pair frequencies
    local function get_pair_counts(tokenized)
      local counts = {}
      for i = 1, #tokenized - 1 do
          local pair = tokenized[i] .. " " .. tokenized[i + 1] -- space separates byte ids
          counts[pair] = (counts[pair] or 0) + 1
      end
      return counts
    end

    -- 4. Merge loop
    for merge_iter = 1, num_merges do
      local pair_counts = get_pair_counts(tokenized_text)
        if next(pair_counts) == nil then
            print("No more pairs to merge.")
            break  -- Exit if no pairs are found
        end

        local best_pair, max_count = nil, 0
        for pair, count in pairs(pair_counts) do
            if count > max_count then
                max_count = count
                best_pair = pair
            end
        end

        if best_pair == nil then break end -- No pairs to merge.

        local new_word = idx_to_word[tonumber(string.match(best_pair, "^(%d+)"))] .. idx_to_word[tonumber(string.match(best_pair, "%s(%d+)$"))]
        local new_vocab_id = #idx_to_word

        vocab[new_word] = new_vocab_id
        idx_to_word[new_vocab_id] = new_word

      --Update tokenized_text
      local new_tokenized_text = {}
        local i = 1
        while i <= #tokenized_text do
            if i < #tokenized_text then
              local current_pair = tokenized_text[i] .. " " .. tokenized_text[i + 1]
              if current_pair == best_pair then
                table.insert(new_tokenized_text, new_vocab_id)
                i = i + 2
              else
                  table.insert(new_tokenized_text, tokenized_text[i])
                  i = i + 1
              end
            else
                table.insert(new_tokenized_text, tokenized_text[i])
                i = i + 1
            end
        end
        tokenized_text = new_tokenized_text
    end

    return vocab, idx_to_word
end
--------------------------------------------------
-- Tokenization
--------------------------------------------------
local function tokenize(text, vocab)
    local function byte_tokenize(text)
        local tokens = {}
        for i = 1, #text do
            local byte = string.sub(text, i, i)
            table.insert(tokens, vocab[byte])
        end
        return tokens
    end

    local tokenized = byte_tokenize(text)

    local function merge_tokens(tokenized, vocab)
      local merged_tokens = {}
        local i = 1
        while i <= #tokenized do
            if i < #tokenized then
              local first_word = idx_to_word[tokenized[i]]
              local second_word = idx_to_word[tokenized[i + 1]]
              local combined = first_word .. second_word
                if vocab[combined] then  -- Check for merged token
                    table.insert(merged_tokens, vocab[combined])
                    i = i + 2
                else
                    table.insert(merged_tokens, tokenized[i])
                    i = i + 1
                end
            else
                table.insert(merged_tokens, tokenized[i])
                i = i + 1
            end
        end
      return merged_tokens
    end

    local merged = merge_tokens(tokenized, vocab)
    -- Repeat merging until no more merges are possible
    while #merged < #tokenized do
      tokenized = merged
      merged = merge_tokens(tokenized,vocab)
    end

    return merged
end
--------------------------------------------------
--  Generate text
--------------------------------------------------
local function generate(prompt, max_len, temperature)
    temperature = temperature or 1.0

    local input_tokens = tokenize(prompt, vocab)
    local input_seq = {}
    table.insert(input_seq, input_tokens)

    for _ = 1, max_len - #input_tokens do
        local logits, _, _ = forward_with_cache(input_seq)
        local last_logits = logits[1][#logits[1]]

        -- Apply temperature
        if temperature ~= 1.0 then
          for i = 0, cfg.vocab_size do
              last_logits[i] = last_logits[i] / temperature
          end
        end


        local probs, _ = softmax({{last_logits}})
        local next_token_probs = probs[1][1]

        -- Sample from the probability distribution
        local rand_val = math.random()
        local cumulative_prob = 0
        local next_token = 0 -- Initialize
        for i = 0, cfg.vocab_size do
          cumulative_prob = cumulative_prob + next_token_probs[i]
          if rand_val <= cumulative_prob then
              next_token = i
              break
          end
        end

        if next_token == 0 then break end  -- EOS

      table.insert(input_seq[1], next_token)
      if #input_seq[1] > cfg.seq_len then
        -- Truncate to maintain sequence length.  Remove oldest token.
        table.remove(input_seq[1], 1)
      end
    end

    -- Convert tokens back to text:
    local generated_text = ""
    for _, token_id in ipairs(input_seq[1]) do
        generated_text = generated_text .. idx_to_word[token_id]
    end

    return generated_text
end
--------------------------------------------------
-- Data Loading
--------------------------------------------------
local function load_data(filename)
    local file, err = io.open(filename, "r")
    if not file then
        error("Could not open file: " .. filename .. " Error: " .. (err or "unknown error"))
    end
    local text = file:read("*all")
    file:close()
    return text
end
--------------------------------------------------
-- Batch Generation
--------------------------------------------------
local function get_batch(tokenized_data, batch_size, seq_len)
    local data_len = #tokenized_data
    local batch_starts = {}
    for i = 1, batch_size do
        table.insert(batch_starts, math.random(1, data_len - seq_len - 1))
    end

    local inputs = {}
    local targets = {}
    for i, start in ipairs(batch_starts) do
        inputs[i] = {}
        targets[i] = {}
        for j = 0, seq_len -1 do
            table.insert(inputs[i], tokenized_data[start + j])
            table.insert(targets[i], tokenized_data[start + j + 1]) -- Shifted by one
        end
    end
    return inputs, targets
end

--------------------------------------------------
-- Parameter initialization
--------------------------------------------------

local function init_parameters()
    GPT.wte = create_tensor(cfg.vocab_size + 1, cfg.embed_dim)
    GPT.wpe = create_tensor(cfg.seq_len, cfg.embed_dim)
    GPT.blocks = {}
    for i = 1, cfg.num_layers do
        GPT.blocks[i] = transformer_block()
    end
    GPT.head = create_tensor(cfg.embed_dim, cfg.vocab_size + 1)  -- +1 for EOS
    GPT.head_bias = create_bias(cfg.vocab_size + 1) -- and bias

    -- Initialize embedding weights
    local init_range = 0.02
    for i = 1, GPT.wte.rows do
        for j = 1, GPT.wte.cols do
            GPT.wte:set(i, j, (math.random() - 0.5) * 2 * init_range)
        end
    end
    for i = 1, GPT.wpe.rows do
      for j = 1, GPT.wpe.cols do
        GPT.wpe:set(i, j, (math.random() - 0.5) * 2 * init_range)
      end
    end
end
--------------------------------------------------
-- Save and Load Model
--------------------------------------------------
local function save_model()
    -- Start a transaction for atomicity and speed
    db:exec("BEGIN TRANSACTION;")

    -- 1. Save Configuration
    local config_stmt = db:prepare("INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)")
    if not config_stmt then
        error("Failed to prepare config statement: " .. db:errmsg())
    end

    local config_values = {
        vocab_size = cfg.vocab_size,
        embed_dim = cfg.embed_dim,
        num_heads = cfg.num_heads,
        num_layers = cfg.num_layers,
        seq_len = cfg.seq_len,
        lr = cfg.lr,
        dropout = cfg.dropout,
        beta1 = cfg.beta1,
        beta2 = cfg.beta2,
        eps = cfg.eps,
        grad_clip = cfg.grad_clip,
        bpe_merges = cfg.bpe_merges
    }

    for key, value in pairs(config_values) do
        config_stmt:bind(1, key)
        config_stmt:bind(2, value)
        config_stmt:step()
        config_stmt:reset()
    end
    config_stmt:finalize()


    -- 2. Save Vocabulary
    local vocab_stmt = db:prepare("INSERT OR REPLACE INTO vocab (word, id) VALUES (?, ?)")
    if not vocab_stmt then
        error("Failed to prepare vocab statement: " .. db:errmsg())
    end
    for word, id in pairs(vocab) do
        vocab_stmt:bind(1, word)
        vocab_stmt:bind(2, id)
        vocab_stmt:step()
        vocab_stmt:reset()
    end
    vocab_stmt:finalize()

    -- 3. Save Embeddings (wte and wpe)
    local embed_stmt = db:prepare("INSERT OR REPLACE INTO embeddings (type, position, dim, value) VALUES (?, ?, ?, ?)")
    if not embed_stmt then
        error("Failed to prepare embeddings statement: " .. db:errmsg())
    end

    -- Save wte
    for i = 1, GPT.wte.rows do
        for j = 1, GPT.wte.cols do
            embed_stmt:bind(1, 'wte')
            embed_stmt:bind(2, i)
            embed_stmt:bind(3, j)
            embed_stmt:bind(4, GPT.wte:get(i, j))
            embed_stmt:step()
            embed_stmt:reset()
        end
    end

    -- Save wpe
    for i = 1, GPT.wpe.rows do
        for j = 1, GPT.wpe.cols do
            embed_stmt:bind(1, 'wpe')
            embed_stmt:bind(2, i)
            embed_stmt:bind(3, j)
            embed_stmt:bind(4, GPT.wpe:get(i, j))
            embed_stmt:step()
            embed_stmt:reset()
        end
    end
    embed_stmt:finalize()

    -- 4. Save Layers (Transformer Blocks)
    local layer_stmt = db:prepare("INSERT OR REPLACE INTO layers (layer, component, i, j, value) VALUES (?, ?, ?, ?, ?)")
    if not layer_stmt then
        error("Failed to prepare layer statement: " .. db:errmsg())
    end
    for layer_num = 1, cfg.num_layers do
        local layer = GPT.blocks[layer_num]

        -- Save attention weights and biases
        for component_name, component in pairs(layer.attn) do
            if component.rows then  -- It's a weight tensor
                for i = 1, component.rows do
                    for j = 1, component.cols do
                        layer_stmt:bind(1, layer_num)
                        layer_stmt:bind(2, component_name)
                        layer_stmt:bind(3, i)
                        layer_stmt:bind(4, j)
                        layer_stmt:bind(5, component:get(i, j))
                        layer_stmt:step()
                        layer_stmt:reset()
                    end
                end
            elseif component.size then -- It's a bias
                for i = 1, component.size do
                    layer_stmt:bind(1, layer_num)
                    layer_stmt:bind(2, component_name)
                    layer_stmt:bind(3, i)
                    layer_stmt:bind(4, 1)
                    layer_stmt:bind(5, component:get(i))
                    layer_stmt:step()
                    layer_stmt:reset()
                end
            end
        end

        -- Save MLP weights and biases
        for component_name, component in pairs(layer.mlp) do
            if component.rows then  -- It's a weight tensor
                for i = 1, component.rows do
                    for j = 1, component.cols do
                        layer_stmt:bind(1, layer_num)
                        layer_stmt:bind(2, component_name)
                        layer_stmt:bind(3, i)
                        layer_stmt:bind(4, j)
                        layer_stmt:bind(5, component:get(i, j))
                        layer_stmt:step()
                        layer_stmt:reset()
                    end
                end
            elseif component.size then
                for i = 1, component.size do
                    layer_stmt:bind(1, layer_num)
                    layer_stmt:bind(2, component_name)
                    layer_stmt:bind(3, i)
                    layer_stmt:bind(4, 1)
                    layer_stmt:bind(5, component:get(i))
                    layer_stmt:step()
                    layer_stmt:reset()
                end
            end
        end
    end
    layer_stmt:finalize()

    -- 5. Save Head and Head Bias
    local head_stmt = db:prepare("INSERT OR REPLACE INTO layers (layer, component, i, j, value) VALUES (?, ?, ?, ?, ?)")
    if not head_stmt then
        error("Failed to prepare head statement: " .. db:errmsg())
    end
    -- Save head weights (treat as layer 0)
    for i = 1, GPT.head.rows do
        for j = 1, GPT.head.cols do
            head_stmt:bind(1, 0)
            head_stmt:bind(2, 'head')
            head_stmt:bind(3, i)
            head_stmt:bind(4, j)
            head_stmt:bind(5, GPT.head:get(i, j))
            head_stmt:step()
            head_stmt:reset()
        end
    end
    -- Save Head bias
    for i = 1, GPT.head_bias.size do
        head_stmt:bind(1, 0)
        head_stmt:bind(2, 'head_bias')
        head_stmt:bind(3, i)
        head_stmt:bind(4, 1)
        head_stmt:bind(5, GPT.head_bias:get(i))
        head_stmt:step()
        head_stmt:reset()
    end
    head_stmt:finalize()

    -- Commit the transaction
    db:exec("COMMIT;")

    print("Model saved to " .. cfg.model_db)
end



local function load_model()
    -- Load config
    local stmt = db:prepare("SELECT key, value FROM config")
    while stmt:step() do
        local key = stmt:get_value(0)
        local value = stmt:get_value(1)
        cfg[key] = value
    end
    stmt:finalize()

    -- Load vocabulary
    vocab = {}
    idx_to_word = {}
    local stmt2 = db:prepare("SELECT word, id FROM vocab")
    while stmt2:step() do
        local word = stmt2:get_value(0)
        local id = stmt2:get_value(1)
        vocab[word] = id
        idx_to_word[id] = word
    end
    stmt2:finalize()

    -- Init parameters based on loaded config
    init_parameters()

    -- Load embeddings
    local embed_stmt = db:prepare("SELECT type, position, dim, value FROM embeddings")
    while embed_stmt:step() do
        local type = embed_stmt:get_value(0)
        local pos = embed_stmt:get_value(1)
        local dim = embed_stmt:get_value(2)
        local val = embed_stmt:get_value(3)
        if type == 'wte' then
            GPT.wte:set(pos, dim, val)
        elseif type == 'wpe' then
            GPT.wpe:set(pos, dim, val)
        end
    end
    embed_stmt:finalize()

    -- Load all layers (including head and biases)
    local layer_stmt = db:prepare("SELECT layer, component, i, j, value FROM layers ORDER BY layer")
    while layer_stmt:step() do
        local layer_num = layer_stmt:get_value(0)
        local component_name = layer_stmt:get_value(1)
        local i = layer_stmt:get_value(2)
        local j = layer_stmt:get_value(3)
        local value = layer_stmt:get_value(4)

        -- Handle head and head_bias (layer 0)
        if layer_num == 0 then
            if component_name == 'head' then
                GPT.head:set(i, j, value)
            elseif component_name == 'head_bias' then
                GPT.head_bias:set(i, j, value)
            end
        else
            -- Handle transformer blocks
            if not GPT.blocks[layer_num] then
                GPT.blocks[layer_num] = transformer_block()
            end
            local layer = GPT.blocks[layer_num]
            
            -- Check if component exists in attn or mlp
            if layer.attn[component_name] then
                layer.attn[component_name]:set(i, j, value)
            elseif layer.mlp[component_name] then
                layer.mlp[component_name]:set(i, j, value)
            end
        end
    end
    layer_stmt:finalize()

    print("Model loaded from " .. cfg.model_db)
end

--------------------------------------------------
-- Main Training Loop
--------------------------------------------------

local function main()
    local data_file = "input.txt"  -- Or any other text file
    if #arg > 0 then
      data_file = arg[1]
    end
    local text = load_data(data_file)
    print("Data loaded. Length: " .. #text)

    -- Try loading existing model, otherwise train BPE and initialize
    local model_exists = (db:prepare("SELECT 1 FROM config LIMIT 1"):step() == sqlite3.ROW)
    if model_exists then
      load_model()
      print("Loaded existing model.")
    else
      print("Training BPE tokenizer...")
      vocab, idx_to_word = train_bpe(text, cfg.bpe_merges)
      cfg.vocab_size = #idx_to_word
      print("BPE vocab size: " .. cfg.vocab_size)
      init_parameters()
      print("Initialized new model.")
    end

    local tokenized_data = tokenize(text, vocab)
    print("Data tokenized. Token count: " .. #tokenized_data)

    print("Starting training...")
    local start_time = os.clock()

    for iter = 1, cfg.max_iters do
        local inputs, targets = get_batch(tokenized_data, cfg.batch_size, cfg.seq_len)
        zero_gradients()
        local logits, caches, activations = forward_with_cache(inputs)
        local probs, dlogits_from_softmax = softmax(logits)
        local loss, dlogits_from_loss = cross_entropy_loss(probs, targets)
        backward(dlogits_from_loss, caches, activations)
        update_parameters(iter)

        if iter % 10 == 0 then
          local elapsed_time = os.clock() - start_time
          print(string.format("Iteration: %d, Loss: %.4f, Time: %.2f s", iter, loss, elapsed_time))
          start_time = os.clock()
        end

        if iter % 100 == 0 then
          save_model()
          local prompt = "The quick brown"  -- Example prompt
          local generated = generate(prompt, 50, 0.8) -- Generate 50 tokens with temp 0.7
          print("Generated text (prompt: '"..prompt.."'):\n".. generated)
          print("------------------------")
        end
    end
    print("Training complete!")
    save_model() --save at the end
end

math.randomseed(os.time())
main()
