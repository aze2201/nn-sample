math.randomseed(os.time())

-- Configuration
local HIDDEN_SIZE = 8
local LEARNING_RATE = 0.1
local TARGET_ERROR = 0.01
local MODEL_FILE = "model.lua"

-- Data storage
local training_data = {}
local word_vocab = {}
local char_vocab = {}
local word_to_ix = {}
local char_to_ix = {}
local ix_to_char = {}

-- Network parameters
local W1, B1, W2, B2

-- Read dataset and build vocabularies
local function load_data()
    for line in io.lines("dataset.csv") do
        local parts = {}
        for part in line:gmatch("[^,]+") do
            table.insert(parts, part)
        end
        if #parts ~= 4 then error("Invalid CSV line: "..line) end
        
        local input_words = {parts[1], parts[2], parts[3]}
        local output_str = parts[4]
        table.insert(training_data, {input=input_words, output=output_str})

        -- Build word vocabulary
        for _, word in ipairs(input_words) do
            if not word_to_ix[word] then
                table.insert(word_vocab, word)
                word_to_ix[word] = #word_vocab
            end
        end

        -- Build character vocabulary
        for c in output_str:gmatch(".") do
            if not char_to_ix[c] then
                table.insert(char_vocab, c)
                char_to_ix[c] = #char_vocab
                ix_to_char[#char_vocab] = c
            end
        end
    end
end

-- Dynamic sizes
local INPUT_SIZE = 3 * #word_vocab
local OUTPUT_SIZE = 3 * #char_vocab

-- Activation functions
local function sigmoid(x)
    return 1 / (1 + math.exp(-x))
end

local function softmax(t)
    local max = -math.huge
    for i=1,#t do
        if t[i] > max then max = t[i] end
    end
    local e, sum = {}, 0
    for i=1,#t do
        e[i] = math.exp(t[i] - max)
        sum = sum + e[i]
    end
    for i=1,#t do
        e[i] = e[i] / sum
    end
    return e
end

-- Initialize weights
local function initialize_weights()
    W1 = {}
    for i=1,INPUT_SIZE do
        W1[i] = {}
        for j=1,HIDDEN_SIZE do
            W1[i][j] = (math.random() - 0.5) * 0.1
        end
    end
    
    B1 = {}
    for j=1,HIDDEN_SIZE do B1[j] = 0 end
    
    W2 = {}
    for i=1,HIDDEN_SIZE do
        W2[i] = {}
        for j=1,OUTPUT_SIZE do
            W2[i][j] = (math.random() - 0.5) * 0.1
        end
    end
    
    B2 = {}
    for j=1,OUTPUT_SIZE do B2[j] = 0 end
end

-- Model persistence
local function save_model()
    local file = io.open(MODEL_FILE, "w")
    file:write("return {\n")
    file:write("HIDDEN_SIZE = "..HIDDEN_SIZE..",\n")
    
    file:write("word_vocab = {\n")
    for _, w in ipairs(word_vocab) do
        file:write(string.format("%q,\n", w))
    end
    file:write("},\n")
    
    file:write("char_vocab = {\n")
    for _, c in ipairs(char_vocab) do
        file:write(string.format("%q,\n", c))
    end
    file:write("},\n")
    
    file:write("W1 = {\n")
    for i=1,INPUT_SIZE do
        file:write("{")
        for j=1,HIDDEN_SIZE do
            file:write(string.format("%.8f,", W1[i][j]))
        end
        file:write("},\n")
    end
    file:write("},\n")
    
    file:write("B1 = {")
    for j=1,HIDDEN_SIZE do
        file:write(string.format("%.8f,", B1[j]))
    end
    file:write("},\n")
    
    file:write("W2 = {\n")
    for i=1,HIDDEN_SIZE do
        file:write("{")
        for j=1,OUTPUT_SIZE do
            file:write(string.format("%.8f,", W2[i][j]))
        end
        file:write("},\n")
    end
    file:write("},\n")
    
    file:write("B2 = {")
    for j=1,OUTPUT_SIZE do
        file:write(string.format("%.8f,", B2[j]))
    end
    file:write("},\n")
    file:write("}\n")
    file:close()
end

local function load_model()
    local model = dofile(MODEL_FILE)
    HIDDEN_SIZE = model.HIDDEN_SIZE
    
    -- Verify vocabularies
    if #model.word_vocab ~= #word_vocab then
        error("Word vocabulary size mismatch")
    end
    for i, w in ipairs(model.word_vocab) do
        if w ~= word_vocab[i] then
            error("Word vocabulary mismatch at index "..i)
        end
    end
    
    if #model.char_vocab ~= #char_vocab then
        error("Character vocabulary size mismatch")
    end
    for i, c in ipairs(model.char_vocab) do
        if c ~= char_vocab[i] then
            error("Character vocabulary mismatch at index "..i)
        end
    end
    
    W1 = model.W1
    B1 = model.B1
    W2 = model.W2
    B2 = model.B2
end

-- Forward propagation
local function forward(input)
    local hidden = {}
    for j=1,HIDDEN_SIZE do
        local sum = B1[j]
        for i=1,INPUT_SIZE do
            sum = sum + input[i] * W1[i][j]
        end
        hidden[j] = sigmoid(sum)
    end
    
    local output = {}
    for k=1,OUTPUT_SIZE do
        local sum = B2[k]
        for j=1,HIDDEN_SIZE do
            sum = sum + hidden[j] * W2[j][k]
        end
        output[k] = sum
    end

    -- Apply softmax to each character group
    local sm = {}
    for i=1,3 do
        local start_idx = (i-1)*#char_vocab + 1
        local end_idx = i*#char_vocab
        local group = {}
        for j=start_idx,end_idx do
            table.insert(group, output[j])
        end
        sm[i] = softmax(group)
    end
    
    -- Flatten the output
    local final_output = {}
    for i=1,3 do
        for j=1,#char_vocab do
            table.insert(final_output, sm[i][j])
        end
    end
    
    return final_output, hidden
end

-- Data converters
local function input_to_vector(words)
    local vec = {}
    for _,word in ipairs(words) do
        local ix = word_to_ix[word]
        if not ix then
            error("Unknown word in input: "..word)
        end
        for i=1,#word_vocab do
            table.insert(vec, (i == ix) and 1 or 0)
        end
    end
    if #vec ~= INPUT_SIZE then
        error("Input vector size mismatch")
    end
    return vec
end

local function output_to_vector(chars)
    local vec = {}
    for i=1,3 do
        local c = chars:sub(i,i)
        local ix = char_to_ix[c]
        if not ix then
            error("Unknown character in output: "..c)
        end
        for j=1,#char_vocab do
            table.insert(vec, (j == ix) and 1 or 0)
        end
    end
    if #vec ~= OUTPUT_SIZE then
        error("Output vector size mismatch")
    end
    return vec
end

-- Training procedure
local function train()
    load_data()
    INPUT_SIZE = 3 * #word_vocab
    OUTPUT_SIZE = 3 * #char_vocab
    
    if #char_vocab == 0 then
        error("No output characters found in dataset")
    end
    
    local file_exists = io.open(MODEL_FILE, "r")
    if file_exists then
        file_exists:close()
        load_model()
        print("Loaded existing model")
    else
        initialize_weights()
        print("Initialized new model")
    end

    local epoch = 0
    while true do
        epoch = epoch + 1
        local total_error = 0
        local correct = 0

        for _, sample in ipairs(training_data) do
            local input = input_to_vector(sample.input)
            local target = output_to_vector(sample.output)
            local output, hidden = forward(input)

            -- Calculate error
            local error = 0
            for i=1,OUTPUT_SIZE do
                error = error + (output[i] - target[i])^2
            end
            total_error = total_error + error/OUTPUT_SIZE

            -- Check prediction
            local predicted = ""
            for i=1,3 do
                local max_val = -math.huge
                local max_ix = 1
                local start_idx = (i-1)*#char_vocab + 1
                for j=0,#char_vocab-1 do
                    local val = output[start_idx + j]
                    if val > max_val then
                        max_val = val
                        max_ix = j + 1
                    end
                end
                predicted = predicted .. ix_to_char[max_ix]
            end
            if predicted == sample.output then
                correct = correct + 1
            end

            -- Backpropagation
            local delta3 = {}
            for k=1,OUTPUT_SIZE do
                delta3[k] = output[k] - target[k]
            end

            local delta2 = {}
            for j=1,HIDDEN_SIZE do
                local sum = 0
                for k=1,OUTPUT_SIZE do
                    sum = sum + delta3[k] * W2[j][k]
                end
                delta2[j] = sum * hidden[j] * (1 - hidden[j])
            end

            -- Update weights with gradient clipping
            for j=1,HIDDEN_SIZE do
                for k=1,OUTPUT_SIZE do
                    local grad = delta3[k] * hidden[j]
                    grad = math.max(-0.1, math.min(0.1, grad))  -- Clip gradients
                    W2[j][k] = W2[j][k] - LEARNING_RATE * grad
                end
            end
            
            for i=1,INPUT_SIZE do
                for j=1,HIDDEN_SIZE do
                    local grad = delta2[j] * input[i]
                    grad = math.max(-0.1, math.min(0.1, grad))  -- Clip gradients
                    W1[i][j] = W1[i][j] - LEARNING_RATE * grad
                end
            end
        end

        local error_rate = 1 - (correct / #training_data)
        print(string.format("Epoch %d - Error: %.2f%%", epoch, error_rate * 100))
        if error_rate <= TARGET_ERROR then break end
    end
    save_model()
    print("Training complete!")
end

-- Prediction function
local function predict(input_str)
    local words = {}
    for word in input_str:gmatch("%S+") do
        table.insert(words, word)
    end
    if #words ~= 3 then
        return "Invalid input: need exactly 3 words"
    end
    
    for _, word in ipairs(words) do
        if not word_to_ix[word] then
            return "Unknown word: "..word
        end
    end

    local input = input_to_vector(words)
    local output = forward(input)

    local predicted = ""
    for i=1,3 do
        local max_val = -math.huge
        local max_ix = 1
        local start_idx = (i-1)*#char_vocab + 1
        for j=0,#char_vocab-1 do
            local val = output[start_idx + j]
            if val > max_val then
                max_val = val
                max_ix = j + 1
            end
        end
        predicted = predicted .. ix_to_char[max_ix]
    end
    return predicted
end

-- Main execution flow
local function main()
    train()
    
    print("\nEnter names (3 space-separated words):")
    while true do
        io.write("> ")
        local input = io.read()
        if not input or input == "" then break end
        print("Result:", predict(input))
    end
end

main()