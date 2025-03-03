#!/bin/bash

# Neural Network in Bash/AWK - Fixed Version
HIDDEN_SIZE=8
LEARNING_RATE=0.1
TARGET_ERROR=0.01
MODEL_PREFIX="model"
DATA_FILE="dataset.csv"

# Initialize colors
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

# 1. Data Loading and Vocabulary Building
declare -A word_counts
declare -A char_counts

echo -e "${GREEN}[1/3] Building vocabularies...${NC}"
while IFS=, read -r w1 w2 w3 target; do
    for word in "$w1" "$w2" "$w3"; do
        ((word_counts[$word]++))
    done
    while IFS= read -n1 c; do
        [[ -n "$c" ]] && ((char_counts[$c]++))
    done <<< "$target"
done < "$DATA_FILE"

# Generate vocab files
words=("${!word_counts[@]}")
chars=("${!char_counts[@]}")
echo "${words[@]}" > "${MODEL_PREFIX}_words.txt"
echo "${chars[@]}" > "${MODEL_PREFIX}_chars.txt"

# 2. Training Core
awk -v HIDDEN="$HIDDEN_SIZE" \
    -v LR="$LEARNING_RATE" \
    -v TARGET_ERR="$TARGET_ERROR" \
    -v MODEL="$MODEL_PREFIX" '
BEGIN {
    # Load vocabularies
    getline < (MODEL "_words.txt")
    split($0, words, " ")
    getline < (MODEL "_chars.txt")
    split($0, chars, " ")
    
    INPUT_SIZE = 3 * length(words)
    OUTPUT_SIZE = 3 * length(chars)
    
    # Initialize weights
    srand()
    for (i=1; i<=INPUT_SIZE; i++) 
        for (j=1; j<=HIDDEN; j++) 
            W1[i,j] = (rand()-0.5)*0.1
    
    for (j=1; j<=HIDDEN; j++) 
        for (k=1; k<=OUTPUT_SIZE; k++) 
            W2[j,k] = (rand()-0.5)*0.1
    
    # Training loop
    epoch = 0
    while (1) {
        epoch++
        total_error = 0
        correct = 0
        count = 0
        
        while ((getline < ARGV[1]) > 0) {
            split($0, parts, ",")
            if (length(parts) != 4) continue
            count++
            
            # Convert input to one-hot
            delete input
            for (i=1; i<=3; i++) {
                word = parts[i]
                for (w=1; w<=length(words); w++) 
                    input[(i-1)*length(words)+w] = (words[w] == word) ? 1 : 0
            }
            
            # Convert target
            target_str = parts[4]
            delete target
            for (i=1; i<=3; i++) {
                c = substr(target_str, i, 1)
                for (j=1; j<=length(chars); j++) 
                    target[(i-1)*length(chars)+j] = (chars[j] == c) ? 1 : 0
            }
            
            # Forward pass
            for (j=1; j<=HIDDEN; j++) {
                sum = 0
                for (i=1; i<=INPUT_SIZE; i++) sum += input[i] * W1[i,j]
                hidden[j] = 1 / (1 + exp(-sum))
            }
            
            for (k=1; k<=OUTPUT_SIZE; k++) {
                sum = 0
                for (j=1; j<=HIDDEN; j++) sum += hidden[j] * W2[j,k]
                output[k] = sum
            }
            
            # Softmax groups
            for (g=0; g<3; g++) {
                max = -1e100
                start = g*length(chars)+1
                end = (g+1)*length(chars)
                
                for (k=start; k<=end; k++) 
                    if (output[k] > max) max = output[k]
                
                sum_exp = 0
                for (k=start; k<=end; k++) {
                    exps[k] = exp(output[k]-max)
                    sum_exp += exps[k]
                }
                for (k=start; k<=end; k++) 
                    probs[k] = exps[k] / sum_exp
            }
            
            # Calculate error
            error = 0
            for (k=1; k<=OUTPUT_SIZE; k++) 
                error += (probs[k] - target[k])^2
            total_error += error / OUTPUT_SIZE
            
            # Backpropagation
            for (k=1; k<=OUTPUT_SIZE; k++) 
                delta3[k] = probs[k] - target[k]
            
            for (j=1; j<=HIDDEN; j++) {
                sum = 0
                for (k=1; k<=OUTPUT_SIZE; k++) sum += delta3[k] * W2[j,k]
                delta2[j] = sum * hidden[j] * (1 - hidden[j])
            }
            
            # Update weights
            for (j=1; j<=HIDDEN; j++) 
                for (k=1; k<=OUTPUT_SIZE; k++) 
                    W2[j,k] -= LR * delta3[k] * hidden[j]
            
            for (i=1; i<=INPUT_SIZE; i++) 
                for (j=1; j<=HIDDEN; j++) 
                    W1[i,j] -= LR * delta2[j] * input[i]
            
            # Check accuracy
            predicted = ""
            for (g=0; g<3; g++) {
                max_p = -1
                max_idx = 1
                start = g*length(chars)+1
                for (k=0; k<length(chars); k++) {
                    if (probs[start+k] > max_p) {
                        max_p = probs[start+k]
                        max_idx = k+1
                    }
                }
                predicted = predicted chars[max_idx]
            }
            if (predicted == target_str) correct++
        }
        close(ARGV[1])
        
        # Save model weights
        printf "" > MODEL "_W1.txt"
        for (i=1; i<=INPUT_SIZE; i++) {
            for (j=1; j<=HIDDEN; j++) 
                printf "%.6f ", W1[i,j] >> MODEL "_W1.txt"
            printf "\n" >> MODEL "_W1.txt"
        }
        
        printf "" > MODEL "_W2.txt"
        for (j=1; j<=HIDDEN; j++) {
            for (k=1; k<=OUTPUT_SIZE; k++) 
                printf "%.6f ", W2[j,k] >> MODEL "_W2.txt"
            printf "\n" >> MODEL "_W2.txt"
        }
        
        # Calculate metrics
        error_rate = total_error / count
        accuracy = (correct / count) * 100
        printf "Epoch %4d - Loss: %.4f - Acc: %5.1f%%\n", epoch, error_rate, accuracy
        
        if (error_rate <= TARGET_ERR || epoch >= 1000) break
    }
}
' "$DATA_FILE"

# 3. Interactive Prediction
echo -e "\n${GREEN}[3/3] Prediction mode${NC}"
awk -v MODEL="$MODEL_PREFIX" '
BEGIN {
    # Load vocabularies
    getline < (MODEL "_words.txt")
    split($0, words, " ")
    getline < (MODEL "_chars.txt")
    split($0, chars, " ")
    
    # Load weights
    INPUT_SIZE = 3 * length(words)
    HIDDEN = 8  # Must match training config
    OUTPUT_SIZE = 3 * length(chars)
    
    # Load W1
    i = 1
    while ((getline < (MODEL "_W1.txt")) > 0) {
        split($0, vals, " ")
        for (j=1; j<=HIDDEN; j++) 
            W1[i,j] = vals[j]
        i++
    }
    
    # Load W2
    j = 1
    while ((getline < (MODEL "_W2.txt")) > 0) {
        split($0, vals, " ")
        for (k=1; k<=OUTPUT_SIZE; k++) 
            W2[j,k] = vals[k]
        j++
    }
    
    # Prediction loop
    while (1) {
        printf "Enter 3 space-separated words: "
        if (!getline input_str) exit
        
        split(input_str, input_words, " ")
        if (length(input_words) != 3) {
            print "Invalid input: need exactly 3 words"
            continue
        }
        
        # Convert to one-hot
        delete input_vec
        for (i=1; i<=3; i++) {
            word = input_words[i]
            found = 0
            for (w=1; w<=length(words); w++) {
                if (words[w] == word) {
                    for (pos=1; pos<=length(words); pos++) 
                        input_vec[(i-1)*length(words)+pos] = (pos == w) ? 1 : 0
                    found = 1
                    break
                }
            }
            if (!found) {
                print "Unknown word:", word
                break
            }
        }
        if (!found) continue
        
        # Forward pass
        for (j=1; j<=HIDDEN; j++) {
            sum = 0
            for (i=1; i<=INPUT_SIZE; i++) 
                sum += input_vec[i] * W1[i,j]
            hidden[j] = 1 / (1 + exp(-sum))
        }
        
        for (k=1; k<=OUTPUT_SIZE; k++) {
            sum = 0
            for (j=1; j<=HIDDEN; j++) 
                sum += hidden[j] * W2[j,k]
            output[k] = sum
        }
        
        # Softmax
        predicted = ""
        for (g=0; g<3; g++) {
            max = -1e100
            start = g*length(chars)+1
            end = (g+1)*length(chars)
            
            for (k=start; k<=end; k++) 
                if (output[k] > max) max = output[k]
            
            sum_exp = 0
            for (k=start; k<=end; k++) {
                exps[k] = exp(output[k]-max)
                sum_exp += exps[k]
            }
            
            max_p = -1
            max_char = ""
            for (k=0; k<length(chars); k++) {
                prob = exps[start+k] / sum_exp
                if (prob > max_p) {
                    max_p = prob
                    max_char = chars[k+1]
                }
            }
            predicted = predicted max_char
        }
        
        print "Prediction:", predicted
    }
}
'
