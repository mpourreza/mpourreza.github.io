---
title: 'Attention is all You Need'
date: 2025-07-10
permalink: /posts/2025/07/attention-is-all-you-need/
tags:
  - large language models
  - machine learning
  - llm
  - attention
---

### The Core Concepts
**Self-Attention** 🧠 is a mechanism that allows a sequence (like a sentence) to interact with itself. It calculates how much "attention" each word should pay to every other word in the same sentence. For example, in the sentence "The animal didn't cross the street because **it** was too tired," self-attention helps the model realize that "**it**" refers to the **animal**, not the street.

**Multi-Head Attention** 🧵 is essentially several self-attention mechanisms running in parallel. Instead of having just one "thought process" about a sentence, the model has multiple "heads" looking for different things simultaneously—one might focus on grammar, another on vocabulary relationships, and another on long-range dependencies.

---

To understand the math, we look at how a word "asks" for information from the rest of the sentence. We represent every word using three learned vectors:

* **Query ($$Q$$) 🔍:** This represents "What am I looking for?"
* **Key ($$K$$) 🔑:** This represents "What do I contain?"
* **Value ($$V$$) 📦:** This represents "What information do I provide if I'm a match?"

### The Scaled Dot-Product Formula
The model calculates the relationship using this equation:

$$Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

1.  **Similarity ($$QK^T$$) 📐:** We take the dot product of the Query of one word and the Keys of all other words. A higher score means the words are more mathematically "similar" or relevant to each other.
2.  **Scaling ($$\sqrt{d_k}$$) ⚖️:** We divide by the square root of the dimension of the keys ($$d_k$$). This prevents the scores from getting so large that the gradients vanish during training.
3.  **Softmax 📊:** This turns the scores into probabilities between 0 and 1 that sum to 100%. This decides exactly how much "weight" to give to each word in the sequence.
4.  **Weighting ($$V$$) 🏗️:** We multiply these percentages by the Value vectors. The final output is a weighted sum—a new version of the word that has "absorbed" context from its neighbors.

In **Multi-Head Attention**, we perform this process multiple times in parallel by splitting the original large vectors into smaller "heads." Each head uses its own learned weight matrices, allowing them to focus on different types of relationships simultaneously.

If the dot product ($$QK^T$$) between a specific Query and a Key results in a very high positive number compared to the other keys, what does that suggest about the connection between those two words?

---

In a standard model, a word might be represented by a vector of **512 numbers** (the model dimension, or $$d_{model}$$). Instead of having one giant mechanism look at all 512 numbers at once, Multi-Head Attention divides that workload.

If we use **8 heads**, we don't just cut the list of numbers in half. We use math (linear projections) to "map" that 512-dimensional vector into 8 different 64-dimensional spaces ($$512 \div 8 = 64$$). 

### An Example: "Bank" 🏦🚣
Imagine the word "**Bank**" is represented by a 12-dimensional vector. We split it into **3 heads**, so each head handles **4 dimensions**.

| Head | Vector Size | Focus (Hypothetical) |
| :--- | :---: | :--- |
| **Head 1** 💰 | 4 | Financial context (Is there a mention of "money" or "interest"?) |
| **Head 2** 🌊 | 4 | Geographic context (Is there a mention of "river" or "water"?) |
| **Head 3** 🏗️ | 4 | Grammar/Syntax (Is the word acting as a noun or a verb?) |

---

### Why is this better than one big head?
Each head has its own set of **learned weights** ($$W_Q, W_K, W_V$$). This allows:
* **Parallelism:** The model checks for "money" and "rivers" at the exact same time. ⚡
* **Nuance:** In the sentence *"He sat on the river **bank** to deposit his check,"* Head 1 can focus on the check while Head 2 focuses on the river. A single head might get "confused" trying to average those two very different meanings together. 🎭

After all the heads finish their individual calculations, we **concatenate** (glue) their results back together into one final 512-dimensional vector.


There isn’t a single "perfect" number of heads for every model; instead, the number of heads ($$h$$) is a **hyperparameter**—a setting that engineers choose and test before training begins. 

The decision involves balancing three main factors:

* **Dimensionality ($$d_{model}$$):** The number of heads must be a divisor of the total model dimension. If your model uses 512 dimensions and 8 heads, each head gets 64 dimensions to work with ($$512 \div 8 = 64$$). If you increased to 16 heads, each head would only have 32 dimensions, which might not be enough "room" for a head to learn complex patterns. 📏
* **Representational Diversity:** More heads allow the model to attend to many different parts of a sequence simultaneously (e.g., one head for syntax, one for pronouns, one for specific verbs). However, research (ablation studies) has shown that after a certain point, adding more heads provides diminishing returns or can even hurt performance. 🎭
* **Computational Cost:** While the total number of parameters stays roughly the same because we split the original dimension, the overhead of managing many small matrix multiplications can slow down training on certain hardware. ⚡
  
---

BERT's input is actually a combination of three different types of embeddings summed together. If it only used token embeddings, the model would lose crucial information about the structure and order of the text.

### The Three Components
1.  **Token Embeddings 🔤:** This is the base representation of the words. BERT uses **WordPiece** tokenization, which breaks down complex words into smaller sub-units (e.g., "playing" might become "play" and "##ing"). This helps the model handle rare words or different forms of the same root.
2.  **Segment Embeddings 🧱:** BERT is often trained on pairs of sentences (like for question answering or sentence similarity). Segment embeddings tell the model which tokens belong to "Sentence A" and which belong to "Sentence B."
3.  **Position Embeddings 📍:** Unlike older models (RNNs) that process words one by one, Transformers process all words in a sequence at the same time. Without position embeddings, the model would see a "bag of words" and wouldn't know the difference between "The dog bit the man" and "The man bit the dog."

### Special Tokens
Before these embeddings are summed, two special markers are added to the sequence:
* **[CLS]**: Always the very first token. Its final representation is used for classification tasks (like sentiment analysis).
* **[SEP]**: A separator token used to mark the boundary between two sentences or the end of a single sentence.

---

Positional embeddings are the reason a Transformer knows the difference between "The dog chased the cat" and "The cat chased the dog." Without them, the model treats the input as a "bag of words" with no specific order. 🧩

### Impact on Short vs. Long Text

* **Short Input Text ⚡:** For shorter sequences, positional embeddings are highly effective. The model easily learns the relationship between words that are close together (local context). Since most training data contains short-to-medium sentences, the model becomes very "confident" about what word #2 usually does in relation to word #1.
* **Long Input Text 📏:** This is where things get tricky.
    * **The Hard Limit:** BERT uses **learned absolute positional embeddings**. This means it has a specific vector for "Position 1," "Position 2," up to "Position 512." If you try to feed it 513 tokens, the model physically doesn't have a representation for that extra position.
    * **Out of Distribution:** Even if a model *could* take longer text, if it was only trained on sequences of 128 tokens, it might struggle to understand the "meaning" of position 500 because it never saw it during training.
    * **Context Dilution:** In very long texts, the "signal" from a positional embedding at the start of a document might become too weak to help the model relate it to a word at the very end.

---

# How about BERT?

In BERT, the token and positional embeddings are **learned**. 🛠️ 

They are initialized as random numbers and updated during the training process via backpropagation. By the end of training, the model has developed a unique vector for every index (from 0 to 511) that helps it distinguish the "meaning" of a word based on its location.


Once a model like BERT finishes its training phase, the positional embeddings are **frozen** ❄️. They become a static lookup table that the model uses whenever it processes text. 

### Training vs. Inference
* **During Training 🏗️:** The positional embeddings are "trainable parameters." The model adjusts the numbers in these vectors to minimize error, effectively learning that "Position 1" should look a certain way compared to "Position 2."
* **During Inference 🚀:** When you actually use the model (e.g., asking a chatbot a question), the weights are locked. The model simply retrieves the vector for "Position 1" from its memory. It does not learn or change these values based on the new text it sees.

### The Fine-Tuning Process
During this stage, we usually add a new, randomly initialized "head" (like a linear layer) on top of BERT for a specific task, such as sentiment analysis. We then have two main choices for the positional embeddings:

1.  **Frozen ❄️:** We keep the token and positional embeddings exactly as they were after pre-training. The model relies on its "general" understanding of word order.
2.  **Unfrozen/Trainable 🔥:** We allow the token and positional embeddings to be updated by the new data. This is common when the specific task has a very different structure than the general text BERT was trained on (Wikipedia and Books).

While the model already has a strong grasp of general language from its initial pre-training, fine-tuning allows the "meaning" of specific words to shift slightly to better fit your specific data. 

### Why Update Token Embeddings? 🔄
* **Jargon & Context:** In a general model, the word "cell" might refer to a biological cell. If you fine-tune on legal data, the model can adjust that embedding to lean more toward a "prison cell."
* **New Relationships:** Fine-tuning helps the model learn which words are most important for your specific task (e.g., learning that the word "not" is incredibly important for sentiment analysis).

### Why Unfreeze Position Embeddings?
Even though the model is limited to 512 positions, the *meaning* of those positions can shift during fine-tuning:
* **Domain Adaptation:** In legal or medical documents, the first 50 tokens might always be a specific header or ID. The model can learn to treat these positions as "special" or structural rather than just the start of a sentence. 📂
* **Task-Specific Logic:** For a task like "Sentence Entailment," where two sentences are separated by a `[SEP]` token, the model needs to understand the relationship between the positions in Sentence A and Sentence B. Fine-tuning helps it sharpen that focus. 🎯

### The Learning Rate Balance
To prevent the model from "forgetting" everything it learned during pre-training (a problem called **Catastrophic Forgetting**), we typically use a very small learning rate for the positional and token embeddings. This allows them to "nudge" toward the new data without losing their foundational knowledge of English. ⚖️

## Long Input Text
In the context of AI models, **extrapolation** refers to the ability of a model to handle sequence lengths longer than those it saw during training. 📏

Since BERT uses **learned absolute positional embeddings**, it faces a "hard ceiling." Imagine a hotel with exactly 512 rooms, each with a unique, custom-designed key. 🔑

* **The Trained Limit:** During training, the model "learns" the keys for rooms 1 through 512. It understands exactly where Room 10 is in relation to Room 20.
* **The Extrapolation Failure:** If a guest arrives and asks for Room 513, the model doesn't just lack the key—the room doesn't exist in its "world." Because the embeddings were learned as specific vectors, there is no mathematical rule to "guess" what the vector for 513 should look like.

---

# How to Prevent Vanishing Gradients

## 1. The Variance Problem
In the attention mechanism, we calculate the scores by taking the dot product of a Query ($$Q$$) and a Key ($$K$$):

$$Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Assume that the components of $$Q$$ and $$K$$ are independent random variables with a mean of **0** and a variance of **1**. When you compute the dot product of two vectors of length $$d_k$$:

$$q \cdot k = \sum_{i=1}^{d_k} q_i k_i$$

The variance of the resulting sum is the sum of the variances of the individual terms. Therefore, the variance of the dot product is $$d_k$$. 

As the dimensionality ($$d_k$$) of your keys increases (e.g., to 512 or 1024), the **magnitude** of these dot products can grow quite large, pushing the values far away from zero.



---

## 2. The Softmax "Saturating" Effect
The dot products are passed into a **softmax** function to create a probability distribution. The softmax function for an input vector $$z$$ is:

$$\sigma(z)_i = \frac{e^{z_i}}{\sum e^{z_j}}$$

When the input values ($$z_i$$) have very high magnitudes and large differences between them, the softmax function acts like a "hard-max." It pushes the largest value toward **1.0** and all other values toward **0.0**.



---

## 3. Vanishing Gradients
The derivative of the softmax function is proportional to $$\sigma(z) \cdot (1 - \sigma(z))$$. 

* If the softmax output is near **1** or **0**, the gradient becomes **extremely small** (near zero).
* During backpropagation, if the gradient of the attention layer is nearly zero, the weights in the earlier layers of the network will not be updated effectively.

This is the "vanishing gradient" problem: the model stops learning because the optimization signal cannot pass through the saturated softmax.

---

## 4. How Scaling Fixes It
By dividing the dot product by $$\sqrt{d_k}$$, we effectively scale the variance of the result back down to **1**. 

* **Before scaling:** Variance = $$d_k$$
* **After scaling:** Variance = $$\frac{d_k}{(\sqrt{d_k})^2} = 1$$

By keeping the variance constant regardless of the model's internal size, the inputs to the softmax remain in a range where the function is "sensitive" (the slopes are steep). This ensures that gradients remain large enough to propagate through the network, allowing the model to train efficiently even with very large hidden layers.

---

### Comparison Summary

| Feature | Unscaled Attention | Scaled Attention |
| :--- | :--- | :--- |
| **Dot Product Variance** | Grows with $$d_k$$ | Constant (1) |
| **Softmax Output** | Sparse (one-hot like) | Distributed/Smooth |
| **Gradients** | Vanishing (near 0) | Healthy/Informative |
| **Training Stability** | Poor/Unstable | High |
