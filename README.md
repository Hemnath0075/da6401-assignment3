# DA6401_CH24S016: Assignment 3

## Wandb Report link 
https://wandb.ai/ch24s016-iitm/Seq2SeqAssignment3/reports/Assignment-3--VmlldzoxMjg1NTIyOA

## Problem Statement
The objective of this assignment is to build a **character-level sequence-to-sequence transliteration model** that converts Tamil words (e.g., `"amma"`) into their corresponding tamil script (e.g., `"அம்மா"`). This task involves training an **encoder-decoder** architecture using recurrent neural networks (RNNs) on the **Dakshina dataset**.

---

## Dataset
**Dataset Name:** Dakshina Dataset v1.0  
**Source:** Google Research Datasets

**Files Used:**
- `ta.translit.sampled.train.tsv` — for training
- `ta.translit.sampled.test.tsv` — for evaluation

**File Format:**  
Each line contains three tab-separated columns:  
1. Tamil word in Tamil (Target)  
2. Latin-script transliteration (Input)  
3. Frequency count (Ignored for this task)

---

## Model Overview

This is a **character-level encoder-decoder model** designed to transliterate Latin-script Hindi words into Devanagari. The model architecture includes:

### 1. Encoder
- Takes in the Latin characters (input sequence)
- Converts them into a fixed-size context representation using an RNN-based model
- Supports **RNN**, **GRU**, or **LSTM** cells
- Option to use a **bidirectional encoder** for better context understanding

### 2. Decoder
- Generates the Devanagari characters (output sequence) one token at a time
- Also uses an RNN-based model (RNN/GRU/LSTM)
- Uses **teacher forcing** during training to improve convergence

### 3. (Optional) Attention Mechanism
- Helps the decoder focus on relevant parts of the input sequence at each time step
- Enhances transliteration quality by enabling better alignment between input and output characters

---

## Training Details
- **Loss Function:** `CrossEntropyLoss` with padding mask to ignore padded tokens
- **Optimization:** Standard gradient descent (e.g., Adam optimizer)
- **Evaluation Metric:** Word-level accuracy (percentage of fully correct transliterated words)

---

## Key Features
- Configurable RNN cell types: **RNN**, **GRU**, or **LSTM**
- Option to enable **bidirectional encoding**
- Supports **teacher forcing** for improved training stability
- (Optional) **Attention visualization** to understand decoder focus

---
