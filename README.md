# IndoPII-Bench: A Synthetic Dataset for Indonesian PII Detection

## Overview
**IndoPII-Bench** is a specialized synthetic dataset designed to benchmark Data Loss Prevention (DLP) and Named Entity Recognition (NER) systems against Indonesian Personally Identifiable Information (PII). 

Unlike standard PII datasets, this collection is specifically engineered to simulate **"Shadow AI" traffic**: the informal, noisy, and code-mixed prompts that employees often send to LLMs in enterprise environments.

---

## Key Features

*   **Localized Accuracy:** Generates 16-digit National IDs (NIK) using valid regional area codes (e.g., Surabaya, Jakarta) and gender-specific offsets.
*   **Financial Validation:** Credit card numbers are generated via the **Luhn algorithm** to ensure they pass basic checksum validation, and bank accounts mimic Indonesian bank digit lengths (8–15 digits).
*   **Robustness Testing (Decoy IDs):** Includes 16-digit distractor strings that "look" like NIKs but use invalid prefixes (e.g., starting with '99') to measure and minimize False Positives.
*   **Multi-Linguistic Styles:** Moves beyond formal text to include Indonesian-English "Jaksel" code-mixing and heavy slang.
*   **Noise Simulation:** Features OCR-style typos ($0 \rightarrow O$), unconventional spacing, and partial masking to test "Reasoning" vs. "Pattern Matching" paradigms.

---

## Dataset Structure

The dataset is organized into two interconnected CSV files:

### 1. `ground_truth.csv` (The Identities)
This file contains the raw, "clean" data for 5,000 unique synthetic identities.
- id: A unique integer identifier for each record.
- name: Full name generated using Indonesian male and female locales.
- gender: Biological gender (M/F), used to calculate NIK day offsets.
- dob: Date of Birth in YYYY-MM-DD format (ages 17–70).
- nik: 16-digit National ID (Nomor Induk Kependudukan).
- phone: Indonesian mobile numbers using various prefixes (e.g., 0812, 0857) and formats (+62 vs 08).
- bank_account: Numeric strings (8–15 digits) simulating Indonesian bank accounts.
- credit_card: 16-digit numbers that pass the Luhn checksum validation.
- email: Simulated emails based on the identity's name and common domains.
- address: Residential addresses localized to various Indonesian provinces.
- decoy_id: A 16-digit distractor string that mimics NIK formatting but uses invalid prefixes.

### 2. `prompt_dataset.csv` (The Contextual Prompts)
Features the PII embedded into three distinct linguistic paradigms. **Crucially, Decoy IDs are intentionally placed in the same context as NIKs within the Slang templates.** This forces models to "reason" through context rather than simply flagging any 16-digit string, making this dataset significantly more challenging than standard `Faker`-generated lists.

*   **Formal:** Standard business Indonesian (e.g., *"Mohon bantuannya untuk memproses..."*).
*   **Code-Mixed:** A blend of Indonesian and English (e.g., *"Guys, please help me check this user info ASAP ya!"*).
*   **Slang/Noisy:** Highly informal text with typos and masking. This style utilizes **Decoy IDs** (e.g., *"NIKnya.. either {nik} ato {decoy_id}"*) to test if the model can distinguish between valid Indonesian ID structures and valid-length distractors.

---

## Evaluation Methodology

To ensure statistical validity (addressing concerns of sample size alignment), the evaluation framework uses:

1.  **Stratified Sampling:** All models (Regex, BERT, LLM) are evaluated on the exact same sample size ($n=500$ per linguistic style, 1,500 total) to ensure direct comparability of Precision, Recall, and F1 scores.
2.  **Bootstrap Resampling:** Includes $1,000$ iterations of bootstrap resampling ($95\%$ Confidence Interval) to ensure performance metrics are stable and not outliers of the synthetic generation.
3.  **Paradigm Comparison:**
    *   **Pattern-Matching:** (Regex) - Usually fails on noisy/masked data.
    *   **Discriminative:** (IndoBERT-NER) - Strong on formal and code-mixed.
    *   **Reasoning:** (Llama-3-8B-Instruct) - Best for disambiguating NIKs from Decoy IDs in noisy contexts.

---

## Usage

### Noise Injection Logic
The dataset applies a variable noise level based on the linguistic style:
*   **Formal:** 0% noise.
*   **Code-Mixed:** 10% noise.
*   **Slang:** 30% noise (includes OCR typos, dash-spacing, and partial masking).
