# IndoPII-Context: A Synthetic Indonesian Dataset for PII Detection and Privacy Leakage Benchmarking

## Description

This dataset consists of synthetic Indonesian Personally Identifiable Information (PII) designed for benchmarking Data Loss Prevention (DLP) and Named Entity Recognition (NER) systems. It is specifically built to test model robustness against "Shadow AI" traffic, informal, noisy, or code-mixed text often found in enterprise prompts.

The data is structured into two interconnected CSV files:
1. ground_truth.csv (The Identities): Contains 5,000 unique synthetic identities. Key features include:
- Regional NIKs: 16-digit National IDs generated using valid regional area codes (e.g., Surabaya, Jakarta, West Java) and gender-specific day offsets.
- Financial Entities: Credit card numbers generated via the Luhn algorithm and bank accounts ranging from 8–15 digits.
- Decoy IDs: 16-digit distractor strings starting with invalid prefixes (e.g., '99') to test for false-positive detection.
- Localized Metadata: Indonesian names, addresses, and birth dates generated using the id_ID locale.
2. prompt_dataset.csv (The Contextual Prompts): Features the PII embedded into three distinct linguistic styles:
- Formal: Standard business Indonesian.
- Code-Mixed: A blend of Indonesian and English common in professional tech environments.
- Slang/Noisy: Informal language featuring slang text, typos, and specific noise logic (OCR-simulated typos like 0 to O, dash-spacing, and partial masking).
