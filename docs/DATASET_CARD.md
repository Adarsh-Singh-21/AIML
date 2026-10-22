# Numerical dataset card

**Source:** [Akash Nath (Kaggle: akashnath29)](https://www.kaggle.com/datasets/akashnath29/lung-cancer-dataset), Kaggle version 2. Filename in archive: dataset.csv.

**Observed CSV:** 3,000 records, 15 input fields, target LUNG_CANCER (NO/YES). Raw target counts: 1,482 NO and 1,518 YES. Actual age range: 30–80. No vital-sign, treatment, scan, or survival columns occur in this CSV, despite broader language in the publisher description.

**Audit:** two duplicate rows removed; four records in two feature-identical but differently labelled groups quarantined. Such conflicts need not be data-entry errors: different people can have the same observed profile. Exclusion is a conservative prototype choice. Patient identifiers are unavailable, so duplicates also cannot be proved to refer to the same person. Usable records: 2,994; training 2,395; test 599. See exclusions.csv and prepared_records.csv.

**Encoding:** AGE retains years. F→0 and M→1 in GENDER. All other source feature codes map 1→0 and 2→1. Numeric values do not imply ordinal symptom severity. A complete validated clinical codebook is not supplied with the CSV; no extra semantics are invented. Target NO→0 and YES→1.

| Field | Representation |
|---|---|
| AGE | Whole years, 30–80 |
| GENDER | F=0; M=1 |
| SMOKING | Source 1→0; source 2→1 |
| YELLOW_FINGERS | Source 1→0; source 2→1 |
| ANXIETY | Source 1→0; source 2→1 |
| PEER_PRESSURE | Source 1→0; source 2→1 |
| CHRONIC_DISEASE | Source 1→0; source 2→1 |
| FATIGUE | Source 1→0; source 2→1 |
| ALLERGY | Source 1→0; source 2→1 |
| WHEEZING | Source 1→0; source 2→1 |
| ALCOHOL_CONSUMING | Source 1→0; source 2→1 |
| COUGHING | Source 1→0; source 2→1 |
| SHORTNESS_OF_BREATH | Source 1→0; source 2→1 |
| SWALLOWING_DIFFICULTY | Source 1→0; source 2→1 |
| CHEST_PAIN | Source 1→0; source 2→1 |

**Split and integrity:** duplicate/ambiguous profile groups handled before stratified seed-42 80/20 row split. Raw and prepared data hashes are checked before training. Preprocessing is fitted only on training folds. No synthetic records are added to this medical demonstration dataset.

**License label from Kaggle:** Database: Open Database, Contents: © Original Authors. Retain source attribution and applicable publisher terms; this is not the CC0 dataset initially considered.

**SHA-256:** `417052af60ea9da985478d93d6be49749d25c8d0168a14bbd9f94515ab80c6ce`.

**Limitations:** Clinical ground truth, individual patient identity, recruitment, and collection procedures are unverified. Source descriptions mention fields absent from the actual CSV, so only observed columns are used. Binary codes are preserved without assuming clinical severity or inventing a codebook. This is a retrospective dataset-label classifier, not a future-risk model, screening recommendation, or diagnosis. Scores are not calibrated individual cancer probabilities. No independent external or patient-level validation is established.
