# Numerical project demonstration

1. Start with run.ps1 and open http://localhost:8501.
2. State the task accurately: predicting NO/YES dataset labels from numerical coded inputs.
3. Show the dataset source, actual columns, encoding dictionary, exclusions, and fixed split.
4. Load example 1 in Single record. Explain that it is a training example selected by row hash, not correctness. Predict and download the JSON. Change an input to show the previous result clearing.
5. On the CSV batch page, download the example, upload it, choose numeric encoding, run prediction, and download the CSV. Show the separate original-source encoding option.
6. Open 4 · Evaluation. Show the Accuracy/Precision/Recall/F1 cards, highlighted metrics table, and grouped comparison chart. Explore confusion matrices, ROC, feature importance, class breakdown, and cross-validation. Use the sidebar Evaluation focus to switch between YES-class and macro-average metrics. Explain training-only CV selection and held-out evaluation. State the near-chance result and the majority-label baseline; do not present it as a useful cancer detector.
7. Show the notebook and retraining command. Subsequent runs do not repeatedly evaluate the viewed test split.
8. Discuss future work: verify source labels and clinical codebook, obtain an appropriate independent cohort, then evaluate calibration and clinical relevance. More models cannot substitute for informative, verified data.

Suggested slides: objective; requested dataset; encoding and audit; CV pipeline; model comparison; test and baseline; live numerical app; limitations and future work.
