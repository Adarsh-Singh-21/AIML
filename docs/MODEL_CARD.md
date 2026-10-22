# Numerical model card

**Task:** classify the selected Kaggle dataset's NO/YES labels from 15 numerical inputs. No image processing or CT model is used.

**Evaluated run:** numerical_v1. **Model:** hist_gradient_boosting; scikit-learn 1.9.1. The saved Pipeline includes StandardScaler and the trained estimator. All parameters are learned from the numerical training data; no pretrained predictor is used. Later active CV-only runs do not inherit this test result.

**Protocol:** 2,395 training records; five-fold stratified CV; selection by mean macro F1. CV macro F1 0.5236; CV ROC AUC 0.5309. The selected Pipeline is fitted to training records only. Training time 7.02 seconds on CPU. See saved metadata for exact parameters.

**Fixed test:** 599 records; accuracy 50.92%; balanced accuracy 50.99%; macro F1 0.5079; ROC AUC 0.5309. Majority-label baseline accuracy 50.58%. Accuracy Wilson 95% interval 46.92%–54.90%.

**Conclusion:** this evaluation is close to chance and does not demonstrate useful clinical prediction. The application shows this limitation prominently. It does not output low/medium/high medical risk bands or make claims about cancer stage or type.

**Artifact:** artifacts/numerical/numerical_v1/model.joblib. SHA-256 `b03b0498c128de3028234b7fbbed9abe227f0e2e26de97e823c9ebf5f5b4387c`. Only load trusted local artifacts; uploaded models are never accepted.

**Limitations:** Clinical ground truth, individual patient identity, recruitment, and collection procedures are unverified. Source descriptions mention fields absent from the actual CSV, so only observed columns are used. Binary codes are preserved without assuming clinical severity or inventing a codebook. This is a retrospective dataset-label classifier, not a future-risk model, screening recommendation, or diagnosis. Scores are not calibrated individual cancer probabilities. No independent external or patient-level validation is established.
