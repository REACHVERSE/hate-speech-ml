---
obsidianUIMode: preview
---

This report summarizes the performance of your hate speech detection model on a test dataset. Understanding these metrics helps us gauge how well our "robot" is learning to distinguish between hate speech and regular content.

## 1. Overall F1 Score

- **F1 Score:** `0.6613`
    

The F1 score is a single metric that balances the model's _precision_ (how many times it's correct when it predicts "hate speech") and its _recall_ (how many actual "hate speech" instances it successfully identifies). A perfect F1 score is 1.0, while 0.0 is the worst.

In isolation, 0.66 might seem moderate. However, the detailed Classification Report below reveals a critical issue that makes this score misleading.

## 2. Classification Report

```
              precision    recall  f1-score   support

           0       0.00      0.00      0.00      2530
           1       0.49      1.00      0.66      2470

    accuracy                           0.49      5000
   macro avg       0.25      0.50      0.33      5000
weighted avg       0.24      0.49      0.33      5000
```

### Understanding the Terms:

- **`precision`**: Out of all the times the model _predicted_ a certain class (e.g., "hate speech"), how many times was it actually correct?
    
- **`recall`**: Out of all the _actual_ instances of a certain class (e.g., actual "hate speech" tweets), how many did the model correctly identify?
    
- **`f1-score`**: The harmonic mean of precision and recall for that specific class. It provides a balanced measure.
    
- **`support`**: The number of actual examples for each class in your test dataset.
    
- **`accuracy`**: The overall percentage of correct predictions made by the model across all classes.
    
- **`macro avg`**: The unweighted average of precision, recall, and F1-score across all classes.
    
- **`weighted avg`**: The average of precision, recall, and F1-score across all classes, weighted by the number of samples in each class.
    

### Detailed Breakdown:

- **Class 0 (Not Hate Speech):**
    
    - **Precision: 0.00**
        
    - **Recall: 0.00**
        
    - **F1-score: 0.00**
        
    - **Support: 2530**
        
    - **Interpretation:** This is the most critical part. A precision and recall of 0.00 for Class 0 means that your model **never predicted any tweet as "Not Hate Speech."** Out of the 2530 actual "Not Hate Speech" tweets, it failed to identify any of them correctly. This also triggered the `UndefinedMetricWarning` messages, as it couldn't calculate precision because it never made a positive prediction for this class.
        
- **Class 1 (Hate Speech):**
    
    - **Precision: 0.49**
        
    - **Recall: 1.00**
        
    - **F1-score: 0.66**
        
    - **Support: 2470**
        
    - **Interpretation:**
        
        - **Recall of 1.00 (100%):** This means your model successfully identified _every single actual "hate speech" tweet_ in the dataset. This sounds impressive!
            
        - **Precision of 0.49 (49%):** However, when your model _predicted_ a tweet was "hate speech," it was only correct about half the time. This implies that many "Not Hate Speech" tweets were incorrectly flagged as "Hate Speech" (false positives).
            
- **Overall Accuracy: 0.49**
    
    - This confirms the problem: the model is only correct about half the time overall.
        

## 3. Conclusion

These results show that your model is essentially **always predicting "Hate Speech" (Class 1)**, regardless of the actual content.

- It correctly identifies all actual hate speech because it labels everything as hate speech.
    
- But it fails to identify any "Not Hate Speech" because it never predicts that class.
    
- This leads to a high recall for "Hate Speech" but a very poor precision (and 0.0 for all metrics on "Not Hate Speech").
    

**This is a strong indication that your model is not learning to discriminate between the two classes. It has a severe bias towards one class.**