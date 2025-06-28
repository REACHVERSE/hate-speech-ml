# Metrics Scoring Report for Quantization-Aware Trained (QAT) Model

This report summarizes the training, evaluation, and fairness metrics for a model prepared with Quantization-Aware Training (QAT), followed by pruning and quantization.

## 1. Training Performance

The model was trained for 3 epochs. The training loss consistently decreased, indicating effective learning.

|Epoch|Training Loss|
|---|---|
|1|0.6288|
|2|0.4052|
|3|0.2496|

## 2. Final Evaluation Metrics

The model's performance was evaluated using a classification report, providing precision, recall, and f1-score for each class, along with overall accuracy.

### Classification Report:

|Class|Precision|Recall|F1-Score|Support|
|---|---|---|---|---|
|Normal|0.92|0.92|0.92|252|
|Offensive|0.49|0.49|0.49|49|
|Hate|0.60|0.58|0.59|31|
|**Accuracy**|||**0.83**|**332**|
|Macro Avg|0.67|0.66|0.67|332|
|Weighted Avg|0.82|0.83|0.82|332|

## 3. Fairness Metrics

Fairness metrics were assessed for two protected attributes: `is_gender_topic` and `is_race_topic`.

### 3.1. Fairness Metrics for Protected Attribute: `is_gender_topic`

- **Statistical Parity Difference:** 0.1246
    
- **Disparate Impact:** 1.1763
    
- **Average Odds Difference:** 0.1622
    

### 3.2. Fairness Metrics for Protected Attribute: `is_race_topic`

- **Statistical Parity Difference:** -0.5795
    
- **Disparate Impact:** 0.2566
    
- **Average Odds Difference:** -0.0454
    

## 4. Pruning and Quantization

After training, pruning was applied, and the model was converted to a fully quantized model.

## 5. Model Export

The quantized and pruned model was successfully exported to ONNX format.

- **Exported Model Name:** `hate_speech_model_quantized_pruned.onnx`
    
- **Verification:** The ONNX model was loaded and verified successfully.
    

## 6. Test Prediction

A test tweet was used to demonstrate the model's prediction capability.

- **Test Tweet:** 'Women are weak and should not be leaders.'
    
- **Prediction:** Normal
    
- **Probabilities:** [0.34572056 (Normal), 0.32293898 (Offensive), 0.33134043 (Hate)]
    

This report concludes the evaluation of the QAT model, including its performance, fairness, and successful export for deployment.