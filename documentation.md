# Pipeline

Below is a detailed **pipeline explanation** of how the `train_knn.py` and `evaluation.py` scripts work, what steps are taken, and the reasons behind each step.

---

## **Pipeline: Factuality Detection Using KNN Classifier**

The pipeline consists of **two main stages**:
1. **Training a KNN Classifier** (`train_knn.py`): Builds the classifier to identify non-factual hallucinations in generated summaries.
2. **Evaluating Factuality** (`evaluation.py`): Uses the trained KNN model to evaluate new summaries and determine the percentage of non-factual entities.

---

### **1. Train KNN Classifier (`train_knn.py`)**

#### **Objective**:  
Train a KNN (K-Nearest Neighbors) classifier to detect **non-factual hallucinations** in generated summaries at the entity level.

---

### **Steps in `train_knn.py`**

1. **Load Training Data**:
   - Training data (`train.json`) is loaded.
   - Each entry includes:
     - **source**: Original document.
     - **prediction**: Generated summary.
     - **entities**: Annotated entities with labels:
       - `Non-hallucinated`: Entity appears in the source text.
       - `Factual Hallucination`: Entity is not in the source but factually correct.
       - `Non-factual Hallucination`: Entity is not in the source and factually incorrect.

   ```python
   train_set = json.load(open(args.train_path, 'r'))
   ```

---

2. **Load Pretrained Models**:
   - **Prior Model**: A pretrained **Masked Language Model (MLM)** (e.g., BART) is loaded to compute **prior probabilities** of entities.
   - **Posterior Model**: A pretrained **Conditional Masked Language Model (CMLM)** is loaded to compute **posterior probabilities** given the source document.

   ```python
   prior_bart = BARTModel.from_pretrained(args.mlm_path, checkpoint_file='model.pt')
   bart = BARTModel.from_pretrained(args.cmlm_model_path, checkpoint_file='checkpoint_best.pt')
   ```

   **Why?**
   - **Prior Probability**: Represents the likelihood of an entity occurring **without context** (general world knowledge).
   - **Posterior Probability**: Represents the likelihood of an entity occurring **given the source document**.

---

3. **Feature Extraction**:
   For each entity in the training set:
   - Calculate the **prior probability** using the MLM model.
   - Calculate the **posterior probability** using the CMLM model.
   - Compute an **overlap feature**: A binary value indicating whether the entity appears in the source text.

   Example:
   ```python
   priors = get_probability_parallel(prior_model, inputs[0], inputs[1], ...)
   posteriors = get_probability_parallel(model, inputs[0], inputs[1], ...)
   overlaps = [1. if e['ent'].lower() in source.lower() else 0. for e in entities]
   ```

   - Each entity is represented as:
     ```
     [prior_probability, posterior_probability, overlap_feature]
     ```

   - Labels:
     - `0` for **Non-hallucinated** or **Factual Hallucination**.
     - `1` for **Non-factual Hallucination**.

   **Why?**
   - Combining these three features helps the KNN classifier distinguish between factual and non-factual entities.

---

4. **Train the KNN Classifier**:
   - The extracted features and labels are used to train a **K-Nearest Neighbors** (KNN) classifier.

   ```python
   classifier = neighbors.KNeighborsClassifier(n_neighbors=30)
   classifier.fit(features, labels)
   ```

   **Why KNN?**
   - KNN is simple, interpretable, and works well for low-dimensional feature spaces.
   - It classifies entities by comparing them to the nearest labeled examples in the training data.

---

5. **Save the Classifier**:
   - The trained KNN model is saved as a `.pkl` file for later use.

   ```python
   pickle.dump(classifier, open(save_path, 'wb'))
   ```

---

6. **Evaluate on Test Set (Optional)**:
   - If a test set (`test.json`) is provided, features are extracted for test entities.
   - The classifier predicts their labels, and accuracy/F1 scores are calculated.

---

### **2. Evaluate Summaries (`evaluation.py`)**

#### **Objective**:  
Use the trained KNN classifier to evaluate factuality in new summaries and identify non-factual entities.

---

### **Steps in `evaluation.py`**

1. **Load Trained KNN Classifier**:
   - The saved KNN model (`knn_classifier.pkl`) is loaded.

   ```python
   classifier = pickle.load(open(args.knn_model_path, 'rb'))
   ```

---

2. **Load Input Data**:
   - **source_path**: Path to source documents (`test.source`).
   - **target_path**: Path to generated summaries (`test.hypothesis`).

   Each file has one document/summary per line.

   ```python
   source = read_lines(args.source_path)
   hypothesis = read_lines(args.target_path)
   ```

---

3. **Feature Extraction**:
   For each summary:
   - Extract **entities** using a Named Entity Recognition (NER) tool (e.g., spaCy).
   - Compute features for each entity:
     - **Prior probability** using the MLM model.
     - **Posterior probability** using the CMLM model.
     - **Overlap feature** (binary).

   ```python
   pri_probs = get_probability_parallel(prior_model, ...)
   pos_probs = get_probability_parallel(model, ...)
   overlap = [1 if e.lower() in source_doc else 0 for e in entities]
   ```

   **Why?**
   - These features allow the classifier to determine whether an entity is factual or non-factual.

---

4. **Predict Factuality**:
   - The KNN classifier predicts the label (`Factual` or `Non-Factual`) for each entity based on extracted features.

   ```python
   Z = classifier.predict(features)
   ```

---

5. **Generate Results**:
   - Calculate the percentage of **non-factual entities** across all summaries.
   - Report the total number of entities and the proportion of non-factual ones.

   ```python
   print('- Total extracted entities: ', Z.shape[0])
   print('- Non-factual entities: {:.2f}%'.format((Z.sum() / Z.shape[0]) * 100))
   ```

---

### **Summary of Steps**

1. **Train KNN Classifier** (`train_knn.py`):
   - Load annotated training data.
   - Extract features for each entity: `prior`, `posterior`, `overlap`.
   - Train a KNN classifier to detect non-factual entities.

2. **Evaluate Summaries** (`evaluation.py`):
   - Load the trained KNN classifier.
   - Extract features for entities in new summaries.
   - Use the classifier to predict which entities are non-factual.
   - Report the percentage of non-factual entities.

---

### **Why This Works**:
- By leveraging:
  1. **Prior probability**: World knowledge about entities.
  2. **Posterior probability**: Document-specific relevance of entities.
  3. **Overlap feature**: Explicit appearance of entities in the source text.

The pipeline effectively distinguishes between:
- Non-hallucinated entities (appearing in the source text),
- Factual hallucinations (not in the source but correct), and
- Non-factual hallucinations (incorrect entities).

The KNN classifier is a simple yet effective method to identify **non-factual hallucinations**, improving the factual reliability of generated summaries. 🚀


# Paper Explanation

---

### **Title**: *Hallucinated but Factual! Inspecting the Factuality of Hallucinations in Abstractive Summarization*

---

### **What is the Paper About?**
The paper deals with an issue in **abstractive text summarization** models, where models sometimes generate:
1. **Hallucinated content**: Information that is **not directly present** in the original document.
   - This is often seen as a problem since the model is "hallucinating" details.
2. **Types of Hallucinations**:
   - **Factual hallucinations**: Hallucinated information that is factually **correct** (it may come from world knowledge, not the document).
   - **Non-factual hallucinations**: Hallucinated information that is factually **incorrect** or misleading.

The paper focuses on **identifying and analyzing hallucinated entities** and improving the factual quality of generated summaries.

---

### **Why Does This Matter?**
- **Abstractive summarization** aims to generate concise summaries of text using natural language. 
- Hallucinated entities are common in these summaries, and distinguishing between **factual** and **non-factual hallucinations** is critical.
- **Factual hallucinations** can actually be beneficial:
   - They add accurate external knowledge.
   - They make the summary more informative and complete.
- The challenge: Detect **non-factual hallucinations** while retaining useful factual ones.

---

### **Key Contributions of the Paper**:
1. **Entity-Level Factuality Classification**:
   - The authors develop a method to classify entities (people, places, events, etc.) in a summary as:
     - **Non-hallucinated**: Present in the source text.
     - **Hallucinated-Factual**: Not present in the text but factually correct.
     - **Hallucinated-Non-factual**: Not present and factually incorrect.

2. **Building a Human-Annotated Dataset**:
   - They annotate entities from summaries of the **XSUM dataset** at the entity level.
   - This dataset, called **XENT**, marks entities with their hallucination and factuality status.

3. **Classifier Using Prior and Posterior Probabilities**:
   - The authors use two probabilities to identify hallucinations:
     - **Prior Probability**: How likely an entity is to occur in general (using **Masked Language Models**, or MLM).
       - Example: "Barack Obama" has a high prior probability since it's common knowledge.
     - **Posterior Probability**: How likely the entity is to occur **given the source document**.
       - Example: If the document is about politics, "Barack Obama" may have a high posterior probability.
   - If the **posterior probability is low** but the **prior probability is high**, the entity is likely **hallucinated**.

4. **Improving Summarization with RL (Reinforcement Learning)**:
   - The classifier is used to reward the summarization model during training:
     - Factual summaries get higher rewards.
     - Non-factual hallucinations are penalized.

---

### **The Method in Detail**:
1. **Entity Extraction**:
   - The authors extract entities (names, places, etc.) from the generated summaries using tools like **spaCy**.

2. **Probability Features**:
   - They compute:
     - **Prior Probability**: Using a pretrained Masked Language Model (like BERT or BART).
     - **Posterior Probability**: Using a Conditional Masked Language Model (CMLM), which considers the source text.
   - These probabilities are compared to determine hallucination and factuality.

3. **Overlapping Features**:
   - To help identify non-hallucinated entities, the authors also check if the entity appears **verbatim** in the source document.

4. **KNN Classifier**:
   - A **K-Nearest Neighbors (KNN)** classifier is trained to predict whether an entity is:
     - Non-hallucinated,
     - Hallucinated-Factual, or
     - Hallucinated-Non-factual.

---

### **Results**:
1. **Analysis of Hallucinations**:
   - Around **30% of entities** in generated summaries are hallucinated.
   - Of these, over **half** are factually correct (factual hallucinations).

2. **Classifier Performance**:
   - The classifier performs well in distinguishing between factual and non-factual hallucinations.
   - It correlates strongly with **human annotations**.

3. **Improved Summaries**:
   - When the classifier is used as a reward in **Reinforcement Learning**:
     - **Factuality improves**: Summaries contain fewer non-factual hallucinations.
     - **Abstractiveness is preserved**: Summaries remain creative and concise, instead of being overly extractive.

---

### **Findings**:
1. Hallucinated entities are not always bad; **factual hallucinations** can add useful information.
2. Posterior probabilities (considering the source text) are effective at identifying **non-factual hallucinations**.
3. By incorporating factuality checks into the training process, summaries can become **more reliable** without sacrificing creativity.

---

### **Conclusion**:
The paper challenges the notion that hallucinations are inherently negative. By distinguishing between factual and non-factual hallucinations, the authors propose a way to:
- Retain useful hallucinations.
- Filter out harmful, non-factual ones.

Their approach improves the factual accuracy of summaries while maintaining the benefits of abstractive summarization.

---

### **Simple Takeaway**:
- Not all hallucinations in summarization models are bad.
- Some hallucinations (factual ones) can improve the summary by adding background knowledge.
- This method uses probabilities and classification to identify bad hallucinations and improve overall factuality.

