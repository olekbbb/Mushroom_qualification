# Mushroom_qualification
Safety-First ML Pipeline for mushroom classification. Model optimized for Recall=1.0 (0% False Negatives) using custom GridSearchCV and SelectKBest logic in scikit-learn.

## Aim of the project
The main goal of this project is to build a machine learning model capable of classifying mushrooms as edible or poisonous (based on the UCI Mushroom dataset).
**Security:** In this specific problem, classification errors are not equal. Rejecting an edible mushroom (False Positive) is just a loss of a meal, but classifying a poisonous mushroom as edible (False Negative) is a fatal error. Therefore, the architecture of this model has been optimized **solely for maximizing Sensitivity (Recall = 1.0)**, while keeping the model structure as simple as possible.

## Technologies
* **Language:** Python
* **Libraries:** `scikit-learn`, `pandas`, `matplotlib`, `ucimlrepo`

## Highlights
1. **Data Leakage Protection**
   `SelectKBest` is encapsulated in a `Pipeline`, which ensures that the model selects the best features by learning only from the training data inside each cross-validation loop.
2. **Different logic in `GridSearchCV`**
   Instead of selecting the model with the highest average score, I implemented a `simplest_model` function. This returns the model that achieved perfect recall (100% poison detection), but among them, it selects the one that relies on the fewest features (k).       This minimizes tree complexity and prevents overfitting.
3. **Results reporting**
   The script doesn't just return a single final result. It generates a complete set of metrics (including TP, TN, FP, FN) for a varying number of features. This allows the table below to clearly show exactly how many variables the algorithm needs to avoid making    any critical errors (false negatives).

## Results
The impact of the number of selected features ($k$) on the model's performance on the test set was examined. The table below illustrates the point at which the algorithm stopped making critical errors (FN):
| Number of features (k) | Accuracy | Precision | Recall | TN | FP | FN | TP |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 5 | 0.9735 | 0.9660 | 0.9795 | 816 | 27 | 16 | 766 |
| 10 | 0.9742 | 0.9660 | 0.9808 | 816 | 27 | 15 | 767 |
| 15 | 0.9920 | 0.9836 | 1.0000 | 830 | 13 | 0 | 782 |
| 20 | 1.0000 | 1.0000 | 1.0000 | 843 | 0 | 0 | 782 |
| 91 | 1.0000 | 1.0000 | 1.0000 | 843 | 0 | 0 | 782 |
*(Note to the reader: From the table above, it is clear that adding further features above the optimal value does not improve security, but only unnecessarily complicates the model).*

### Final Result of the selected model:
* **Optimal number of features:** `15`
* **Test Recall:** `1.000`
* **False Negatives:** `0`

## Visualization

The code generates two key graphs (for analytical purposes):
1. **Feature Importances** - bar chart (horizontal) showing the most important features for the final model.
2. **Decision Tree Plot** - visualization of tree decision rules (limited to 3 levels of depth for readability).


