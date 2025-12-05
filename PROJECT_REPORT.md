**Introduction & Problem Definition**

This project implements an interactive machine-learning toolkit for classification tasks built with Streamlit. The application allows users to load datasets (either via upload or from bundled demo files), perform basic preprocessing (missing-value removal, one-hot encoding, and scaling), configure and train several classification models (Perceptron, Multilayer Perceptron (MLP), and Decision Tree), compare models, and visualize evaluation results including a metrics table and confusion matrices.

The main problem addressed is providing a compact, user-friendly environment where practitioners and learners can quickly prototype classification experiments, compare model performance, and inspect model internals without writing boilerplate code.

**Dataset Description**

The repository includes several demo datasets stored in the `data/` folder and accessible from the app sidebar. The included demo datasets are:

- `titanic.csv` — passenger survival dataset
- `diabetes.csv` — diabetes classification dataset
- `iris.csv` — species classification (multiclass)
- `mushroom.csv` — edible vs poisonous (categorical features)
- `wine_quality.csv` — wine quality regression converted to classification or grouped quality levels (depending on use)
- `customer_churn.csv` — churn classification dataset
- `breast_cancer.csv` — tumor diagnosis (malignant/benign)

Users may also upload their own CSV files through the UI. The app attempts to infer categorical and numerical columns automatically and asks the user to select the target column.

**GUI Design and Features**

The GUI is implemented in Streamlit and structured around a sidebar for configuration and a main content area for previewing data and viewing results. Key UI features:

- `Data Source` selector: choose between uploading a CSV or selecting a demo dataset.
- `Dataset Preview`: shows the first rows and shape of the loaded dataset.
- `Target Column` selector: required for supervised training.
- `Preprocessing` controls:
  - Checkbox to apply one-hot encoding (drop-first to avoid multicollinearity)
  - Normalization dropdown: `None`, `StandardScaler`, or `Min-Max Scaler`.
- `Model Configuration` area:
  - Execution mode: Single Model Training or Compare Models.
  - In Single mode: choose model and set hyperparameters (e.g., max depth for Decision Tree, hidden layers and activation for MLP, learning rate for Perceptron).
  - Train button(s): `Train Model` or `Train Models` to run comparison.
- Results area:
  - Metrics shown as 4 metric cards (Accuracy, Precision, Recall, F1).
  - Confusion matrix visualized with a Seaborn heatmap.
  - Model internals / debug attributes exposed in an expander (iter counts, weights, feature importances where available).
  - Comparison table + bar chart when running multiple models.

**Preprocessing Choices**

Preprocessing is intentionally minimal and interactive:

- Missing values: the pipeline uses `df.dropna()` to remove rows with any NA values before processing. This simplifies flows but should be revisited for datasets where imputation is preferable.
- Categorical encoding: optional one-hot encoding with `pd.get_dummies(..., drop_first=True)` to avoid dummy-variable trap.
- Scaling: user-selectable. Either `StandardScaler` (z-score) or `MinMaxScaler` (0–1) is applied to numerical columns chosen automatically from the dataset's dtypes. If `None` is selected, numeric features remain as-is.

These choices are exposed to the user, allowing rapid experimentation with their effect on model performance.

**Model Descriptions**

The app supports three classifier types (implemented using scikit-learn):

- Decision Tree (`sklearn.tree.DecisionTreeClassifier`): a non-parametric tree-based model. Default comparison configuration sets `max_depth=5` and `criterion='gini'`.
- Multilayer Perceptron / MLP (`sklearn.neural_network.MLPClassifier`): feed-forward neural network. Default settings include `hidden_layer_sizes=(100,)`, `max_iter=300`, and `activation='relu'`. The UI allows the user to parse custom hidden-layer tuples (e.g., `(100, 50)`).
- Perceptron (`sklearn.linear_model.Perceptron`): a simple linear classifier that supports online learning for binary (and multiclass via one-vs-rest) tasks. The UI exposes `eta0` (learning rate) and `max_iter`.

Model builders are available both for custom parameter construction and for quick default instances used in comparisons.

**Experimental Results**

The application computes predictions and standard evaluation metrics for either a single trained model or multiple models trained on the same split. Experiments are run on a user-selected train/test split (sidebar slider) with an explicit random seed for reproducibility.

Because this report is a static artifact and experiments are interactive, the concrete numeric results depend on dataset choice and UI parameters. Below is a template for reporting metrics and confusion matrices. To produce concrete results, run the app and click the training buttons for the dataset and configuration of interest.

How to run the app locally (PowerShell):

```powershell
cd "c:\Users\USER\Desktop\Workshop\Development Area\Projects\ml-toolset\ml-toolset"
.\.venv\Scripts\Activate.ps1
streamlit run main.py
```

When training a model, the UI displays metrics and confusion matrix immediately after training. The project code uses weighted averages for Precision, Recall, and F1 to support multiclass datasets.

**Metrics table**

Use this table to record results per experiment. Below is a template; fill with values produced by the app.

| Model | Accuracy | Precision (weighted) | Recall (weighted) | F1 Score (weighted) |
|---|---:|---:|---:|---:|
| Example: Decision Tree | 0.873 | 0.869 | 0.873 | 0.870 |
| MLP | - | - | - | - |
| Perceptron | - | - | - | - |

Notes:
- The app computes `Accuracy`, `Precision`, `Recall`, and `F1 Score` using scikit-learn functions with `average='weighted'` and `zero_division=0` to handle label imbalance and avoid division errors.

**Confusion matrices**

Confusion matrices are rendered as heatmaps by default. When reporting, save plots (e.g., via Streamlit's plot export or by adapting the code to write image files) and include the heatmap images for each trained model. Example guidance:

- For binary tasks (e.g., breast cancer): a 2×2 matrix with True Negative, False Positive, False Negative, True Positive counts.
- For multiclass tasks (e.g., iris): N×N matrix where rows correspond to true labels and columns to predicted labels.

The code in `ui_components.py` uses `seaborn.heatmap(..., annot=True, fmt='d')` for clear integer cell values and labels provided by `np.unique(y_processed)`.

**Analysis & Discussion**

- The toolkit is intended for quick iteration rather than large-scale benchmarking. Because it drops missing data by default and uses simple default hyperparameters, results can be biased (especially if missing data are informative or class imbalance is severe).
- For datasets with categorical features, one-hot encoding increases dimensionality; for high-cardinality categorical variables consider embedding approaches or target encoding before training to alleviate the curse of dimensionality.
- The MLP tends to need more careful hyperparameter tuning (learning rate schedules, larger max_iter, regularization) and might require feature scaling; the app exposes scaling options that materially affect MLP performance.
- Decision Trees are robust to unscaled features and categorical encodings but may overfit; the `max_depth` control in the UI helps mitigate that.
- Perceptron is fast but may underperform on nonlinear problems. It's useful as a baseline.

Suggested experiment flow:

1. Choose dataset and define the `Target Column` carefully.
2. Try both `StandardScaler` and `Min-Max Scaler` when using MLP; compare to `None`.
3. Run `Compare Models` mode with default hyperparameters to establish baselines.
4. For the best candidate model, use `Single Model Training` and tune hyperparameters in the UI.

**Conclusion & Future Work**

This project provides a compact, interactive environment to run classification experiments with minimal setup. It is especially useful for education and quick prototyping.

Planned or recommended future enhancements:

- Add imputation strategies (mean/median/mode, KNN imputation) instead of dropping NA rows.
- Add automated hyperparameter tuning (GridSearchCV or RandomizedSearchCV) and persist best models.
- Add cross-validation reporting (mean ± std of metrics) rather than a single train/test split.
- Save/Export functionality for trained models, metrics, and plots (e.g., download buttons or an `exports/` folder).
- Support for preprocessing pipelines and feature selection (variance threshold, mutual information, recursive feature elimination).
- Better handling for large datasets (streaming, sampling, progress indicators) and class imbalance strategies (SMOTE, class-weighted loss).

**References**

- Pandas — data ingestion and manipulation: https://pandas.pydata.org/
- scikit-learn — models, metrics, scalers: https://scikit-learn.org/
- Streamlit — UI for interactive data apps: https://streamlit.io/
- Seaborn / Matplotlib — plotting and heatmaps: https://seaborn.pydata.org/ , https://matplotlib.org/

---

Appendix: Key implementation notes and file references

- Data loading and demo datasets: `data_loader.py`, demo dataset paths in `config.py` (`DEMO_DATASETS`).
- Preprocessing (dropna, one-hot, scaling): `preprocessing.py` -> `process_data`.
- Model creation: `model_building.py` -> `build_single_model` and `default_compare_model`.
- Training and metrics calculation: `model_training.py` -> `train_single_model`, `train_comparison` (weighted precision/recall/f1 used).
- UI layout and controls: `main.py` for the Streamlit app and `ui_components.py` for plotting and metric display helpers.

If you would like, I can:

- Run the app with a selected dataset and capture actual metrics and confusion matrices to populate the report.
- Export this report to PDF.
- Adjust the report tone (short summary vs formal paper-style) or include specific experiment results.

Contact me with which option you prefer and which dataset/config you'd like me to run experiments for.
