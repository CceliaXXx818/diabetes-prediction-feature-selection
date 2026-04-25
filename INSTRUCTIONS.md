# Assignment 1 - Instructions for Claude Code

## Student Info
- Name: ZHANG CHENCHUAN
- Matrix NO: P162614
- Course: TC6414 | Data Knowledge Discovery & Data Mining
- Semester: Sem II 2025/2026
- Lecturer: Assoc. Prof. Dr. Zalinda Othman

## Your Task
Generate two deliverables:
1. notebook.ipynb — complete runnable Jupyter Notebook
2. report.docx — full academic report

Read Assignment_1.docx for the full requirements.
Read reference_notebook.ipynb for reference on missing value
handling and Mutual Information code style only — do NOT copy
its overall structure.
Read all images in ref_images/ for report formatting reference
— layout, figure captions, header style, writing tone.

---

## PART 1: NOTEBOOK

### Datasets
- data/pima.csv
  Source: Pima Indians Diabetes Database (UCI/Kaggle)
  Rows: 768, Columns: 9
  Columns: Pregnancies, Glucose, BloodPressure, SkinThickness,
           Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome

- data/frankfurt.csv
  Source: Frankfurt Hospital Diabetes Dataset (GitHub)
  Rows: 2000, Columns: 9 (same columns as pima.csv)

- Target variable: Outcome (0 = No Diabetes, 1 = Diabetes)
- Task: Binary classification

### Notebook Structure (7 Cells)

Cell 1 — Setup & Data Loading
- Import all libraries (pandas, numpy, matplotlib, seaborn,
  sklearn)
- Load both CSVs with pd.read_csv()
- Print df.info() and df.head() for both datasets
- Print df.describe() for both datasets

Cell 2 — Section 3.1: Handling Missing Values
- Columns with 0 as missing: Glucose, BloodPressure,
  SkinThickness, Insulin, BMI
- Replace 0 with NaN in those columns for both datasets
- Print missing value counts before and after
- Generate and save fig_3_1_1_missing_heatmap.png
  (side-by-side heatmap for both datasets)
- Impute with median for each dataset separately

Cell 3 — Section 3.2: Data Integration
- Add 'source' column to each df before merging
  (df_pima['source']='PIMA', df_frankfurt['source']='Frankfurt')
- Concatenate with pd.concat(), reset_index
- Print final shape, source distribution, class distribution
- Drop 'source' column after printing

Cell 4 — Section 3.3 & 3.4: Transformation & Standardization
- Log transformation: apply np.log1p() to Insulin and
  DiabetesPedigreeFunction
- Generate and save fig_3_3_1_log_transform.png
  (2x2 subplots: before/after for each column)
- Apply StandardScaler to all 8 features
- Generate and save fig_3_4_1_standardization.png
  (boxplot before vs after, side by side)
- Print descriptive stats after scaling

Cell 5 — Section 3.5: PCA
- Fit PCA on scaled features
- Print explained variance ratio per component
- Print number of components needed for 95% variance
- Generate and save fig_3_5_1_pca_variance.png
  (bar chart of individual variance + line of cumulative,
   add horizontal dashed line at 95%)

Cell 6 — Section 4.0: Feature Selection
- Mutual Information:
  - Use mutual_info_classif on scaled features
  - Create sorted DataFrame of feature vs MI score
  - Print the full table
  - Select features above median MI score
  - Print selected features
  - Generate and save fig_4_2_1_mi_barplot.png
    (horizontal bar chart, sorted descending)

- RFE:
  - Use RFE(LogisticRegression(max_iter=1000), n_features_to_select=5)
  - Print support and ranking for all features
  - Print selected features list
  - Generate and save fig_4_3_1_rfe_ranking.png
    (horizontal bar chart of ranking, highlight selected in
     green, rejected in red)

Cell 7 — Section 5.0: Model Performance & Save Output
- Use LogisticRegression(max_iter=1000, random_state=42)
- Evaluate with cross_val_score (cv=5) for all three:
  a. All features (baseline)
  b. MI-selected features
  c. RFE-selected features
- Metrics: Accuracy, Precision, Recall, F1
- Print results table
- Generate and save fig_5_1_performance_comparison.png
  (grouped bar chart, 3 groups x 4 metrics)
- Save preprocessed scaled data to
  output/preprocessed_dataset.csv

### Output Files
output/preprocessed_dataset.csv
output/figures/fig_3_1_1_missing_heatmap.png
output/figures/fig_3_3_1_log_transform.png
output/figures/fig_3_4_1_standardization.png
output/figures/fig_3_5_1_pca_variance.png
output/figures/fig_4_2_1_mi_barplot.png
output/figures/fig_4_3_1_rfe_ranking.png
output/figures/fig_5_1_performance_comparison.png

---

## PART 2: REPORT (report.docx)

### Page Setup
- Font: Times New Roman 12pt
- Line spacing: 1.5
- Margins: 1 inch all sides
- Page numbers: bottom center
- No more than 15 pages

### Cover Page (no header, no page number)
- University: Universiti Kebangsaan Malaysia
- Logo: text only is fine
- Course: TC6414 | Data Knowledge Discovery & Data Mining
- Assignment: Assignment 1 | Data Preprocessing and
  Feature Selection
- Lecturer: Assoc. Prof. Dr. Zalinda Othman
- Student Name: ZHANG CHENCHUAN
- Matrix NO: P162614
- Date: April 2026

### Page Header (every page except cover, right-aligned)
TC6414 | DATA PREPROCESSING AND FEATURE SELECTION |
ASSOC. PROF. DR. ZALINDA OTHMAN

### Figure Caption Format
Figure X.X.X   [Description]
(bold number, normal description text, placed below figure,
centered)

### Section Heading Format
- Level 1: "1.0 INTRODUCTION" — bold, uppercase
- Level 2: "1.1 Background and Objectives" — bold, title case

---

### Report Content

1.0 INTRODUCTION
1.1 Background and Objectives
Write 2 paragraphs:
- Para 1: Diabetes as a global health issue, importance of
  early prediction using ML (cite Olisah et al., 2022 and
  Tasin et al., 2023)
- Para 2: Objective of this study — apply preprocessing and
  feature selection on two diabetes datasets (PIMA and
  Frankfurt) to prepare data for Logistic Regression
  classification (cite Chang et al., 2023)

2.0 DATA ACQUISITION
2.1 Dataset Overview
- Describe Dataset A (PIMA): source, rows, columns, purpose
- Show df.info() output as a code block or table — label as
  Figure 2.1.1 Mental health information displayed about
  features or attributes utilized in the dataset
  (follow ref_images style exactly)
- Describe Dataset B (Frankfurt): source, rows, columns
- Show df.info() output — label as Figure 2.1.2
- Note: 0 values in medical columns represent missing data

2.2 Tools and Libraries Used
- Python via Google Colab
- Libraries: Pandas (data manipulation), NumPy (numerical
  operations), Scikit-learn (preprocessing, feature
  selection, modelling), Matplotlib & Seaborn (visualization)

3.0 DATA PREPARATION
Write 1 intro paragraph citing Li et al. 2023 style
(see ref_images) explaining what data preparation involves.

3.1 Handling Missing Values
- Explain 0-as-missing in medical datasets
- Insert Figure 3.1.1 (missing heatmap)
- Explain median imputation choice and why (robust to
  outliers vs mean)
- Show before/after missing value counts

3.2 Data Integration
- Explain why two datasets were combined (diversity of
  population — US vs Germany)
- Explain pd.concat approach
- State final dataset size after merge
- Note class distribution

3.3 Data Transformation
- Explain log1p transformation purpose (reduce skewness)
- State which columns were transformed: Insulin,
  DiabetesPedigreeFunction
- Insert Figure 3.3.1 (before/after histograms)

3.4 Data Standardization
- Explain StandardScaler (Z-score normalization)
- Explain why needed (features have different scales)
- Insert Figure 3.4.1 (boxplot before/after)

3.5 Dimensionality Reduction
- Explain PCA purpose
- Insert Figure 3.5.1 (variance chart)
- State how many components explain 95% variance
  (fill in actual number from notebook output)
- Note: PCA used for analysis; original scaled features
  retained for feature selection step

4.0 FEATURE SELECTION
4.1 Feature Selection Overview
- Explain purpose: identify most relevant features,
  reduce noise, improve model performance
- State two methods used: MI and RFE (cite Olisah et al.,
  2022)

4.2 Mutual Information
- Explain MI concept (measures dependency between feature
  and target)
- Show MI scores table (Figure 4.2.1.1)
- Insert Figure 4.2.1 bar chart (Figure 4.2.1.2 — follow
  ref_images numbering style)
- State which features were selected and why

4.3 Recursive Feature Elimination (RFE)
- Explain RFE concept (iteratively removes least important
  features using base estimator)
- State base estimator: Logistic Regression
- State n_features_to_select = 5
- Show ranking table
- Insert Figure 4.3.1 ranking chart
- State which 5 features were selected

4.4 Comparison of MI and RFE
- Table comparing: which features each method selected,
  overlap between both methods
- Brief discussion on agreement/disagreement

5.0 MODEL PERFORMANCE ANALYSIS
5.1 Baseline Model
- All 8 features, Logistic Regression, 5-fold CV
- State Accuracy, Precision, Recall, F1
  (fill in actual values from notebook)

5.2 After MI Feature Selection
- State features used
- State Accuracy, Precision, Recall, F1

5.3 After RFE Feature Selection
- State features used
- State Accuracy, Precision, Recall, F1

5.4 Performance Comparison
- Insert Figure 5.1 (grouped bar chart)
- Summary table with all three scenarios and 4 metrics
- Discussion: did feature selection improve performance?
  Which method performed better?

6.0 CONCLUSION
6.1 Summary of Findings
- Summarize preprocessing steps done and their impact
- Summarize feature selection results
- State which method (MI or RFE) gave better results
  based on actual numbers

6.2 Limitations and Future Work
- Limitation: dataset is limited to female patients (PIMA)
- Limitation: Logistic Regression is a simple model
- Future: try more complex models (XGBoost, Random Forest)
- Future: try other feature selection methods

REFERENCES
Olisah, C. C., Smith, L., & Smith, M. (2022). Diabetes
mellitus prediction and diagnosis from a data preprocessing
and machine learning perspective. Computer Methods and
Programs in Biomedicine, 220, 106773.
https://doi.org/10.1016/j.cmpb.2022.106773

Chang, V., Bailey, J., Xu, Q. A., & Sun, Z. (2023). Pima
Indians diabetes mellitus classification based on machine
learning (ML) algorithms. Neural Computing and Applications,
35(22), 16157–16173.
https://doi.org/10.1007/s00521-022-07049-z

Tasin, I., Nabil, T. U., Islam, S., & Khan, R. (2023).
Diabetes prediction using machine learning and explainable
AI techniques. Healthcare Technology Letters, 10(1–2), 1–10.
https://doi.org/10.1049/htl2.12039

Ahmed, A., Aziz, S., & Qidwai, U. (2025). Machine Learning
Algorithm-Based Prediction of Diabetes Among Female
Population Using PIMA Dataset. Healthcare, 13(1), 37.
https://doi.org/10.3390/healthcare13010037

Reza, M. S., Amin, R., Yasmin, R., Kulsum, W., & Ruhi, S.
(2024). Improving diabetes disease patients classification
using stacking ensemble method with PIMA and local
healthcare data. Heliyon, 10(2), e24536.
https://doi.org/10.1016/j.heliyon.2024.e24536

Iparraguirre-Villanueva, O., Espinola-Linares, K.,
Castañeda, R. O. F., & Cabanillas-Carbonell, M. (2023).
Application of Machine Learning Models for Early Detection
and Accurate Classification of Type 2 Diabetes.
Diagnostics, 13(14), 2383.
https://doi.org/10.3390/diagnostics13142383