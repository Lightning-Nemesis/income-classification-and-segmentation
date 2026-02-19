# income-classification-and-segmentation

A machine learning project built on the 1994–1995 US Census Bureau Current Population Survey data. The project addresses two objectives:

1. **Income Classifier** — Train and validate a binary classifier to predict whether an individual earns above or below $50,000 using 40 demographic and employment variables
2. **Segmentation Model** — Build an unsupervised customer segmentation model and demonstrate how resulting groups differ for retail marketing purposes

---

## Project Structure

```
income-classification-and-segmentation/
│
├── data/
│   ├── censusbureau.data              # Raw comma-delimited data file
│   └── census-bureau.columns          # Column names header file
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02__classification.ipynb
│   └── 03_customer_segmentation.ipynb
│
├── requirements.txt
├── README.md
├── ML-TakehhomeProject.pdf
└── Report.pdf
```

---

## Environment

- Python 3.8 or higher
- Jupyter Notebook or Google Colab
- These notebooks were originally developed on **Google Colab**. 

---

## Installation

Clone the repository and install all required packages:

```bash
git clone https://github.com/Lightning-Nemesis/income-classification-and-segmentation.git
cd income-classification-and-segmentation
pip install -r requirements.txt
```


## Data Setup

1. Place `censusbureau.data` and `census-bureau.columns` in the `data/` directory
2. The data file is comma-delimited with 40 demographic and employment variables, a sample weight column, and an income label per row
3. Do not rename the data files

Local paths:

```python
data_file_path = '../data/censusbureau.data'
columns_file_path = '../data/census-bureau.columns'
```

---

## Running the Notebooks

Launch Jupyter from the project root:

```bash
jupyter notebook
```

Run the notebooks in the following order:

---

### 1. EDA — `01_EDA.ipynb`

Exploratory data analysis of the census dataset before modeling.

**What it covers:**

- Loading and inspecting the raw data using the columns header file
- Identifying continuous columns (`age`, `wage per hour`, `capital gains`, `capital losses`, `dividends from stocks`, `num persons worked for employer`, `weeks worked in year`) and categorical columns
- Handling missing values — replacing `NaN` and `?` placeholders with `Unknown`
- Univariate analysis: weighted bar plots for categorical variables, distribution plots and custom weighted boxplots for continuous variables, weighted income label distribution
- Geographic analysis: choropleth map of population by state of previous residence using Plotly
- Migration pattern analysis across migration code variables
- Bivariate analysis: distribution plots split by income label for all categorical variables, violin plots for continuous variables by income label
- Multivariate analysis: correlation heatmap of continuous features, OHE-based correlation ranking against the income label to identify top predictors

**Expected output:** Visual understanding of data distributions, class imbalance (~94% below $50k weighted), and which features correlate most with high income

---

### 2. Classification — `02_classification.ipynb`

Trains and evaluates multiple classifiers to predict whether an individual earns above or below $50,000.

**What it covers:**

**Data Preprocessing:**
- Maps income label to binary (0 / 1)
- Fills `NaN` in `hispanic origin` with the pre-existing `Do not know` category
- Replaces `?` with `NaN` across all categorical columns
- Applies ordinal ranking to education (Children=1 through Doctorate=17)
- Stratified 80/20 train/test split with sample weights preserved

**Imputation Experiments:**
- Tests four imputation strategies — Mode, Extra Category (Unknown), KNN, and Random Forest imputation — using a baseline Random Forest to select the best approach
- Selected strategy: creating an `Unknown` extra category for all missing categoricals

**Feature Engineering — Weight of Evidence (WOE):**
- Calculates WOE and Information Value (IV) for all features
- Uses IV > 0.02 threshold to identify informative features
- Tests WOE-transformed features against Logistic Regression and Random Forest as a feature selection approach

**Feature Selection:**
- Compares RF feature importance ranking vs WOE-based feature selection
- RF importance selected as the better approach
- Top 25 features identified and used for final modeling

**Encoding:**
- OHE applied to low-cardinality categorical columns
- Label encoding applied to binary variables (sex, year)
- Ordinal target encoding (ranked by mean income) applied to high-cardinality occupation and industry codes
- Frequency encoding applied to high-cardinality geographic variables (country of birth, state of previous residence)
- VIF analysis performed to check multicollinearity

**Class Imbalance Handling:**
- SMOTE oversampling applied to training set to address the ~94/6 class split

**Models Trained and Compared:**
- Logistic Regression
- Random Forest
- SGD Classifier
- CatBoost
- XGBoost
- LightGBM

All models evaluated on Accuracy, F1 Score, and ROC-AUC with sample weights applied

**Hyperparameter Tuning:**
- Grid Search CV on LightGBM across depth, learning rate, estimators, subsample, num_leaves, and min_child_samples
- HyperOpt Bayesian optimization on LightGBM as an additional tuning approach
- Final best model: tuned LightGBM

**Fairness Evaluation:**
- Demographic parity ratio and difference across sex and race sensitive features
- Equalized odds ratio and difference across sex and race
- Uses the `fairlearn` library

**Explainability:**
- SHAP TreeExplainer with summary plot and individual force plots
- Decision Tree surrogate model (max depth 4) plotted for interpretability

**Expected output:** Trained LightGBM classifier with evaluation metrics, fairness metrics, SHAP feature importance, and model comparison bar charts

---

### 3. Segmentation — `03_customer_segmentation.ipynb`

Builds an unsupervised customer segmentation model and profiles segments for retail marketing.

**What it covers:**

**Preprocessing:**
- Fills missing categoricals with `Unknown`
- Applies the same encoding pipeline as the classifier notebook (OHE, frequency encoding, ordinal target encoding, label encoding)
- Engineers four derived features: `has_capital_income`, `is_working`, `hourly_earnings_proxy`, `foreign_born`
- Drops income label and weight from the feature matrix before clustering
- Applies `StandardScaler` to all features

**Dimensionality Reduction:**
- PCA retaining 85% of variance, reducing ~164 features to ~71 components
- Cumulative explained variance plot to visualize the 85% threshold
- Separate 3-component PCA purely for 3D visualization

**Cluster Selection:**
- KMeans++ evaluated across k=2 to k=10 with census sample weights, plotting inertia and silhouette scores
- Gaussian Mixture Model evaluated across k=2 to k=10 (without weights — not supported by sklearn), plotting BIC, AIC, and silhouette scores
- Final model selected: KMeans with k=6

**Final Model:**
- KMeans k=6 fitted on 71 PCA components with sample weights
- Segment labels attached back to original unencoded dataframe for profiling

**Visualizations:**
- Elbow and silhouette plots for KMeans
- BIC, AIC, and silhouette plots for GMM
- 3D PCA scatter plot — plain, colored by segment, and colored by income label
- Segment size and high income rate bar charts
- Age distribution histograms per segment
- Employment status stacked bar chart by segment
- Radar chart comparing all six segments across key continuous features
- Occupation breakdown horizontal bar charts for working segments

**Customer Profiling:**
- Continuous feature means per segment
- Top 3 category distributions for key categorical variables per segment
- Weighted population share per segment
- Country of birth, citizenship, veterans benefits, and labor union distributions per segment
- Income label rate per segment


## Data Source

US Census Bureau Current Population Surveys, 1994 and 1995. Contains 40 demographic and employment variables with population sample weights and binary income labels. Originally prepared for the UCI Machine Learning Repository.