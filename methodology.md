# Methodology: Social Media Usage and Mental Health Analysis

## 1. Research Design

### 1.1 Study Type
This study employs a **quantitative, cross-sectional, exploratory data mining approach** to investigate the relationship between social media usage patterns and mental health outcomes. The methodology integrates multiple analytical techniques to provide comprehensive insights from different analytical perspectives.

### 1.2 Research Questions
1. What patterns exist between demographic characteristics, social media usage behaviors, and mental health severity?
2. Can distinct user groups be identified based on their social media usage and mental health profiles?
3. How accurately can mental health risk levels be classified based on usage patterns and demographics?
4. What is the predictive relationship between usage behaviors and continuous mental health severity scores?

### 1.3 Analytical Framework
The study follows a four-stage analytical pipeline:
1. **Pattern Mining** - Discovery of association rules between behaviors and outcomes
2. **Clustering Analysis** - Identification of natural user segments
3. **Classification Analysis** - Binary risk level prediction
4. **Regression Analysis** - Continuous severity score prediction

## 2. Data Collection and Description

### 2.1 Dataset
- **Source**: Social Media and Mental Health (SMMH) survey dataset
- **Format**: CSV file (`smmh.csv`)
- **Data Type**: Self-reported survey responses
- **Collection Method**: Online survey questionnaire

### 2.2 Variables

#### 2.2.1 Demographic Variables
- **Age**: Continuous variable (years)
- **Gender**: Categorical (Male, Female, Non-binary)
- **Relationship Status**: Categorical (Single, In a relationship, Married, Divorced)
- **Occupation Status**: Categorical (University Student, School Student, Salaried Worker, Retired, etc.)
- **Affiliated Organizations**: Multi-select categorical (Company, Government, Private, School, University)

#### 2.2.2 Social Media Usage Variables
- **Use Social Media**: Binary (Yes/No)
- **Social Media Platforms**: Multi-select categorical (Facebook, Instagram, Twitter, Snapchat, TikTok, Discord, Pinterest, Reddit, YouTube, WhatsApp)
- **Daily Social Media Time**: Ordinal scale
  - Less than an Hour (0)
  - Between 1 and 2 hours (1)
  - Between 2 and 3 hours (2)
  - Between 3 and 4 hours (3)
  - Between 4 and 5 hours (4)
  - More than 5 hours (5)

#### 2.2.3 Behavioral Pattern Variables (Scale: 0-4)
- **Frequency of purposeless social media use**
- **Frequency of distraction by social media**
- **Restlessness without social media**
- **General distractibility level**
- **Worry level**
- **Difficulty concentrating**
- **Frequency of comparison to successful people**
- **Feelings about comparisons** (reverse-scored)
- **Frequency of seeking validation**

#### 2.2.4 Mental Health Indicator Variables (Scale: 0-4)
- **Frequency of feeling depressed**
- **Interest fluctuation in daily activities**
- **Sleep issues frequency**

### 2.3 Target Variable Construction

The **Mental Health Severity Score** is a composite metric calculated from four psychological dimensions:

#### ADHD-related Severity
$$\text{ADHD} = \frac{\text{purposeless\_use} + \text{distraction} + \text{distractibility} + \text{difficulty\_concentrating}}{4}$$

#### Anxiety Severity
$$\text{Anxiety} = \frac{\text{restlessness} + \text{worry\_level}}{2}$$

#### Self-Esteem Issues Severity
$$\text{Self-Esteem} = \frac{\text{comparison\_frequency} + \text{comparison\_feelings} + \text{validation\_seeking}}{3}$$

#### Depression Severity
$$\text{Depression} = \frac{\text{feeling\_depressed} + \text{interest\_fluctuation} + \text{sleep\_issues}}{3}$$

#### Overall Mental Health Severity
$$\text{Mental Health Severity} = \text{ADHD} + \text{Anxiety} + \text{Self-Esteem} + \text{Depression}$$

**Range**: 0 to 16 (higher scores indicate greater severity)

## 3. Data Preprocessing

### 3.1 Data Cleaning

#### 3.1.1 Column Renaming
- Original survey question text replaced with descriptive variable names
- Standardized naming convention applied for consistency
- Example: `"1. What is your age?"` → `"age"`

#### 3.1.2 Whitespace Removal
```python
# Remove leading/trailing whitespace from column names
smmh.columns = smmh.columns.str.strip()

# Remove whitespace from all string values
for col in smmh.select_dtypes(include=['object']).columns:
    smmh[col] = smmh[col].str.strip()
```

#### 3.1.3 Timestamp Removal
- Timestamp column removed as it does not contribute to analysis
- Maintains participant anonymity

#### 3.1.4 Missing Value Handling
```python
smmh = smmh.dropna()
```
- **Method**: Complete case analysis (listwise deletion)
- **Rationale**: Ensures data integrity for all analytical methods
- **Impact**: Recorded and reported in results

#### 3.1.5 Duplicate Removal
```python
smmh = smmh.drop_duplicates()
```
- Identifies and removes exact duplicate responses
- Prevents artificial inflation of patterns

### 3.2 Data Standardization

#### 3.2.1 Gender Standardization
```python
# Consolidate non-binary variations
smmh = smmh.replace({
    "gender": {
        "Nonbinary": "Non-binary",
        "NB": "Non-binary",
        "Non binary": "Non-binary"
    }
})

# Remove ambiguous entries
smmh = smmh[(smmh["gender"] != "There are others???") & 
            (smmh["gender"] != "unsure")]

# Final filter: Remove Non-binary due to small sample size
smmh = smmh[smmh['gender'] != 'Non-binary']
```

#### 3.2.2 Scale Normalization
All frequency and severity scales originally measured on 1-5 scale converted to 0-4 scale:
```python
# Example for all frequency variables
smmh['frequency_social_media_no_purpose'] = smmh['frequency_social_media_no_purpose'] - 1
```

**Special Case**: `feelings_about_comparisons` reverse-scored before normalization:
```python
# Original: 1=feel bad, 5=feel good
# Reversed: 4=feel bad, 0=feel good (for consistency with severity measures)
smmh['feelings_about_comparisons'] = 5 - smmh['feelings_about_comparisons'] - 1
```

#### 3.2.3 Occupation Status Consolidation
```python
smmh['occupation_status'] = smmh['occupation_status'].replace({
    'University Student': 'University',
    'School Student': 'School'
})
```

### 3.3 Feature Engineering

#### 3.3.1 One-Hot Encoding

**Social Media Platforms** (Multi-select):
```python
social_media_dummies = smmh['social_media_platforms'].str.get_dummies(sep=', ')
smmh = smmh.join(social_media_dummies)
smmh = smmh.drop(columns=['social_media_platforms'])
```
- Creates binary indicator for each platform
- Preserves information about multiple platform usage

**Affiliated Organizations**:
```python
affiliated_org_dummies = smmh['affiliated_organizations'].str.get_dummies(sep=', ')
if 'N/A' in affiliated_org_dummies.columns:
    affiliated_org_dummies = affiliated_org_dummies.drop(columns=['N/A'])
smmh = smmh.join(affiliated_org_dummies)
```

**Occupation Status**:
```python
occupation_status_dummies = smmh['occupation_status'].str.get_dummies()
smmh = smmh.join(occupation_status_dummies)
```

#### 3.3.2 Ordinal Encoding

**Gender**:
```python
gender_mapping = {"Male": 0, "Female": 1}
smmh['gender'] = smmh['gender'].map(gender_mapping)
```

**Relationship Status**:
```python
relationship_mapping = {
    "Married": 0,
    "Divorced": 1,
    "In a relationship": 2,
    "Single": 3
}
smmh['relationship_status'] = smmh['relationship_status'].map(relationship_mapping)
```

**Daily Social Media Time**:
```python
time_mapping = {
    'Less than an Hour': 0,
    'Between 1 and 2 hours': 1,
    'Between 2 and 3 hours': 2,
    'Between 3 and 4 hours': 3,
    'Between 4 and 5 hours': 4,
    'More than 5 hours': 5
}
smmh['daily_social_media_time'] = smmh['daily_social_media_time'].map(time_mapping)
```

**Binary Social Media Use**:
```python
smmh['use_social_media'] = smmh['use_social_media'].map({'Yes': 1, 'No': 0})
```

#### 3.3.3 Mental Health Severity Calculation

Individual component calculation followed by aggregation:
```python
adhd_severity = (
    smmh['frequency_social_media_no_purpose'] +
    smmh['frequency_social_media_distracted'] +
    smmh['distractibility_scale'] +
    smmh['difficulty_concentrating']
) / 4

anxiety_severity = (
    smmh['restless_without_social_media'] +
    smmh['worry_level_scale']
) / 2

self_esteem_severity = (
    smmh['compare_to_successful_people_scale'] +
    smmh['feelings_about_comparisons'] +
    smmh['frequency_seeking_validation']
) / 3

depression_severity = (
    smmh['frequency_feeling_depressed'] +
    smmh['interest_fluctuation_scale'] +
    smmh['sleep_issues_scale']
) / 3

smmh['mental_health_severity'] = (
    adhd_severity +
    anxiety_severity +
    self_esteem_severity +
    depression_severity
)
```

After calculation, individual components dropped to prevent data leakage:
```python
smmh = smmh.drop(columns=[
    'frequency_social_media_no_purpose',
    'frequency_social_media_distracted',
    'distractibility_scale',
    'difficulty_concentrating',
    'restless_without_social_media',
    'worry_level_scale',
    'compare_to_successful_people_scale',
    'feelings_about_comparisons',
    'frequency_seeking_validation',
    'frequency_feeling_depressed',
    'interest_fluctuation_scale',
    'sleep_issues_scale'
])
```

### 3.4 Feature Scaling

#### 3.4.1 Standardization
StandardScaler applied to numerical features for clustering and supervised learning:
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
numerical_columns = ['age', 'daily_social_media_time']
smmh[numerical_columns] = scaler.fit_transform(smmh[numerical_columns])
```

**Transformation**:
$$z = \frac{x - \mu}{\sigma}$$

Where:
- $z$ = standardized value
- $x$ = original value
- $\mu$ = mean
- $\sigma$ = standard deviation

**Rationale**: 
- Ensures all features on comparable scale
- Required for distance-based algorithms (K-Means, DBSCAN, SVM)
- Improves convergence in gradient-based optimization

### 3.5 Feature Selection

#### 3.5.1 Correlation-Based Selection
```python
# Calculate correlations with target
mh_correlations = corr_matrix["mental_health_severity"].drop("mental_health_severity")

# Select features with |correlation| > 0.15
selected_features = mh_correlations[
    (mh_correlations.abs() > 0.15)
].index.tolist()

smmh = smmh[selected_features + ['mental_health_severity']]
```

**Threshold Justification**: 
- Balances feature informativeness with model parsimony
- Removes weakly correlated noise features
- Prevents multicollinearity issues

## 4. Exploratory Data Analysis

### 4.1 Univariate Analysis

#### 4.1.1 Age Distribution
```python
plt.hist(smmh['age'], bins=20, edgecolor='k')
```
- Histogram visualization
- Calculate mean age
- Identify age range and distribution shape

#### 4.1.2 Categorical Variable Distributions
For each categorical variable (gender, relationship status, occupation status):
```python
variable.value_counts().plot(kind='bar')
```
- Frequency distribution
- Identify class imbalances
- Assess representation across categories

#### 4.1.3 Social Media Platform Usage
```python
platforms = smmh['social_media_platforms'].str.split(', ', expand=True).stack()
platform_counts = platforms.value_counts()
```
- Multi-select response analysis
- Platform popularity ranking
- Usage frequency distribution

#### 4.1.4 Daily Usage Time Distribution
```python
usage_distribution = smmh['daily_social_media_time'].value_counts()
```
- Ordinal distribution analysis
- Identification of modal usage category
- Assessment of heavy vs. light users

### 4.2 Bivariate Analysis

#### 4.2.1 Correlation Matrix
```python
corr_matrix = smmh.corr()
sns.heatmap(corr_matrix, cmap="coolwarm", center=0, square=True)
```
- Pearson correlation coefficients
- Identification of linear relationships
- Visualization of correlation patterns

#### 4.2.2 Feature-Target Correlations
```python
mh_correlations = corr_matrix["mental_health_severity"].drop("mental_health_severity")
mh_correlations.sort_values(ascending=False)
```
- Focused analysis on predictive features
- Ranking of feature importance
- Identification of positive/negative relationships

#### 4.2.3 Platform-Specific Correlations
```python
platform_columns = ["Facebook", "Instagram", "Twitter", ...]
platform_corrs = smmh[platform_columns + ["mental_health_severity"]].corr()
platform_correlations = platform_corrs["mental_health_severity"][platform_columns]
```

#### 4.2.4 Occupation-Specific Correlations
```python
occupation_columns = ["Company", "Government", "Private", "School", "University", ...]
occupation_corrs = smmh[occupation_columns + ["mental_health_severity"]].corr()
```

### 4.3 Class Balance Assessment
```python
for col in smmh.columns:
    counts = smmh[col].value_counts()
    if len(counts) > 2:
        counts.plot(kind='bar', title=f'Class Distribution for {col}')
```
- Identifies imbalanced features
- Informs need for stratified sampling
- Guides choice of evaluation metrics

## 5. Pattern Mining Analysis

### 5.1 Data Preparation for Pattern Mining

#### 5.1.1 Fresh Dataset Loading
```python
smmh_pattern = pd.read_csv('smmh.csv')
smmh_pattern = smmh_pattern.rename(columns=new_column_names)
```
- Separate copy created to preserve raw values
- Allows categorical pattern discovery without numerical encoding

#### 5.1.2 Categorical Feature Creation

**Age Groups**:
```python
smmh_pattern['age_group'] = pd.cut(smmh_pattern['age'], 
                                     bins=[0, 20, 30, 40, 100], 
                                     labels=['Under 20', '20-30', '30-40', '40+'])
```

**Usage Categories**:
```python
smmh_pattern['usage_category'] = pd.cut(smmh_pattern['daily_social_media_time'],
                                         bins=[-1, 1, 3, 5],
                                         labels=['Low (0-1hr)', 'Medium (2-3hr)', 'High (4-5hr+)'])
```

**Mental Health Categories** (based on tertiles):
```python
mh_percentiles = smmh_pattern['mental_health_severity'].quantile([0.33, 0.67])
smmh_pattern['mental_health_category'] = pd.cut(
    smmh_pattern['mental_health_severity'],
    bins=[-1, mh_percentiles.iloc[0], mh_percentiles.iloc[1], 100],
    labels=['Low', 'Medium', 'High']
)
```

### 5.2 Transaction Matrix Construction

#### 5.2.1 One-Hot Transaction Encoding
```python
transactions_df = pd.DataFrame()

for idx, row in smmh_pattern.iterrows():
    trans_row = {}
    
    # Demographics
    trans_row[f"gender_{row['gender']}"] = True
    trans_row[f"relationship_{row['relationship_status']}"] = True
    trans_row[f"occupation_{row['occupation_status']}"] = True
    trans_row[f"age_{row['age_group']}"] = True
    
    # Usage
    trans_row[f"usage_{row['usage_category']}"] = True
    
    # Mental health
    trans_row[f"mental_health_{row['mental_health_category']}"] = True
    
    # Platforms
    if pd.notna(row['social_media_platforms']):
        platforms = row['social_media_platforms'].split(', ')
        for platform in platforms:
            trans_row[f"platform_{platform}"] = True
    
    transactions_df = pd.concat([transactions_df, pd.DataFrame([trans_row])], 
                                 ignore_index=True)

transactions_df = transactions_df.fillna(False)
```

### 5.3 Apriori Algorithm Implementation

#### 5.3.1 Frequent Itemset Mining
```python
from mlxtend.frequent_patterns import apriori

min_support = 0.1  # 10% minimum support threshold

frequent_itemsets = apriori(transactions_df, 
                             min_support=min_support, 
                             use_colnames=True)
```

**Parameters**:
- **min_support**: 0.1 (appears in at least 10% of transactions)
- **use_colnames**: True (returns readable column names)

#### 5.3.2 Association Rule Generation
```python
from mlxtend.frequent_patterns import association_rules

min_confidence = 0.5  # 50% minimum confidence threshold

rules = association_rules(frequent_itemsets, 
                          metric="confidence", 
                          min_threshold=min_confidence)
```

**Metrics Calculated**:
- **Support**: Frequency of pattern
- **Confidence**: Conditional probability
- **Lift**: Strength of association
- **Conviction**: Degree of implication

### 5.4 Pattern Analysis Procedures

#### 5.4.1 Mental Health Outcome Rules
```python
# Filter rules predicting mental health outcomes
mh_rules = rules[rules['consequents'].apply(
    lambda x: any('mental_health' in str(item) for item in x)
)]

# Sort by lift for strongest associations
mh_rules_sorted = mh_rules.sort_values('lift', ascending=False)
```

#### 5.4.2 High Severity Risk Factor Identification
```python
high_mh_rules = rules[rules['consequents'].apply(
    lambda x: 'mental_health_High' in str(x)
)]
high_mh_sorted = high_mh_rules.sort_values('lift', ascending=False)
```

#### 5.4.3 Low Severity Protective Factor Identification
```python
low_mh_rules = rules[rules['consequents'].apply(
    lambda x: 'mental_health_Low' in str(x)
)]
low_mh_sorted = low_mh_rules.sort_values('lift', ascending=False)
```

#### 5.4.4 Usage Pattern Analysis
```python
# High usage patterns
high_usage_rules = rules[rules['antecedents'].apply(
    lambda x: 'usage_High' in str(x)
)]

# Low usage patterns
low_usage_rules = rules[rules['antecedents'].apply(
    lambda x: 'usage_Low' in str(x)
)]
```

#### 5.4.5 Demographic Pattern Analysis
```python
# Relationship status patterns
relationship_rules = rules[rules['antecedents'].apply(
    lambda x: any('relationship_' in str(item) for item in x)
)]

# Age group patterns
age_rules = rules[rules['antecedents'].apply(
    lambda x: any('age_' in str(item) for item in x)
)]
```

### 5.5 Pattern Visualization
```python
# Top patterns for high mental health severity
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

top_high = high_mh_sorted.head(8)
rule_labels = [', '.join([str(x) for x in row['antecedents']]) 
               for _, row in top_high.iterrows()]

axes[0].barh(range(len(top_high)), top_high['lift'], color='crimson')
axes[0].set_yticklabels(rule_labels)
axes[0].set_title('Top Patterns Predicting HIGH Mental Health Severity')
```

## 6. Clustering Analysis

### 6.1 Data Preparation for Clustering

#### 6.1.1 Feature Matrix Construction
```python
# Use preprocessed data, exclude target variable
X_cluster = smmh.drop(columns=['mental_health_severity']).copy()
```

#### 6.1.2 Feature Standardization
```python
scaler_clustering = StandardScaler()
X_cluster_scaled = scaler_clustering.fit_transform(X_cluster)
```
- Separate scaler instance to avoid data leakage
- All features scaled to zero mean, unit variance

### 6.2 Optimal Cluster Number Selection

#### 6.2.1 Elbow Method
```python
inertias = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_cluster_scaled)
    inertias.append(kmeans.inertia_)

# Plot inertia vs k
plt.plot(k_range, inertias, marker='o')
```

**Inertia** (Within-cluster sum of squares):
$$\text{Inertia} = \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2$$

#### 6.2.2 Silhouette Analysis
```python
silhouette_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_cluster_scaled)
    silhouette_scores.append(silhouette_score(X_cluster_scaled, labels))

# Identify optimal k
optimal_k = k_range[np.argmax(silhouette_scores)]
```

### 6.3 K-Means Clustering

#### 6.3.1 Model Training
```python
from sklearn.cluster import KMeans

optimal_k = 2  # Based on elbow and silhouette analysis
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_cluster_scaled)

smmh['kmeans_cluster'] = kmeans_labels
```

**Parameters**:
- `n_clusters`: 2 (determined by optimization)
- `random_state`: 42 (reproducibility)
- `n_init`: 10 (number of initialization attempts)
- `algorithm`: 'lloyd' (default, standard k-means)

#### 6.3.2 Cluster Evaluation
```python
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

kmeans_silhouette = silhouette_score(X_cluster_scaled, kmeans_labels)
kmeans_davies_bouldin = davies_bouldin_score(X_cluster_scaled, kmeans_labels)
kmeans_calinski_harabasz = calinski_harabasz_score(X_cluster_scaled, kmeans_labels)
```

#### 6.3.3 Cluster Characterization

**Original Value Reconstruction**:
```python
# Inverse transform to get original numerical values
original_values = scaler_preprocessing.inverse_transform(
    smmh[['age', 'daily_social_media_time']]
)
smmh['age_original'] = original_values[:, 0]
smmh['daily_social_media_time_original'] = original_values[:, 1]

# Map back to categorical time labels
smmh['daily_time_label'] = smmh['daily_social_media_time_original'].map(
    reverse_time_mapping
)
```

**Cluster Summary Statistics**:
```python
cluster_summary = smmh.groupby('kmeans_cluster').agg({
    'mental_health_severity': 'mean',
    'daily_social_media_time_original': 'mean',
    'age_original': 'mean'
})
```

**Platform/Occupation/Relationship Analysis**:
```python
for cluster in range(optimal_k):
    cluster_data = smmh[smmh['kmeans_cluster'] == cluster]
    
    # Top platforms
    for platform in platform_columns:
        usage_count = cluster_data[platform].sum()
        percentage = (usage_count / len(cluster_data)) * 100
    
    # Top occupations
    # Top relationships
```

### 6.4 DBSCAN Clustering

#### 6.4.1 Model Training
```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=3.5, min_samples=10)
dbscan_labels = dbscan.fit_predict(X_cluster_scaled)

smmh['dbscan_cluster'] = dbscan_labels
```

**Parameters**:
- `eps`: 3.5 (maximum distance for neighborhood)
- `min_samples`: 10 (minimum points to form dense region)
- Selected through iterative experimentation

#### 6.4.2 Noise Point Analysis
```python
n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)
```

#### 6.4.3 Evaluation (excluding noise)
```python
mask = dbscan_labels != -1
if sum(mask) > 0:
    dbscan_silhouette = silhouette_score(X_cluster_scaled[mask], 
                                          dbscan_labels[mask])
    dbscan_davies_bouldin = davies_bouldin_score(X_cluster_scaled[mask], 
                                                   dbscan_labels[mask])
    dbscan_calinski_harabasz = calinski_harabasz_score(X_cluster_scaled[mask], 
                                                         dbscan_labels[mask])
```

### 6.5 Hierarchical Agglomerative Clustering

#### 6.5.1 Model Training
```python
from sklearn.cluster import AgglomerativeClustering

hierarchical = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
hierarchical_labels = hierarchical.fit_predict(X_cluster_scaled)

smmh['hierarchical_cluster'] = hierarchical_labels
```

**Parameters**:
- `n_clusters`: 2 (same as K-Means for comparison)
- `linkage`: 'ward' (minimizes within-cluster variance)
- `affinity`: 'euclidean' (default distance metric)

#### 6.5.2 Dendrogram Visualization
```python
from scipy.cluster.hierarchy import dendrogram, linkage

# Sample for visualization clarity
sample_indices = np.random.choice(len(X_cluster_scaled), 
                                   size=min(100, len(X_cluster_scaled)), 
                                   replace=False)
X_sample = X_cluster_scaled[sample_indices]

linkage_matrix = linkage(X_sample, method='ward')
dendrogram(linkage_matrix)
```

#### 6.5.3 Cluster Analysis
Similar characterization process as K-Means:
- Summary statistics by cluster
- Platform/occupation/relationship distributions
- Original value interpretation

### 6.6 Dimensionality Reduction for Visualization

#### 6.6.1 PCA Application
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_cluster_scaled)
```

**Parameters**:
- `n_components`: 2 (for 2D visualization)
- `random_state`: 42 (reproducibility)

#### 6.6.2 Variance Explained
```python
explained_variance = pca.explained_variance_ratio_
total_variance = explained_variance.sum()
```

#### 6.6.3 Cluster Visualization
```python
plt.scatter(X_pca[:, 0], X_pca[:, 1], 
            c=cluster_labels, 
            cmap='viridis', 
            s=50, 
            alpha=0.6)

# For K-Means, plot centroids
centers_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], 
            c='red', marker='X', s=300)
```

### 6.7 Clustering Method Comparison

#### 6.7.1 Metrics Comparison Table
```python
clustering_results = pd.DataFrame({
    'Method': ['K-Means', 'Hierarchical', 'DBSCAN'],
    'Silhouette Score': [kmeans_silhouette, hierarchical_silhouette, dbscan_silhouette],
    'Davies-Bouldin Index': [kmeans_davies_bouldin, hierarchical_davies_bouldin, dbscan_davies_bouldin],
    'Calinski-Harabasz Score': [kmeans_calinski_harabasz, hierarchical_calinski_harabasz, dbscan_calinski_harabasz]
})
```

#### 6.7.2 Visual Comparison
Side-by-side PCA scatter plots for all three methods

#### 6.7.3 Cluster Profile Comparison
```python
# Compare characteristics across methods
fig, axes = plt.subplots(3, 2, figsize=(16, 18))

# Age by cluster (K-Means)
axes[0, 0].bar(range(optimal_k), cluster_summary_kmeans['Mean Age'])

# Age by cluster (Hierarchical)
axes[0, 1].bar(range(optimal_k), cluster_summary_hierarchical['Mean Age'])

# Similar for usage time and mental health severity
```

## 7. Classification Analysis

### 7.1 Data Preparation

#### 7.1.1 Feature and Target Selection
```python
X_classification = X_cluster.copy()  # Same features as clustering

# Binary classification based on median
y_classification = (smmh['mental_health_severity'] >= 7.575).map({
    True: 'High', 
    False: 'Low'
})
```

**Classification Threshold**: 7.575 (approximately median of severity distribution)

#### 7.1.2 Train-Test Split
```python
from sklearn.model_selection import train_test_split

X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    X_classification, y_classification, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_classification
)
```

**Parameters**:
- `test_size`: 0.2 (80-20 split)
- `random_state`: 42 (reproducibility)
- `stratify`: y_classification (maintains class balance)

### 7.2 Model Configuration

#### 7.2.1 Pipeline Construction
```python
from sklearn.pipeline import Pipeline

models_config_class = {
    'Random Forest': {
        'pipeline': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(random_state=42))
        ]),
        'params': {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [10, 20, None],
            'classifier__min_samples_split': [2, 5]
        }
    },
    'Gradient Boosting': {
        'pipeline': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GradientBoostingClassifier(random_state=42))
        ]),
        'params': {
            'classifier__n_estimators': [100, 200],
            'classifier__learning_rate': [0.01, 0.1],
            'classifier__max_depth': [3, 5]
        }
    },
    'SVM': {
        'pipeline': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(random_state=42))
        ]),
        'params': {
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['rbf', 'linear'],
            'classifier__gamma': ['scale', 'auto']
        }
    }
}
```

### 7.3 Model Training and Hyperparameter Tuning

#### 7.3.1 GridSearchCV Setup
```python
from sklearn.model_selection import GridSearchCV, KFold

kfold_class = KFold(n_splits=5, shuffle=True, random_state=42)

for name, config in models_config_class.items():
    grid_search = GridSearchCV(
        config['pipeline'],
        config['params'],
        cv=kfold_class,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train_class, y_train_class)
    trained_models_class[name] = grid_search.best_estimator_
```

**Parameters**:
- `cv`: 5-fold cross-validation
- `scoring`: 'accuracy' (primary metric)
- `n_jobs`: -1 (use all processors)
- `verbose`: 1 (progress updates)

#### 7.3.2 Model Evaluation
```python
y_pred = grid_search.predict(X_test_class)

accuracy = accuracy_score(y_test_class, y_pred)
f1 = f1_score(y_test_class, y_pred, average='binary', pos_label='High')
```

### 7.4 Model Evaluation Metrics

#### 7.4.1 Primary Metrics
- **Accuracy**: Overall correct prediction rate
- **F1 Score**: Harmonic mean of precision and recall (focus on 'High' class)
- **CV Accuracy**: Cross-validation accuracy from grid search

#### 7.4.2 Detailed Classification Report
```python
from sklearn.metrics import classification_report

print(classification_report(y_test_class, y_pred_best))
```

Provides:
- Precision per class
- Recall per class
- F1-score per class
- Support (sample count) per class

#### 7.4.3 Confusion Matrix
```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test_class, y_pred_best)

# Visualization
plt.imshow(cm, interpolation='nearest', cmap='Blues')
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center")
```

### 7.5 Best Model Selection

#### 7.5.1 Model Comparison
```python
classification_results = pd.DataFrame({
    'Model': model_names,
    'Best Params': best_parameters,
    'CV Accuracy': cv_accuracies,
    'Test Accuracy': test_accuracies,
    'F1 Score': f1_scores
})

classification_results = classification_results.sort_values(
    'Test Accuracy', ascending=False
)
```

#### 7.5.2 Best Model Analysis
```python
best_class_model_name = classification_results.iloc[0]['Model']
best_class_model = trained_models_class[best_class_model_name]
best_class_accuracy = classification_results.iloc[0]['Test Accuracy']
best_class_f1 = classification_results.iloc[0]['F1 Score']
```

## 8. Regression Analysis

### 8.1 Data Preparation

#### 8.1.1 Fresh Dataset Loading
```python
smmh_reg = pd.read_csv('smmh.csv')
smmh_reg = smmh_reg.rename(columns=new_column_names)
```
- Separate copy for regression preserves all component variables
- Allows investigation of which specific behaviors predict severity

#### 8.1.2 Preprocessing Pipeline
Same preprocessing steps as main analysis:
1. Drop timestamp
2. Remove whitespace
3. Standardize gender
4. Normalize scales (1-5 to 0-4)
5. Reverse score feelings_about_comparisons
6. Handle missing values
7. Remove duplicates
8. Filter gender categories

#### 8.1.3 Target Variable Calculation
```python
# Calculate composite severity score
adhd_severity = (...)/ 4
anxiety_severity = (...) / 2
self_esteem_severity = (...) / 3
depression_severity = (...) / 3

smmh_reg['mental_health_severity'] = (
    adhd_severity + anxiety_severity + 
    self_esteem_severity + depression_severity
)

# Drop component variables to prevent data leakage
smmh_reg = smmh_reg.drop(columns=[...])
```

#### 8.1.4 Feature Engineering
All categorical variables encoded:
- One-hot encoding: platforms, organizations, occupation
- Ordinal encoding: gender, relationship status, usage time
- Binary encoding: use_social_media

### 8.2 Feature and Target Separation
```python
X_regression = smmh_reg.drop(columns=['mental_health_severity'])
y_regression = smmh_reg['mental_health_severity']
```

### 8.3 Train-Test Split
```python
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_regression, y_regression, 
    test_size=0.2, 
    random_state=42
)
```

**No Stratification**: Continuous target variable

### 8.4 Feature Scaling
```python
scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)
```

**Critical**: Fit on training only, transform both sets

### 8.5 Model Configuration

#### 8.5.1 Regression Models Setup
```python
models_config = {
    'Random Forest': {
        'model': RandomForestRegressor(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    'Gradient Boosting': {
        'model': GradientBoostingRegressor(random_state=42),
        'params': {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5],
            'subsample': [0.8, 1.0]
        }
    },
    'Support Vector Regressor': {
        'model': SVR(),
        'params': {
            'C': [0.1, 1, 10, 100],
            'epsilon': [0.01, 0.1, 0.5],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        }
    }
}
```

### 8.6 Model Training and Optimization

#### 8.6.1 GridSearchCV Implementation
```python
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

for model_name, config in models_config.items():
    grid_search = GridSearchCV(
        estimator=config['model'],
        param_grid=config['params'],
        cv=kfold,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train_reg_scaled, y_train_reg)
    trained_models[model_name] = grid_search
```

**Scoring Metric**: 'neg_mean_squared_error'
- Negative because GridSearchCV maximizes score
- MSE minimization equivalent to maximizing negative MSE

### 8.7 Model Evaluation

#### 8.7.1 Test Set Performance
```python
y_pred = grid_search.best_estimator_.predict(X_test_reg_scaled)

r2 = r2_score(y_test_reg, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred))
mae = mean_absolute_error(y_test_reg, y_pred)
```

**Metrics**:
- **R² Score**: Proportion of variance explained
- **RMSE**: Root mean squared error (same units as target)
- **MAE**: Mean absolute error (robust to outliers)

#### 8.7.2 Cross-Validation Performance
```python
cv_scores = cross_val_score(
    best_model, 
    X_train_reg_scaled, 
    y_train_reg,
    cv=kfold, 
    scoring='r2'
)

cv_rmse = np.sqrt(-cross_val_score(
    best_model, 
    X_train_reg_scaled, 
    y_train_reg,
    cv=kfold, 
    scoring='neg_mean_squared_error'
))
```

### 8.8 Model Interpretation

#### 8.8.1 Feature Importance Analysis
For tree-based models (Random Forest, Gradient Boosting):
```python
feature_importances = best_model_obj.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X_regression.columns,
    'Importance': feature_importances
}).sort_values('Importance', ascending=False)

# Visualize top 15 features
top_features = feature_importance_df.head(15)
plt.barh(range(len(top_features)), top_features['Importance'])
```

**Interpretation**: Gini importance (reduction in node impurity)

#### 8.8.2 Prediction Analysis

**Predictions vs. Actual Plot**:
```python
plt.scatter(y_test_reg, best_predictions, alpha=0.6)
plt.plot([y_test_reg.min(), y_test_reg.max()], 
         [y_test_reg.min(), y_test_reg.max()], 
         'r--', lw=2, label='Perfect Prediction')
```

**Residuals Plot**:
```python
residuals = y_test_reg - best_predictions
plt.scatter(best_predictions, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
```

Assesses:
- Prediction accuracy
- Heteroscedasticity
- Systematic bias

### 8.9 Model Comparison

#### 8.9.1 Results Table
```python
regression_results = pd.DataFrame({
    'Model': model_names,
    'R² Score': r2_scores,
    'RMSE': rmse_scores,
    'MAE': mae_scores,
    'Best Parameters': best_params
})
```

#### 8.9.2 Visual Comparison
Bar charts for R², RMSE, MAE across models

#### 8.9.3 Best Model Selection
```python
best_model_idx = regression_results['R² Score'].idxmax()
best_model_name = regression_results.loc[best_model_idx, 'Model']
```

## 9. Validation and Reliability

### 9.1 Cross-Validation Strategy

#### 9.1.1 K-Fold Cross-Validation
- **K**: 5 folds
- **Shuffle**: True (randomize before splitting)
- **Random State**: 42 (reproducibility)
- **Application**: All supervised learning tasks

**Process**:
1. Split data into 5 equal parts
2. For each fold:
   - Train on 4 folds
   - Validate on 1 fold
3. Average metrics across all folds
4. Calculate standard deviation for variance estimate

### 9.2 Hyperparameter Optimization Validation

#### 9.2.1 GridSearchCV Inner CV Loop
Each hyperparameter combination evaluated via cross-validation:
- Prevents overfitting to validation set
- Provides unbiased performance estimates
- Enables fair comparison across parameter settings

#### 9.2.2 Test Set as Final Validation
- Held-out test set never seen during training
- Used only for final performance assessment
- Provides realistic generalization estimate

### 9.3 Statistical Significance

#### 9.3.1 Cross-Validation Variance
```python
cv_scores.mean()  # Average performance
cv_scores.std()   # Performance variance
```

Reports performance as: mean ± 2*std (approximate 95% CI)

#### 9.3.2 Multiple Metrics
- Reduces reliance on single metric
- Provides multifaceted performance view
- Identifies trade-offs (e.g., precision vs. recall)

### 9.4 Robustness Checks

#### 9.4.1 Random State Consistency
- Fixed random seeds across all stochastic operations
- Ensures reproducible results
- Enables result verification

#### 9.4.2 Multiple Algorithm Comparison
- Tests hypothesis across different algorithmic approaches
- Reduces algorithm-specific bias
- Validates findings through convergence

## 10. Software and Tools

### 10.1 Programming Environment
- **Language**: Python 3.x
- **Environment**: Jupyter Notebook
- **Platform**: macOS

### 10.2 Core Libraries

#### 10.2.1 Data Manipulation
```python
import numpy as np      # Version: Latest stable
import pandas as pd     # Version: Latest stable
```

#### 10.2.2 Visualization
```python
import matplotlib.pyplot as plt  # Version: Latest stable
import seaborn as sns            # Version: Latest stable
```

#### 10.2.3 Machine Learning
```python
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score, mean_absolute_error,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)
from sklearn.pipeline import Pipeline
```

#### 10.2.4 Pattern Mining
```python
from mlxtend.frequent_patterns import apriori, association_rules
```

#### 10.2.5 Hierarchical Clustering
```python
from scipy.cluster.hierarchy import dendrogram, linkage
```

### 10.3 Computational Resources
- **Parallel Processing**: `n_jobs=-1` for multi-core utilization
- **Memory Management**: Chunked processing for large operations
- **Optimization**: Efficient algorithms and vectorized operations

## 11. Ethical Considerations and Limitations

### 11.1 Ethical Research Practices

#### 11.1.1 Data Privacy
- Dataset anonymized (no personally identifiable information)
- Timestamp removed to prevent temporal identification
- Aggregate reporting to protect individual responses

#### 11.1.2 Responsible Interpretation
- Results presented with appropriate uncertainty
- Limitations clearly stated
- Caution against over-generalization

#### 11.1.3 Beneficial Application
- Focus on supportive interventions
- Avoid stigmatization of mental health conditions
- Emphasis on pattern understanding, not labeling individuals

### 11.2 Methodological Limitations

#### 11.2.1 Data Limitations
- **Self-reported data**: Potential for response bias
- **Cross-sectional design**: Cannot establish causality
- **Sample characteristics**: May not represent broader population
- **Missing data handling**: Listwise deletion reduces sample size

#### 11.2.2 Analysis Limitations
- **Feature selection threshold**: Arbitrary 0.15 correlation cutoff
- **Binary classification**: Loss of nuance in mental health severity
- **Hyperparameter grid**: Limited to pre-specified ranges
- **Computational constraints**: Some parameter combinations not tested

#### 11.2.3 Generalization Limitations
- **Temporal validity**: Patterns may change over time
- **Cultural context**: Results specific to survey population
- **Platform evolution**: Social media landscape constantly changing
- **Intervention effects**: Findings descriptive, not prescriptive

### 11.3 Validation Limitations

#### 11.3.1 Internal Validity
- No external validation dataset
- Single dataset source
- Potential confounding variables not measured

#### 11.3.2 External Validity
- Convenience sampling (likely biased)
- Limited to English-speaking respondents
- May not generalize to non-social media users

## 12. Reproducibility

### 12.1 Reproducibility Measures

#### 12.1.1 Random Seed Control
All stochastic operations use fixed random state (42):
- Train-test splits
- K-fold cross-validation
- Algorithm initialization (K-Means, Random Forest, etc.)
- PCA dimensionality reduction

#### 12.1.2 Explicit Parameter Documentation
All model parameters explicitly specified and documented

#### 12.1.3 Versioned Code
Complete methodology captured in Jupyter Notebook with:
- Sequential cell execution
- Inline comments
- Output preservation

### 12.2 Replication Protocol

#### 12.2.1 Data Requirements
- SMMH survey dataset (CSV format)
- Column structure matching specified schema

#### 12.2.2 Execution Steps
1. Load and rename columns
2. Execute preprocessing pipeline
3. Calculate mental health severity
4. Run exploratory analysis
5. Execute pattern mining
6. Perform clustering analysis
7. Train classification models
8. Train regression models
9. Generate visualizations and reports

#### 12.2.3 Expected Outputs
- Pattern mining rules
- Cluster assignments and characteristics
- Classification performance metrics
- Regression performance metrics
- Feature importance rankings
- Visualization figures

## 13. Summary

This methodology provides a comprehensive, multi-faceted approach to analyzing the relationship between social media usage and mental health. The integration of pattern mining, clustering, classification, and regression techniques enables:

1. **Discovery** of behavioral patterns through association rules
2. **Segmentation** of users into meaningful groups
3. **Prediction** of mental health risk categories
4. **Quantification** of severity relationships

The rigorous preprocessing, feature engineering, cross-validation, and hyperparameter optimization procedures ensure reliable, reproducible results. The explicit documentation of all methodological choices, parameters, and limitations enables critical evaluation and potential replication by other researchers.

The methodology balances exploratory discovery with predictive modeling, providing both interpretable insights (patterns, clusters) and actionable predictions (classification, regression) suitable for different stakeholders in mental health research and practice.
