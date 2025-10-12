# ML Challenge 2025 - Methodology Documentation

## Team Approach Summary

### Problem Understanding
Predict product prices using catalog content (text descriptions and metadata) and image links. The challenge uses SMAPE as the evaluation metric, which penalizes both over and under-predictions equally.

---

## 1. Methodology

### Approach
Our solution employs a **text-based machine learning approach** using XGBoost regression with extensive feature engineering from product catalog content. We focus exclusively on text-derived features, extracting meaningful patterns from item names, descriptions, and metadata.

### Why This Approach?
- **Text-rich data**: Product catalogs contain rich semantic information about pricing factors
- **XGBoost advantages**: Handles non-linear relationships, provides feature importance, robust to outliers
- **SMAPE optimization**: Ensemble approach reduces prediction variance, crucial for SMAPE metric
- **Scalability**: Efficient training on 75k samples with fast inference

---

## 2. Model Architecture

### Algorithm: XGBoost Regressor

**Core Model**: Gradient Boosted Decision Trees optimized for regression

**Key Hyperparameters**:
- Learning rate: 0.05 (conservative for stability)
- Max depth: 7 (balanced complexity)
- Subsample: 0.8 (prevent overfitting)
- Colsample_bytree: 0.8 (feature randomness)
- Regularization: L1=0.1, L2=1.0 (prevents overfitting)
- Early stopping: 50 rounds (optimal iteration detection)

**Training Strategy**:
- 5-fold cross-validation for robust evaluation
- Ensemble averaging across all folds
- Out-of-fold predictions for unbiased performance estimation

---

## 3. Feature Engineering

### 3.1 Text Parsing Features
**Extraction from catalog_content**:
- Item name
- Bullet points (concatenated)
- Product description
- IPQ (Item Pack Quantity) value and unit

**Derived Features**:
- Text length (characters)
- Word count
- Item name length
- Description length
- Bullet points length

### 3.2 TF-IDF Features (Top 100)
**Text Vectorization**:
- N-grams: 1-2 (captures phrases like "organic coffee")
- Stop words removed (English)
- Min document frequency: 2
- Max document frequency: 95%

**Purpose**: Captures product-specific keywords that correlate with price (e.g., "premium", "organic", "pack of")

### 3.3 Numeric Features
**Extracted from text**:
- Count of numbers in text
- Maximum number
- Minimum number
- Average of numbers
- Sum of numbers

**Rationale**: Product sizes, quantities, and pack counts often appear as numbers in descriptions

### 3.4 Category Features (Binary Flags)
- is_organic
- is_food
- is_beverage
- is_snack
- is_kosher
- is_gluten_free
- is_vegan
- is_gmo_free
- is_pack

**Rationale**: Product categories and certifications strongly influence pricing

### 3.5 Unit Features
**Standardized units**:
- ounce, fluid_ounce, count, pound, gram, kilogram, other, unknown
- Label encoded for model consumption

**Rationale**: Unit type affects price interpretation (bulk vs. individual)

### 3.6 Metadata Features
- has_ipq_value (binary flag)
- ipq_value (normalized quantity)
- has_image (binary flag)

**Total Feature Count**: ~130 features
- TF-IDF: 100 features
- Text statistics: 5 features
- Numeric features: 5 features
- Category flags: 9 features
- Unit encoding: 1 feature
- Metadata: 3 features
- Additional derived: ~7 features

---

## 4. Training Process

### Data Preprocessing
1. Parse catalog_content into structured components
2. Extract all feature types simultaneously
3. Handle missing values (fillna strategies)
4. Normalize text (lowercase, clean)
5. Encode categorical variables

### Model Training
1. **K-Fold Split**: 5-fold stratified cross-validation
2. **Per-Fold Training**:
   - Train on 4 folds, validate on 1 fold
   - Early stopping based on validation RMSE
   - Record out-of-fold predictions
3. **Ensemble**: Average predictions from all 5 models
4. **Validation**: Calculate overall SMAPE on out-of-fold predictions

### Prediction Generation
1. Each fold model generates predictions on test set
2. Final prediction = average of 5 fold predictions
3. Post-processing: Ensure all prices > 0.01

---

## 5. Model Strengths

### Text Understanding
- TF-IDF captures brand names, product types, and quality indicators
- N-grams capture multi-word patterns ("extra virgin", "pack of 6")

### Robust to Outliers
- Tree-based models handle extreme prices naturally
- Regularization prevents overfitting to rare patterns

### Feature Importance
- Model provides interpretability through feature rankings
- Top features typically include: pack size, organic flag, specific units, brand keywords

### Ensemble Benefits
- Reduces prediction variance
- More stable for SMAPE metric
- Handles different product categories better

---

## 6. Validation & Results

### Cross-Validation Strategy
- 5-fold CV provides reliable performance estimate
- Out-of-fold predictions simulate real test performance
- Each fold sees different validation data

### Performance Metrics
- Primary: SMAPE (Symmetric Mean Absolute Percentage Error)
- Secondary: RMSE (for early stopping)
- Feature importance scores for interpretability

### Expected Performance
- CV SMAPE typically: 0.15-0.25 (15-25%)
- Lower is better
- Performance depends on data quality and feature relevance

---

## 7. Key Decisions & Trade-offs

### Why Not Images?
- Text features provide strong baseline performance
- Image processing requires significant compute resources
- Text-only approach is simpler and faster
- Future work could incorporate vision models

### Why XGBoost over Neural Networks?
- Excellent performance on tabular features
- Faster training (minutes vs hours)
- Better interpretability
- Lower risk of overfitting with limited data

### Why TF-IDF over Embeddings?
- Interpretable features (can see which words matter)
- No pre-training required
- Works well with product descriptions
- Computationally efficient

---

## 8. Academic Integrity Compliance

### Data Sources
✓ Used only provided training data (train.csv)
✓ No external price lookups
✓ No web scraping
✓ No additional datasets

### Model Compliance
✓ XGBoost: Apache 2.0 License (open source)
✓ Scikit-learn: BSD License (open source)
✓ Model parameters: <8B (actually ~1M parameters)
✓ All code is original

---

## 9. Potential Improvements

### Feature Engineering
- Product hierarchy extraction
- Brand name dictionary
- Seasonal pattern detection
- Competitor price patterns (from text mentions)

### Model Enhancements
- Hyperparameter optimization (Optuna/GridSearch)
- Multi-modal approach (text + images)
- Neural network ensemble
- Target encoding for high-cardinality features

### Engineering Optimizations
- Feature selection (remove low-importance features)
- Dimensionality reduction (PCA on TF-IDF)
- Stacking with multiple model types

---

## 10. Conclusion

Our XGBoost-based solution leverages comprehensive text feature engineering to predict product prices from catalog content. The approach balances performance, interpretability, and computational efficiency while strictly adhering to competition rules.

**Key Success Factors**:
1. Rich feature extraction from text
2. Robust ensemble approach
3. Proper cross-validation
4. SMAPE-optimized training

**Reproducibility**: All code is provided with clear documentation and can be executed with standard Python packages.

---

## Technical Stack

- **Language**: Python 3.9+
- **Core Libraries**:
  - XGBoost 2.0+ (model)
  - Scikit-learn 1.3+ (feature engineering)
  - Pandas 2.0+ (data handling)
  - NumPy 1.24+ (numerical operations)
- **License**: All MIT/Apache 2.0/BSD open source

---

**Team Submission**: ML Challenge 2025
**Approach**: Text-Based XGBoost Regression
**Evaluation Metric**: SMAPE
