# Part B — Business Case Analysis
## Scenario: Promotion Effectiveness at a Fashion Retail Chain

---

## B1. Problem Formulation (8 marks)

---

### B1(a) — ML Problem Formulation (3 marks)

#### Target Variable

The target variable is **`items_sold`** — the number of items sold at a given store during a given month under a specific promotion. This is a continuous, non-negative integer quantity.

#### Candidate Input Features

| Category | Features |
|---|---|
| **Store characteristics** | `store_id`, `store_size` (small / medium / large), `location_type` (urban / semi-urban / rural), `competition_density` |
| **Promotion** | `promotion_type` (Flat Discount, BOGO, Free Gift, Category Offer, Loyalty Points) |
| **Calendar / temporal** | `month`, `year`, `is_weekend`, `is_festival`, `is_month_end` |
| **Customer context** | Monthly footfall estimate, local demographic index (if available) |
| **Interaction terms** | `promotion_type × location_type`, `promotion_type × store_size`, `promotion_type × is_festival` |

Interaction features are particularly valuable here because the core business question is not just *"how many items does a store sell?"* but *"how does a promotion change sales in a specific store context?"* — a relationship that is inherently multiplicative.

#### Type of ML Problem

This is a **supervised regression problem** — specifically a **multi-context predictive regression** task, and at deployment it becomes a **combinatorial optimisation** problem.

- **Regression** because the target (`items_sold`) is a continuous numerical quantity, not a discrete class label.
- At **inference time**, the trained model is called five times per store per month (once for each candidate promotion), and the promotion yielding the highest predicted `items_sold` is selected — making this a **recommendation via prediction** framework, also called a *predict-then-optimise* pipeline.

#### Justification of Problem Type

A classification framing (e.g. "which promotion is best?" as a 5-class problem) is tempting but inferior because:

1. It discards magnitude — classifying BOGO as "best" does not tell you by *how much* it outperforms Flat Discount, which is essential for resource allocation and budgeting.
2. It requires labelling historical records with a "correct" promotion, which is impossible for promotions that were never tested in a given store-month combination.
3. Regression predictions can be directly compared across promotion options, naturally ranking them without any additional modelling layer.

---

### B1(b) — Why `items_sold` is a More Reliable Target than Revenue (3 marks)

#### The Case for `items_sold` over Revenue

**Revenue = items sold × price**, meaning revenue conflates two distinct signals: volume demand (driven by promotions and store behaviour) and pricing strategy (driven by finance and merchandising decisions). Using revenue as the target introduces at least three problems:

1. **Promotion-price confounding.** A Flat Discount promotion mechanically reduces the unit price, so a promotion that increases volume substantially may still show *lower* revenue than a competing promotion that barely shifts volume but keeps prices intact. The model would learn to penalise effective volume-driving promotions — the exact opposite of the business intent.

2. **Sensitivity to exogenous pricing changes.** If the finance team changes a product's price mid-year, revenue figures for that store-month shift even though promotion effectiveness has not changed. This injects noise into the target that the model cannot distinguish from signal, degrading predictive accuracy.

3. **Category mix effects.** A Category-Specific Offer may drive high volume in a lower-priced product category, yielding moderate revenue but strong strategic benefit (clearing stock, acquiring new customers). Revenue as a target would under-rate this promotion.

`items_sold` measures the **direct behavioural response to the promotion** — how many customers took a purchasing action — cleanly separated from pricing policy.

#### Broader Principle: Target Variable Selection in Real-World ML

This illustrates the principle of **target-metric alignment**: the target variable should measure *exactly* the behaviour the model is intended to influence, using the most direct and least contaminated signal available.

A common failure mode in applied ML is using a **proxy metric** that is easy to measure but incorporates confounding factors outside the model's scope of influence. Revenue, like other composite metrics (profit, Net Promoter Score, customer lifetime value estimates), bundles together multiple processes. The analyst's job is to decompose the business objective into its most direct, causally clean observable signal — in this case, the act of a customer placing an item in their basket and completing a purchase.

---

### B1(c) — Alternative to a Single Global Model (2 marks)

#### The Problem with a Single Global Model

A single model trained on all 50 stores learns one average set of feature-target relationships. If — as stated — stores in urban, semi-urban, and rural locations respond *differently* to the same promotion, the global model will produce coefficients that represent a blend of these responses, accurate on average but systematically biased for each segment. An urban flagship store and a rural convenience store do not share the same demand elasticity, competitive dynamics, or customer demographics; forcing them into a shared model suppresses these structural differences.

#### Proposed Strategy: Hierarchical / Stratified Modelling

The recommended alternative is a **hierarchical modelling strategy** combining:

**1. Stratified models by location type (primary split)**

Train three separate models — one each for `urban`, `semi-urban`, and `rural` stores. This directly addresses the stated heterogeneity in promotion response by location, ensures each model's training data reflects only comparable store contexts, and remains interpretable to the marketing team ("the urban model says BOGO drives 40 more items sold in large urban stores").

**2. Store-level random effects / fixed effects (secondary refinement)**

Within each stratum, individual stores will still differ (e.g. two large urban stores may have different footfall and competition density). A **mixed-effects regression** or a tree-based model with `store_id` as a feature can capture store-specific baselines while still generalising across stores with limited individual history.

**3. Fallback pooling for data-sparse stores**

For stores with fewer than, say, 12 months of promotion history (e.g. newly opened stores), the stratum-level model acts as a prior — a partial pooling approach that borrows strength from similar stores rather than fitting an unstable store-specific model on thin data.

#### Justification

This strategy applies the statistical principle of **partial pooling** — more sophisticated than either extreme of (a) one global model that ignores group differences or (b) 50 fully independent models that ignore shared structure and suffer from data sparsity at the individual store level. It mirrors standard industry practice in retail analytics, where location tier is the first-order segmentation variable for demand modelling.

---

## B2. Data and EDA Strategy (10 marks)

---

### B2(a) — Joining Tables and Defining Dataset Grain (4 marks)

#### Table Schemas and Join Strategy

The four source tables are joined sequentially into a single flat modelling dataset:

| Join Step | Left Table | Right Table | Join Key(s) | Type |
|---|---|---|---|---|
| 1 | `transactions` | `store_attributes` | `store_id` | LEFT JOIN |
| 2 | (result of 1) | `promotion_details` | `promotion_id` | LEFT JOIN |
| 3 | (result of 2) | `calendar` | `transaction_date` | LEFT JOIN |

**Step 1 — transactions ⟕ store_attributes on `store_id`**
Every transaction row is enriched with static store properties (`store_size`, `location_type`, `competition_density`, monthly footfall estimate). A LEFT JOIN retains all transactions even if a store attribute record is temporarily missing, with nulls flagged for investigation.

**Step 2 — (result) ⟕ promotion_details on `promotion_id`**
Attaches promotion metadata (`promotion_type`, discount depth, duration, applicable categories). Transactions with no active promotion receive `promotion_type = "none"` rather than a null, preserving them as valid training records for baseline demand estimation.

**Step 3 — (result) ⟕ calendar on `transaction_date`**
Appends date-level flags (`is_weekend`, `is_festival`, public holiday indicators). The calendar table acts as a pre-computed feature store for temporal signals that are store-agnostic.

#### Grain of the Final Modelling Dataset

> **One row = one store × one calendar month × one promotion type**

This is a **store-month-promotion** grain. Transactions are aggregated up from individual receipt level because the business decision is made at the **monthly planning level** ("which promotion should store X run in month Y?"), not at the individual transaction level. Item-level modelling would introduce high within-store variance and be computationally intractable across 50 stores × 12 months × 5 promotions.

#### Aggregations Performed Before Modelling

| Aggregation | Description |
|---|---|
| `items_sold` | SUM of items across all transactions for that store-month-promotion |
| `transaction_count` | COUNT of distinct transactions (proxy for footfall) |
| `avg_basket_size` | MEAN items per transaction |
| `revenue` | SUM of revenue (retained as a diagnostic column, not the target) |
| `promotion_days_active` | COUNT DISTINCT calendar days the promotion ran in the month |
| `peak_day_items` | MAX daily items sold in the month (captures festival/surge behaviour) |

Store attributes (`store_size`, `location_type`, `competition_density`) are static and carried forward without aggregation. Calendar flags (`is_festival`, `is_month_end`) are aggregated as binary indicators: 1 if any day in the month satisfies the condition, 0 otherwise.

---

### B2(b) — EDA Strategy (4 marks)

#### Analysis 1 — Target Distribution by Promotion Type (Grouped box plot)

**What to produce:** A grouped box plot of `items_sold` for each of the five promotion types plus the no-promotion baseline, faceted by `location_type`.

**What to look for:** Whether any promotion type has a systematically higher median lift over baseline, and whether the ranking of promotions differs across urban, semi-urban, and rural facets. Wide interquartile ranges within a promotion type signal high store-to-store variance.

**Modelling influence:** If BOGO dominates in urban stores but Flat Discount dominates in rural ones, this directly justifies the stratified modelling strategy from B1(c) and motivates `promotion_type × location_type` interaction features.

#### Analysis 2 — Temporal Trend and Seasonality Decomposition (Line chart)

**What to produce:** A monthly time series of mean `items_sold` across all stores, with promotion periods highlighted. Overlay a 3-month rolling average to expose underlying trend beneath short-term promotion noise.

**What to look for:** Systematic seasonal peaks (e.g. months 10–12 for the festive season), year-over-year growth trends, and whether promotion spikes sit visibly above the seasonal baseline or merely coincide with naturally high-demand periods.

**Modelling influence:** If strong seasonality is present, `month` should be treated as a cyclical feature (sine/cosine encoding) rather than a raw integer, which would falsely imply December (12) is far from January (1). Confirmed year-over-year growth justifies including `year` or a monotonic trend index as a feature.

#### Analysis 3 — Promotion Lift Heatmap (Store size × Promotion type)

**What to produce:** A pivot-table heatmap where rows are `store_size` categories, columns are `promotion_type` values, and cell values are mean lift — defined as (mean `items_sold` with promotion / mean `items_sold` without promotion) − 1 — expressed as a percentage.

**What to look for:** Cells with strongly positive lift (promotion works in this context), near-zero lift (promotion is ineffective), or negative lift (possible cannibalisation of full-price sales). Identify which store-size × promotion combinations deliver the highest returns.

**Modelling influence:** Cells that differ substantially across rows confirm that `store_size × promotion_type` interaction terms belong in the feature set. Near-zero lift cells suggest the model may benefit from target encoding with shrinkage rather than raw one-hot encoding for this interaction.

#### Analysis 4 — Competition Density vs Promotion Effectiveness (Scatter plot)

**What to produce:** A scatter plot of `competition_density` (x-axis) against promotion lift (y-axis), with one point per store-month, coloured by promotion type. Add a regression line per promotion type.

**What to look for:** A negative slope would indicate promotions are less effective in highly competitive areas (competitors' offers neutralise the effect). A positive slope would suggest promotions are amplified in competitive areas as customers are already in deal-seeking mode. Non-linearity or cluster separation by promotion type indicates that `competition_density` should be binned or interacted with promotion type.

**Modelling influence:** A significant correlation confirms `competition_density` as a meaningful predictor and indicates whether a linear term is sufficient or whether polynomial/binned representation is needed. Colour separation by promotion type motivates a `promotion_type × competition_density` interaction feature.

---

### B2(c) — Handling the 80% No-Promotion Imbalance (2 marks)

#### How the Imbalance Affects the Model

When 80% of training records carry `promotion_type = "none"`, the model is overwhelmingly trained on baseline demand behaviour. This creates two linked problems:

1. **Underrepresentation of promotion effects.** With only 20% of rows across all five promotion types (roughly 4% per promotion), the model has limited signal from which to learn promotion-specific lift. It defaults to predicting something close to baseline demand regardless of the promotion input — effectively ignoring the variable the business most wants to act on.

2. **Biased loss contribution.** In gradient-based and tree-based models, the majority class dominates the loss function. Splits that correctly predict the 80% no-promotion cases are heavily rewarded; splits that capture the rarer but business-critical promotion uplift are under-weighted.

#### Steps to Address the Imbalance

**1. Reframe the target as promotion lift, not raw volume**
Compute a store-month-level baseline (e.g. rolling 3-month average `items_sold` for that store during no-promotion months) and define the target as `lift = items_sold − baseline`. This centres the problem specifically on the incremental promotion effect, making all records — promotion and no-promotion alike — equally informative, and directly answering the business question: *"by how much does this promotion change sales?"*

**2. Oversample promotion records**
If the raw `items_sold` formulation is retained, apply random oversampling of minority promotion rows in the training set to achieve a more balanced representation. This must be applied **only to the training set** after the temporal split to avoid data leakage. SMOTE (Synthetic Minority Over-sampling Technique) can generate synthetic promotion records by interpolating between real ones in feature space.

**3. Sample weights in model training**
Assign higher sample weights to promotion-period rows when fitting the model. Most scikit-learn estimators accept a `sample_weight` parameter; setting weights inversely proportional to promotion-type frequency (e.g. 5× for any promotion record vs 1× for no-promotion) directly counteracts the imbalance without altering the dataset itself.

**4. Stratified cross-validation**
During hyperparameter tuning, use stratified k-fold cross-validation stratified on `promotion_type` to ensure each fold contains a representative proportion of each promotion type. This prevents any fold from being evaluated almost entirely on no-promotion records, which would produce misleadingly high cross-validation scores that mask poor promotion-effect prediction.

---

## B3. Model Evaluation and Deployment (12 marks)

---

### B3(a) — Train-Test Split, Metrics, and Interpretation (4 marks)

#### Setting Up the Train-Test Split

With three years of monthly data across 50 stores, the dataset contains approximately 1,800 store-month records (36 months × 50 stores). The correct split strategy is a **temporal holdout** combined with a **walk-forward validation** scheme:

**Step 1 — Fixed holdout test set (last 6 months)**
Reserve the final 6 months (months 31–36) as a completely untouched test set. This simulates the production condition exactly: the model is trained on all available history and evaluated on a future it has never seen. The test window covers at least one full seasonal cycle (spanning both a peak festive period and a quieter post-season period), ensuring the evaluation reflects realistic demand variability.

**Step 2 — Walk-forward cross-validation on the training set (months 1–30)**
Rather than a single train/validation split within the training window, use **expanding-window cross-validation** (also called walk-forward or time-series CV):

```
Fold 1:  Train months 1–12  │  Validate months 13–15
Fold 2:  Train months 1–15  │  Validate months 16–18
Fold 3:  Train months 1–18  │  Validate months 19–21
Fold 4:  Train months 1–21  │  Validate months 22–24
Fold 5:  Train months 1–24  │  Validate months 25–27
Fold 6:  Train months 1–27  │  Validate months 28–30
```

Each fold trains on all data up to a cutoff and validates on the immediately following 3-month window. This prevents any validation window from being informed by future data and produces six honest estimates of generalisation error under different training-data volumes — revealing whether performance is still improving as more data is added (a sign the model is not yet saturated).

**Why a random split is inappropriate** (extending the reasoning from B1(c)):
A random split would scatter future months into the training set and past months into the test set. The model would be trained on, say, month 34 data and evaluated on month 5 data — a temporally inverted regime impossible in production. More subtly, because retail demand is autocorrelated (month 5 sales are correlated with month 4 and 6 sales), randomly splitting breaks this autocorrelation structure, leaking temporal information across the train/test boundary and producing test-set metrics that are far too optimistic. The model would appear to generalise well during evaluation but fail immediately upon live deployment.

#### Evaluation Metrics and Business Interpretation

**1. Root Mean Squared Error (RMSE)**
Measures the standard deviation of prediction errors in the same units as `items_sold`. RMSE penalises large errors disproportionately due to the squared term.

*Business interpretation:* An RMSE of 30 items means the model's predictions are, on average, off by about 30 items per store-month. Since each item has a margin attached, RMSE translates directly into expected financial exposure from mis-allocation of promotions. Large RMSE spikes in specific months (e.g. festive season) indicate where uncertainty is highest and where human override may be warranted.

**2. Mean Absolute Error (MAE)**
Measures the average absolute prediction error in units of items sold. Unlike RMSE, MAE weights all errors equally regardless of magnitude.

*Business interpretation:* MAE is the more operationally intuitive metric for marketing teams — "our promotion volume forecasts are wrong by X items on average." It is also more robust to outlier months (e.g. an unexpected supply disruption) that would inflate RMSE without representing typical model behaviour. Reporting both RMSE and MAE together reveals whether errors are broadly consistent (RMSE ≈ MAE × 1.25) or driven by a small number of large misses (RMSE >> MAE).

**3. Mean Absolute Percentage Error (MAPE) — by store tier**
Measures prediction error as a percentage of actual items sold, computed separately for urban, semi-urban, and rural store strata.

*Business interpretation:* MAPE normalises for store size — a 30-item error in a large urban store (selling 500 items/month) is far less concerning than the same error in a small rural store (selling 80 items/month). Stratified MAPE reveals whether the model performs consistently across all location types or whether one tier is systematically less well-served, informing decisions about tier-specific model refinement.

**4. Promotion Recommendation Accuracy (custom metric)**
For each store-month in the test set, identify the *actual* best-performing promotion (highest observed `items_sold`) and compare it to the model's top recommendation. Compute the proportion of store-months where the model's top-1 recommendation matches the empirically best promotion.

*Business interpretation:* This is the most direct measure of decision quality. A model with low RMSE can still make the wrong promotion recommendation if it correctly predicts the volume for four promotions but mis-ranks the top two. A recommendation accuracy of 70% means the model selects the empirically optimal promotion 7 times out of 10, allowing the marketing team to quantify expected revenue upside vs a random or intuition-based baseline.

---

### B3(b) — Investigating and Communicating Different Recommendations (4 marks)

#### Why the Model Recommends Differently for the Same Store

The model recommends **Loyalty Points Bonus in December** and **Flat Discount in March** for Store 12 because the two months present fundamentally different feature vectors — even though the store identifier is the same. The recommendation is not a property of the store alone; it is the output of the model evaluated on the full feature context for that store-month combination.

#### Investigation Using Feature Importance

**Step 1 — Construct the two feature vectors side by side**

Extract the exact input rows the model used to generate each recommendation and display them in a comparison table:

| Feature | Store 12 — December | Store 12 — March |
|---|---|---|
| `month` | 12 | 3 |
| `is_festival` | 1 | 0 |
| `is_weekend` (proportion) | 0.48 | 0.43 |
| `is_month_end` | 1 | 1 |
| `competition_density` | 5 | 5 |
| `store_size` | medium | medium |
| `location_type` | urban | urban |

The store-level features are identical — `store_id`, `store_size`, `location_type`, and `competition_density` do not change. The meaningful differences are confined to **temporal context**: `month = 12` vs `3`, and critically `is_festival = 1` vs `0`.

**Step 2 — Apply SHAP values to each prediction**

Use **SHAP (SHapley Additive exPlanations)** to decompose each of the ten promotion-score predictions (5 promotions × 2 months) into individual feature contributions. SHAP assigns each feature a signed contribution (in units of `items_sold`) showing how much that feature pushed the predicted score above or below the model's baseline.

For December, the SHAP waterfall for Loyalty Points Bonus would show large positive contributions from `is_festival = 1` and `month = 12`, because the training data has revealed that loyalty-rewarding promotions resonate strongly with customers making considered, high-value festive purchases.

For March, the SHAP waterfall for Flat Discount would show that without the festival signal, price sensitivity (`competition_density` and `month` seasonality) becomes the dominant driver, and Flat Discount wins because it directly addresses value-seeking behaviour in a low-engagement month.

**Step 3 — Produce a promotion score comparison chart**

Generate a bar chart of predicted `items_sold` for all five promotions in each month. This shows not just which promotion wins but by how much — communicating the confidence margin of the recommendation to the marketing team.

#### Communicating to the Marketing Team

Translate the technical findings into a business narrative without jargon:

> *"In December, we predict Loyalty Points Bonus will drive the most sales for Store 12. The main reason is the festival period — our data shows that during high-footfall festive months, customers respond particularly well to promotions that reward loyalty rather than cutting prices immediately, likely because they are already planning to spend and appreciate recognition. In March, the festival effect is absent and customer traffic is lower, so a Flat Discount works harder — it gives customers a concrete, immediate reason to visit the store and make a purchase in an otherwise quiet month."*

This narrative directly maps model feature importance to the human intuitions the marketing team already holds, building trust in the model's reasoning and making the recommendations actionable.

---

### B3(c) — End-to-End Deployment and Monitoring (4 marks)

#### 1. Saving the Model

At the end of training, serialise the **complete pipeline object** — preprocessor and model together — using `joblib`:

```python
import joblib
joblib.dump(pipeline, 'promotion_model_v1.pkl')
```

Saving the full pipeline (not just the model weights) is essential because the `StandardScaler` mean/variance statistics and `OneHotEncoder` category vocabularies learned during training must be applied identically at inference time. Separating the preprocessor from the model risks version mismatch. The saved artefact is versioned (e.g. `v1`) and stored in a model registry (MLflow, AWS S3, or an internal artefact store) alongside:
- Training data date range and row count
- Evaluation metrics on the holdout test set
- The full feature schema (column names, dtypes, expected value ranges)
- Python and library version manifest (`requirements.txt`)

#### 2. Preparing and Feeding New Monthly Data

At the start of each month, an automated pipeline runs the following steps:

**a) Data extraction**
Pull the previous month's completed transaction data from the operational database and join it to the store attributes, promotion details, and calendar tables using the same join logic described in B2(a).

**b) Feature engineering**
Apply identical transformations to those used during training: aggregate to store-month grain, compute `items_sold` totals and derived columns, extract date features (`month`, `year`, `is_festival`, `is_month_end`). This logic is encapsulated in a versioned feature engineering script to guarantee consistency.

**c) Inference — predict-then-optimise**
For each of the 50 stores, construct five input rows (one per promotion type) with all other features held constant at their current values. Pass all 250 rows through the loaded pipeline:

```python
pipeline = joblib.load('promotion_model_v1.pkl')
predictions = pipeline.predict(inference_df)   # shape: (250,)
```

For each store, select the promotion with the highest predicted `items_sold` as the recommendation. Output a 50-row recommendation table with the top promotion and its predicted volume for each store, delivered to the marketing team's dashboard or CRM system.

#### 3. Monitoring for Model Degradation

Three complementary monitoring layers are put in place:

**a) Data drift monitoring (input distribution)**
Each month, compute summary statistics (mean, standard deviation, category proportions) for every input feature and compare them to the training-set baseline using statistical tests (Kolmogorov-Smirnov for continuous features, chi-squared for categorical). Flag features where drift exceeds a threshold (e.g. KS statistic > 0.15). Drift in `competition_density` could indicate new competitor store openings; drift in `promotion_type` distribution could indicate a shift in marketing strategy. Either warrants human review before the next inference cycle.

**b) Prediction drift monitoring (output distribution)**
Track the distribution of predicted `items_sold` values each month. A sudden shift — e.g. all predictions compressing toward the mean or the model repeatedly recommending only one promotion type — signals that input drift has caused the model to enter an extrapolation regime where its predictions are unreliable.

**c) Ground-truth performance tracking (actuals vs predictions)**
Once each month's actual `items_sold` figures are available (typically 2–4 weeks after month close), compute RMSE and MAE for that month and plot them on a rolling control chart alongside the training-period baseline. Define explicit retraining triggers:

- **Soft alert:** Rolling 3-month RMSE exceeds the training-set RMSE by more than 15% — schedule a model review.
- **Hard alert:** Rolling 3-month RMSE exceeds the training-set RMSE by more than 30%, or recommendation accuracy drops below 55% for two consecutive months — pause automated recommendations and retrain immediately on the expanded dataset.

**d) Retraining cadence**
Even in the absence of hard alerts, retrain the model every 6 months on an expanding window of data to incorporate new promotion-response patterns, new store openings, and evolving customer behaviour. The walk-forward cross-validation scheme from B3(a) is re-run at each retraining cycle to confirm the new model improves on the previous version before it is promoted to production.