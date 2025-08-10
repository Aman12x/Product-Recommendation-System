# Recommender Systems using Collaborative Filtering (Memory-Based)

## 1. Overview
Recommendation system built using Memory Based Collaborative Filtering on Amazon Beauty Products Dataset

This notebook demonstrates how to build a **memory-based collaborative filtering** recommender system using explicit user–item ratings.  
It walks through:
- Preparing and cleaning ratings data
- Building a **user–item rating matrix**
- Measuring similarity between users
- Recommending items based on similar users’ preferences

---

## 2. Dataset
- **Example file**: `ratings_Beauty.csv`  
- **Approximate size**: ~2.0M+ ratings, ~250K users, ~74K products  
- **Columns**:
  - `UserId`: Unique user identifier  
  - `ProductId`: Unique product identifier  
  - `Rating`: Numeric score (e.g., 1–5)  
  - `Timestamp` *(optional)*: When the rating was given  

> The dataset is stored in CSV format with one row per (user, item) rating.  
> Final counts will depend on any filtering applied.

---

## 3. Methodology
The notebook uses **User-Based Collaborative Filtering**.  
The idea: if two users rate many items similarly, they are likely to have similar taste — so we can recommend to one user the items liked by the other.

**Step-by-step process:**
1. **Data Cleaning**  
   - Remove duplicate ratings for the same (user, item) pair.  
   - Drop rows with missing values.  
   - Ensure ratings are numeric and within the expected range.

2. **Filtering**  
   - Exclude very unpopular items (few ratings) — reduces noise and sparsity.  
   - Optionally, exclude users with very few ratings — keeps only active users.

3. **User Mean-Centering**  
   - For each user, subtract their average rating from all their ratings.  
   - This normalizes differences in personal rating scales (e.g., harsh vs. generous raters).

4. **Matrix Construction**  
   - Create a **User–Item Matrix** where:  
     - Rows = users  
     - Columns = items  
     - Values = normalized ratings (0 if unrated)

5. **Similarity Computation**  
   - Use **cosine similarity** to measure how alike two users’ rating patterns are.  
   - For a given user, find the **Top-K most similar users**.

6. **Recommendation Generation**  
   - Look at the items rated highly by these similar users.  
   - Exclude items the target user has already rated.  
   - Sort remaining items by aggregated neighbor scores.  
   - Recommend the **Top-N items**.

> This approach is **memory-based** because it directly uses the entire ratings dataset for each recommendation, rather than training a predictive model.

---

## 4. Environment Setup
Install dependencies:
```bash
pip install pandas numpy scikit-learn jupyter matplotlib tqdm
