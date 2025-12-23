# K-Nearest Neighbors (KNN) Classifier - Iris Dataset

A from-scratch Python implementation of the K-Nearest Neighbors algorithm for multi-class classification.

**Author**: SHAN Coralia

##  Project Overview

This project implements the K-Nearest Neighbors (KNN) algorithm from scratch in Python without using scikit-learn. The implementation classifies data points (specifically Iris flowers) based on their features using Euclidean distance and majority voting.

##  Project Objectives

1. **Load and Parse CSV Data**: Read training data with features and labels
2. **Calculate Distances**: Compute Euclidean distance between points
3. **Classify New Points**: Predict class using k-nearest neighbors voting
4. **Batch Processing**: Generate predictions for entire test dataset
5. **Export Results**: Save predictions to CSV format

##  Project Structure

```
knn-iris-classifier/
│
├── data/
│   ├── iris/
│   │   ├── train.csv              # Training data (features + labels)
│   │   ├── test.csv               # Test data (features only)
│   │   └── Predictions.csv        # Generated predictions
│
├── src/
│   └── knn_classifier.py          # Main implementation
│
├── docs/
│   └── SHAN_Coralia_Rapport_KNN.docx  # Detailed report (French)
│
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

##  Installation

### Prerequisites

- Python 3.6+
- pip

### Install Dependencies

```bash
pip install numpy
```

### Required Modules

- **numpy**: Array manipulation and numerical operations
- **csv**: CSV file reading/writing (built-in)
- **math**: Mathematical functions for distance calculations (built-in)

##  Implementation Details

### Core Functions

#### Function 1: `knn(chemin_fichier, nouveau_point, k=3)`

**Purpose**: Classify a single point using k-nearest neighbors.

**Parameters**:
- `chemin_fichier` (str): Path to training CSV file
- `nouveau_point` (list): Feature vector to classify
- `k` (int): Number of neighbors to consider (default: 3)

**Returns**: Predicted class label (int)

**Algorithm**:
```python
def knn(chemin_fichier, nouveau_point, k=3):
    # 1. Load training data
    donnees = []      # Features
    etiquettes = []   # Labels
    
    # 2. Parse CSV file (skip header)
    # Extract features (columns 1-7) and labels (last column)
    
    # 3. Calculate Euclidean distances
    distances = []
    for point in donnees:
        distance = √(Σ(point[i] - nouveau_point[i])²)
        distances.append(distance)
    
    # 4. Find k-nearest neighbors
    indices_voisins = sort(distances)[:k]
    etiquettes_voisins = [etiquettes[i] for i in indices_voisins]
    
    # 5. Majority voting
    count = count_frequency(etiquettes_voisins)
    classe_predite = most_common(count)
    
    return classe_predite
```

#### Function 2: `KNN(entrainement, teste, predictions, k=3)`

**Purpose**: Batch classification for test dataset with CSV output.

**Parameters**:
- `entrainement` (str): Path to training CSV file
- `teste` (str): Path to test CSV file
- `predictions` (str): Path to output CSV file
- `k` (int): Number of neighbors (default: 3)

**Process**:
```python
def KNN(entrainement, teste, predictions, k=3):
    # 1. Read test file
    resultats = []
    
    # 2. For each test point
    for ligne in test_data:
        identifiant = ligne['Id']
        caracteristiques = ligne[features]
        
        # 3. Predict using knn()
        etiquette_predite = knn(entrainement, caracteristiques, k)
        resultats.append((identifiant, etiquette_predite))
    
    # 4. Write results to CSV
    save_csv(predictions, resultats, headers=['Id', 'Label'])
```


### Quick Start

```python
# Single point prediction
nouveau_point = [5.1, 3.5, 1.4, 0.2, 2.5, 3.1, 4.2]
classe = knn("data/iris/train.csv", nouveau_point, k=5)
print(f"Predicted class: {classe}")

# Batch prediction
KNN("data/iris/train.csv", "data/iris/test.csv", 
    "data/iris/Predictions.csv", k=5)
```

##  Data Format

### Training Data (train.csv)

```csv
Id,Feature1,Feature2,Feature3,Feature4,Feature5,Feature6,Feature7,Label
1,5.1,3.5,1.4,0.2,2.3,3.4,1.5,0
2,4.9,3.0,1.4,0.2,2.1,3.2,1.4,0
3,6.7,3.1,4.4,1.4,3.8,4.2,2.8,1
4,5.8,2.7,5.1,1.9,4.5,5.0,3.2,2
...
```

**Structure**:
- **Column 0**: `Id` - Unique identifier
- **Columns 1-7**: Seven feature values (floats)
- **Column 8**: `Label` - Class label (int: 0, 1, or 2)

### Test Data (test.csv)

```csv
Id,Feature1,Feature2,Feature3,Feature4,Feature5,Feature6,Feature7
501,5.0,3.3,1.4,0.2,2.2,3.3,1.4
502,6.3,2.5,4.9,1.5,3.9,4.5,2.9
503,5.8,2.7,5.1,1.9,4.6,5.1,3.3
...
```

**Structure**:
- **Column 0**: `Id` - Unique identifier
- **Columns 1-7**: Seven feature values (floats)
- **No Label column** (to be predicted)

### Output Format (Predictions.csv)

```csv
Id,Label
501,0
502,1
503,2
...
```

##  Algorithm Explanation

### Step-by-Step Process

1. **Data Loading**
   ```python
   # Read CSV, skip header
   # Extract features (columns 1-7)
   # Extract labels (column 8)
   ```

2. **Distance Calculation** (Euclidean)
   ```
   distance = √[(x₁-y₁)² + (x₂-y₂)² + ... + (x₇-y₇)²]
   ```

3. **Neighbor Selection**
   ```python
   # Sort distances ascending
   # Select k smallest distances
   # Get corresponding labels
   ```

4. **Majority Voting**
   ```python
   # Count frequency of each label
   # Return most common label
   ```

### Example Calculation

```python
# Given: nouveau_point = [5.1, 3.5, 1.4, 0.2, 2.3, 3.4, 1.5]
#        k = 3

# Step 1: Calculate distances to all training points
distances = [0.14, 0.58, 2.15, 3.42, ...]

# Step 2: Find 3 nearest neighbors
indices = [0, 1, 7]  # Indices of 3 smallest distances
labels = [0, 0, 1]   # Their corresponding labels

# Step 3: Majority voting
count = {0: 2, 1: 1}
prediction = 0  # Class 0 appears most (2 times)
```

##  Performance Analysis

### Computational Complexity

| Operation | Complexity | Description |
|-----------|-----------|-------------|
| Load Training Data | O(n) | n = number of training samples |
| Calculate Distances | O(n × d) | d = 7 features |
| Sort Distances | O(n log n) | Sort all distances |
| Majority Vote | O(k) | k neighbors |
| **Single Prediction** | **O(n × d + n log n)** | Dominated by distance + sort |
| **Batch (m points)** | **O(m × n × d + m × n log n)** | m = test samples |

### Space Complexity

- **Training data storage**: O(n × d)
- **Distance array**: O(n)
- **Total**: O(n × d)

##  Key Features

###  Advantages

- **Simple and Intuitive**: Easy to understand and implement
- **No Training Phase**: Instance-based learning
- **Multi-class Support**: Works with 3+ classes naturally
- **Non-parametric**: No assumptions about data distribution

###  Limitations

- **Computationally Expensive**: O(n) for each prediction
- **Memory Intensive**: Must store entire training set
- **Sensitive to Scale**: Features with larger ranges dominate
- **Curse of Dimensionality**: Performance degrades with many features

##  Optimization Recommendations

### 1. Feature Normalization

```python
from sklearn.preprocessing import StandardScaler

def normalize_features(train_data, test_data):
    """Standardize features to zero mean and unit variance"""
    scaler = StandardScaler()
    train_normalized = scaler.fit_transform(train_data)
    test_normalized = scaler.transform(test_data)
    return train_normalized, test_normalized
```

### 2. Vectorized Distance Calculation

```python
def knn_optimized(chemin_fichier, nouveau_point, k=3):
    # Load data (same as before)
    donnees = np.array(donnees)
    
    # Vectorized distance calculation
    distances = np.sqrt(np.sum((donnees - nouveau_point) ** 2, axis=1))
    
    # Rest of algorithm...
```

### 3. K-D Tree for Fast Neighbor Search

```python
from scipy.spatial import KDTree

# Build tree once
tree = KDTree(donnees)

# Fast queries
distances, indices = tree.query(nouveau_point, k=k)
```

### 4. Cross-Validation for Optimal K

```python
def find_best_k(train_file, k_range=(1, 21, 2)):
    """Find optimal k using cross-validation"""
    from sklearn.model_selection import cross_val_score
    
    best_k = 1
    best_score = 0
    
    for k in range(*k_range):
        # Implement cross-validation
        score = evaluate_k(train_file, k)
        if score > best_score:
            best_k = k
            best_score = score
    
    return best_k, best_score
```

##  Testing Examples

### Test Case 1: Single Prediction

```python
# Iris Setosa example
point_setosa = [5.1, 3.5, 1.4, 0.2, 2.3, 3.4, 1.5]
result = knn("train.csv", point_setosa, k=5)
# Expected: 0
```

### Test Case 2: Edge Cases

```python
# Test with k=1 (nearest neighbor only)
result_k1 = knn("train.csv", test_point, k=1)

# Test with k=n (all training points)
result_all = knn("train.csv", test_point, k=len(training_data))
```

### Test Case 3: Performance Measurement

```python
import time

start = time.time()
KNN("train.csv", "test.csv", "predictions.csv", k=5)
end = time.time()

print(f"Processing time: {end - start:.2f} seconds")
```

##  Results Validation

### Accuracy Calculation

```python
def calculate_accuracy(predictions_file, ground_truth_file):
    """Calculate classification accuracy"""
    predictions = read_csv(predictions_file)
    ground_truth = read_csv(ground_truth_file)
    
    correct = sum(p == gt for p, gt in zip(predictions, ground_truth))
    total = len(predictions)
    accuracy = correct / total * 100
    
    return accuracy
```


##  References

- Cover, T. M., & Hart, P. E. (1967). "Nearest neighbor pattern classification"
- Iris Dataset - UCI Machine Learning Repository
- Python NumPy Documentation


---

**Note**: This is an educational implementation. For production use, consider using optimized libraries like **scikit-learn**'s `KNeighborsClassifier`.
