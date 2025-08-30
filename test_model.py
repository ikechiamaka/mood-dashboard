#!/usr/bin/env python3
# mood_model_full.py

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from preprocessing import engineer_features, ensure_feature_order, FEATURE_COLUMNS

warnings.filterwarnings("ignore")


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = engineer_features(df, drop_original_timestamp=True)
    return ensure_feature_order(df)


def main():
    # 1. Load **balanced** dataset
    df = pd.read_csv('synthetic_environment_data_adjusted_with_time.csv')
    # Quick sanity check
    print("Class distribution before split:\n", df['mood'].value_counts(), "\n")

    # 2. Preprocess
    df = preprocess_data(df)

    # 3. Define features and target
    X = df[FEATURE_COLUMNS]
    y = df['mood']

    # 4. Train/test split (stratified now works because each class ≥2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. Build preprocessing + modeling pipeline
    numeric_feats = [
        'Temperature', 'Humidity', 'MQ-2', 'BH1750FVI',
        'Radar', 'Ultrasonic', 'temp_humidity_ratio', 'hour_sin', 'hour_cos'
    ]
    cat_song = ['song']
    cat_day = ['day_part']
    cat_light = ['light_category']

    numeric_transformer = SimpleImputer(strategy='mean')

    song_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first'))
    ])

    day_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder())
    ])

    light_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder())
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_feats),
        ('song', song_transformer, cat_song),
        ('day', day_transformer, cat_day),
        ('light', light_transformer, cat_light)
    ], remainder='drop')

    pipeline = Pipeline([
        ('preproc', preprocessor),
        ('clf', RandomForestClassifier(
            class_weight='balanced_subsample',
            random_state=42
        ))
    ])

    # 6. Hyperparameter tuning (Randomized Search)
    param_dist = {
        'clf__n_estimators': [100, 200, 300],
        'clf__max_depth': [5, 8, 12, None],
        'clf__min_samples_split': [2, 5, 10],
        'clf__max_features': ['sqrt', 'log2', 0.5]
    }
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=20,
        cv=5,
        scoring='balanced_accuracy',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    print("\n🔍 Starting hyperparameter search...")
    search.fit(X_train, y_train)

    print("\n🏆 Best parameters:", search.best_params_)
    print(f"🏅 Best CV balanced accuracy: {search.best_score_:.3f}")

    # 7. Evaluate on test set
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)

    bal_acc = balanced_accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    print("\n📊 Test Set Evaluation")
    print(f"  • Balanced Accuracy: {bal_acc:.3f}")
    print(f"  • Cohen's Kappa:   {kappa:.3f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique())
    )
    plt.title('Confusion Matrix (Test Set)')
    plt.xlabel('Predicted Mood')
    plt.ylabel('True Mood')
    plt.tight_layout()
    plt.show()

    # 8. Save the trained model
    joblib.dump(best_model, 'mood_model.pkl')
    print("\n💾 Model saved to mood_model.pkl")


if __name__ == '__main__':
    main()
