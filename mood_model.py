# mood_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# 1. Load and preprocess data
df = pd.read_csv("synthetic_environment_data_adjusted_with_time.csv")

def preprocess_data(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_part'] = pd.cut(df['hour'],
                            bins=[0,6,12,18,24],
                            labels=['Night','Morning','Afternoon','Evening'])
    df['temp_humidity_ratio'] = df['Temperature'] / (df['Humidity'] + 1e-6)
    df['light_category'] = pd.cut(df['BH1750FVI'],
                                  bins=[0, 100, 500, 1000, np.inf],
                                  labels=['Dark','Low','Medium','Bright'])
    df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
    df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)
    df['MQ-2'] = df['MQ-2'].map({'OK':0, 'N_OK':1})
    return df.drop(columns=['timestamp','hour'])

df = preprocess_data(df)

# 2. Define features/target
feature_cols = [
    'Temperature','Humidity','MQ-2','BH1750FVI','Radar','Ultrasonic',
    'song','day_part','light_category','temp_humidity_ratio','hour_sin','hour_cos'
]
X = df[feature_cols]
y = df['mood']

# 3. Build preprocessing + modeling pipeline
numeric_feats = ['Temperature','Humidity','MQ-2','BH1750FVI','Radar','Ultrasonic','temp_humidity_ratio','hour_sin','hour_cos']
categorical_song = ['song']
categorical_day = ['day_part']
categorical_light = ['light_category']

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
    ('song', song_transformer, categorical_song),
    ('day', day_transformer, categorical_day),
    ('light', light_transformer, categorical_light)
], remainder='drop')

pipeline = Pipeline([
    ('preproc', preprocessor),
    ('clf', RandomForestClassifier(class_weight='balanced', random_state=42))
])

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Hyperparameter tuning for best balanced accuracy
param_dist = {
    'clf__n_estimators': [100,200,300],
    'clf__max_depth': [None, 8, 12, 16],
    'clf__min_samples_split': [2,5,10],
    'clf__max_features': ['sqrt','log2',0.5]
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

print("Starting hyperparameter search...")
search.fit(X_train, y_train)
best_model = search.best_estimator_

print("\nBest Parameters:", search.best_params_)
print(f"Best CV Balanced Accuracy: {search.best_score_:.3f}")

# 6. Final evaluation on test set
y_pred = best_model.predict(X_test)
print("\nTest Set Evaluation:")
print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.3f}")
print(f"Cohen's Kappa: {cohen_kappa_score(y_test, y_pred):.3f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred, labels=best_model.named_steps['clf'].classes_)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=best_model.named_steps['clf'].classes_,
            yticklabels=best_model.named_steps['clf'].classes_)
plt.xlabel('Predicted Mood')
plt.ylabel('True Mood')
plt.title('Confusion Matrix')
plt.show()

# 7. Feature importance
feat_names = []
# get feature names after preprocessing
ohe_feats = best_model.named_steps['preproc'] \
    .named_transformers_['song'] \
    .named_steps['onehot'] \
    .get_feature_names_out(categorical_song)
feat_names.extend(numeric_feats)
feat_names.extend(ohe_feats)
feat_names.append('day_part_encoded')
feat_names.append('light_category_encoded')

importances = best_model.named_steps['clf'].feature_importances_
imp_series = pd.Series(importances, index=feat_names).sort_values(ascending=True)

plt.figure(figsize=(8,6))
imp_series.plot(kind='barh')
plt.title('Feature Importances')
plt.show()

# 8. Save the optimized model
joblib.dump(best_model, 'optimized_mood_predictor.pkl')
print("Optimized model saved to optimized_mood_predictor.pkl")
