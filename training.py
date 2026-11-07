# %% [librerie]

import os
import joblib
import pandas as pd

# 1. Carichiamo il dataset di TRAINING
df_train = pd.read_csv("data/train_data.csv").dropna()

target_column = "Churn" 

X_train = df_train.drop(columns=["CustomerID", target_column])
y_train = df_train[target_column]

# 2. Carichiamo il dataset di TEST 
df_test = pd.read_csv("data/test_data.csv").dropna()

X_test = df_test.drop(columns=["CustomerID", target_column])
y_test = df_test[target_column]

y_train = y_train.astype(int)
y_test = y_test.astype(int)


# %% [Pipeline e Addestramento]

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

numeric_features = ["Age", "Tenure", "Usage Frequency", "Support Calls", "Payment Delay", "Total Spend"]
categorical_features = ["Gender", "Subscription Type", "Contract Length"]

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor), 
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ]
)

# Addestriamo sull'INTERO dataset di training

pipeline.set_params(
    classifier__n_estimators=150,
    classifier__max_depth=8,          # Evita che l'albero memorizzi troppo a fondo
    classifier__min_samples_leaf=10,  # Evita foglie troppo piccole e instabili
    classifier__class_weight="balanced", # Aiuta il modello a bilanciare le classi
    classifier__random_state=42
)
pipeline.fit(X_train, y_train)

# Valutiamo direttamente sul file di test separato!
accuracy = pipeline.score(X_test, y_test)
print(f"Modello addestrato! Accuratezza sul file di test separato: {accuracy*100:.1f}%")


# %% [Salvataggio Modello]

import joblib
import os

os.makedirs("../model", exist_ok=True) # O "model" a seconda di dove ti trovi
joblib.dump(pipeline, "model/classifier.joblib")
print("Modello salvato con successo!")

# %% [Controllo Distribuzione]

from sklearn.metrics import classification_report

print("--- DISTRIBUZIONE TARGET NEL TRAIN ---")
print(y_train.value_counts(normalize=True))

print("\n--- DISTRIBUZIONE TARGET NEL TEST ---")
print(y_test.value_counts(normalize=True))

print("\n--- CLASSIFICATION REPORT ---")
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# %%

import pandas as pd
import numpy as np

# Estraiamo i nomi delle feature dopo il preprocessing
cat_encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat']
encoded_cat_features = list(cat_encoder.get_feature_names_out(categorical_features))
all_features = numeric_features + encoded_cat_features

# Recuperiamo le importanze dal Random Forest
importances = pipeline.named_steps['classifier'].feature_importances_

# Creiamo un dataframe ordinato
feature_imp_df = pd.DataFrame({
    'Feature': all_features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(feature_imp_df.head(15))

# %%
probs = pipeline.predict_proba(X_test)
print("Prime 10 probabilità predette [Classe 0, Classe 1]:")
print(probs[:10])

print("Min:", probs[:, 1].min(),"- Max:", probs[:, 1].max(), "- Media:", probs[:, 1].mean())

# %% [Verifica bilanciamento reale nel test]
print("Conto esatto dei valori in y_test:")
print(y_test.value_counts())

print("\nConto delle predizioni fatte dal modello:")
print(pd.Series(y_pred).value_counts())


# %% [Test con Logistic Regression]
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Sostituiamo temporaneamente il classificatore nella pipeline
pipeline.set_params(
    classifier=LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
)

# Riaddestriamo
pipeline.fit(X_train, y_train)

# Valutiamo
y_pred = pipeline.predict(X_test)
print("--- REPORT CON LOGISTIC REGRESSION ---")
print(classification_report(y_test, y_pred))

# %%
from sklearn.linear_model import LogisticRegression
import joblib

# Addestriamo una Logistic Regression (pesa pochissimi KB)
pipeline.set_params(classifier=LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
pipeline.fit(X_train, y_train)

# Salviamo il file compatto
joblib.dump(pipeline, "model/classifier.joblib")
# %%
