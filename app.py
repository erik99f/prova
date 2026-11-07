# %% [Streamlit App]
import streamlit as st
import pandas as pd
import numpy as np
import os
import shap
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# Configurazione della pagina
st.set_page_config(
    page_title="Customer Churn Prediction App",
    page_icon="📊",
    layout="wide"
)

# Titolo principale
st.title("Customer Churn Prediction Dashboard")
st.markdown("""
Questa applicazione analizza il rischio di abbandono (churn) dei clienti basandosi sui dati storici. 
*Nota sui dati:* Trattandosi di un dataset con segnali complessi e rumorosi, il modello riflette le sfide tipiche dei benchmark di settore, offrendo una base solida per l'analisi esplorativa e la simulazione.
""")

# Funzione robusta per trovare il dataset di training
def find_train_file():
    possible_paths = ["data/train_data.csv", "train_data.csv", "./data/train_data.csv"]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

# Addestramento modello
@st.cache_resource
def get_trained_pipeline():
    train_path = find_train_file()
    if train_path is None:
        return None, None, None
    
    train_df = pd.read_csv(train_path)
    
    # Identifica la colonna target
    target_column = 'Churn'
    train_df = train_df.dropna(subset=[target_column])
    
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    
    # Rimuoviamo preventivamente qualsiasi colonna ID dalle feature di training
    id_cols = [col for col in X_train.columns if 'id' in col.lower()]
    X_train = X_train.drop(columns=id_cols)
    
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100, 
            class_weight='balanced', 
            random_state=42))
    ])
    
    model.fit(X_train, y_train)
    return model, X_train.columns.tolist(), train_df

pipeline, feature_columns, train_df = get_trained_pipeline()

if pipeline is None:
    st.error("⚠️ Attenzione: Impossibile trovare il file `train_data.csv` in nessun percorso conosciuto.")
else:
    st.success("Modello addestrato e caricato con successo")

    # Sidebar per l'input utente dinamico e pulito
    st.sidebar.header("Simula un Cliente")
    
    def user_input_features():
        data = {}
        train_path = find_train_file()
        original_train = pd.read_csv(train_path)
        target_column = 'Churn' if 'Churn' in original_train.columns else original_train.columns[-1]
        original_train = original_train.drop(columns=[target_column])
        
        # Rimuove gli ID anche qui per coerenza
        id_cols = [col for col in original_train.columns if 'id' in col.lower()]
        original_train = original_train.drop(columns=id_cols)
        
        for col in feature_columns:
            if col in original_train.select_dtypes(include=['int64', 'float64']).columns:
                min_val = float(original_train[col].min())
                max_val = float(original_train[col].max())
                mean_val = float(original_train[col].mean())
                
                # Se è una colonna di spesa o monetaria, diamo un formato pulito
                if 'spend' in col.lower() or 'amount' in col.lower():
                    data[col] = st.sidebar.slider(f"{col}", min_val, max_val, float(round(mean_val, 2)), step=0.1)
                elif original_train[col].dropna().apply(lambda x: x.is_integer()).all():
                    data[col] = st.sidebar.slider(f"{col}", int(min_val), int(max_val), int(mean_val), step=1)
                else:
                    data[col] = st.sidebar.slider(f"{col}", min_val, max_val, mean_val)
            else:
                unique_vals = original_train[col].dropna().unique().tolist()
                data[col] = st.sidebar.selectbox(f"{col}", unique_vals)
                
        return pd.DataFrame([data])

    input_df = user_input_features()

    # Layout principale a colonne
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Parametri Cliente Selezionati")
        st.dataframe(input_df, use_container_width=True)

        if st.button("Predici Rischio Churn", type="primary"):
            prediction = pipeline.predict(input_df)
            prediction_proba = pipeline.predict_proba(input_df)
            
            st.subheader("Risultato della Predizione:")
            if prediction[0] == 1:
                st.error(f"⚠️ **Alto Rischio di Churn!** (Probabilità: {prediction_proba[0][1]*100:.1f}%)")
            else:
                st.success(f"**Cliente Stabile / Fedele** (Probabilità Churn: {prediction_proba[0][1]*100:.1f}%)")
            st.markdown("---")
            
    st.subheader("🔍 Spiegazione della Predizione (SHAP Values)")
    st.info("Questo grafico mostra quali feature hanno spinto la previsione verso l'alto (rischio churn) o verso il basso (fedeltà) per questo specifico cliente.")

    # Estraiamo il modello e il preprocessor dalla pipeline
    # Estraiamo il modello e il preprocessor dalla pipeline
    model_clf = pipeline.named_steps['classifier']
    preprocessor = pipeline.named_steps['preprocessor']
    
    # Trasformiamo l'input dell'utente usando lo stesso preprocessor della pipeline
    X_input_transformed = preprocessor.transform(input_df)
    
    # Ricaviamo i nomi delle feature trasformate dall'encoder/scaler
    try:
        raw_feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        raw_feature_names = [f"feature_{i}" for i in range(X_input_transformed.shape[1])]

    # --- COSTRUZIONE ETICHETTE CON VALORI REALI E GESTIONE CATEGORICHE ---
    clean_feature_names = []
    for i, name in enumerate(raw_feature_names):
        # Valore trasformato (per le categoriche one-hot sarà 0.0 o 1.0)
        transformed_val = X_input_transformed[0, i]
        
        cleaned_name = name.replace("num__", "").replace("cat__", "").replace("_", ": ")
        
        # Controlliamo se questa feature deriva da una variabile categorica
        is_categorical_dummy = any(col in name for col in input_df.select_dtypes(include=['object', 'category']).columns)
        
        if is_categorical_dummy:
            # Per le categoriche, mostriamo 1 o 0 in base allo stato attivo/inattivo dell'encoder
            val_display = int(round(transformed_val))
            clean_feature_names.append(f"{val_display} = {cleaned_name}")
        else:
            # Per le numeriche, peschiamo il valore esatto inserito dall'utente nei slider
            original_val = None
            for col in input_df.select_dtypes(include=['int64', 'float64']).columns:
                if col in name:
                    original_val = input_df[col].values[0]
                    break
            
            if original_val is not None:
                # Se la feature contiene "Spend" o "Amount" (o simili), arrotondiamo a 0 o 2 decimali puliti
                if 'spend' in name.lower() or 'amount' in name.lower() or 'cost' in name.lower():
                    val_str = int(round(original_val)) # Oppure int(round(original_val)) se vuoi zero decimali fissi
                else:
                    val_str = str(original_val)
                clean_feature_names.append(f"{val_str} = {cleaned_name}")
            else:
                clean_feature_names.append(f"{transformed_val:.2f} = {cleaned_name}")

    # Inizializziamo l'explainer e calcoliamo i valori SHAP
    explainer = shap.TreeExplainer(model_clf)
    shap_values = explainer(X_input_transformed)

    # Generiamo il plot
    fig, ax = plt.subplots(figsize=(8, 4))
    
    if len(shap_values.shape) == 3:
        shap.plots.waterfall(shap.Explanation(
            values=shap_values.values[0, :, 1],
            base_values=explainer.expected_value[1],
            data=None,
            feature_names=clean_feature_names  # Etichette con valori reali!
        ), max_display=10, show=False)
    else:
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        
    st.pyplot(fig, clear_figure=True)

    with col2:
        st.subheader("Informazioni sul Progetto")
        st.info("""
- **Tech Stack:** Python, Scikit-learn, Streamlit.
- **Deployment:** GitHub & Streamlit Community Cloud.
- **Focus:** Pipeline ML dinamica e riproducibile.
        """)

    # Sezione Dataset di Test
    st.markdown("---")
    st.subheader("Visualizzazione rapida dataset di test")
    test_path = "data/test_data.csv" if os.path.exists("data/test_data.csv") else "test_data.csv"
    if os.path.exists(test_path):
        test_preview = pd.read_csv(test_path)
        st.dataframe(test_preview.head(10), use_container_width=True)
    else:
        st.warning("File di test non trovato.")