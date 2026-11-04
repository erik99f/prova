# %% [Streamlit App]
import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

# Configurazione della pagina
st.set_page_config(
    page_title="Customer Churn Prediction App",
    page_icon="📊",
    layout="wide"
)

# Titolo principale
st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("""
Questa applicazione analizza il rischio di abbandono (churn) dei clienti basandosi sui dati storici. 
*Nota sui dati:* Trattandosi di un dataset con segnali complessi e rumorosi, il modello riflette le sfide tipiche dei benchmark di settore, offrendo una base solida per l'analisi esplorativa e la simulazione.
""")

# Funzione per addestrare al volo il modello leggero usando i dati di training
@st.cache_resource
def get_trained_pipeline():
    train_path = "data/train_data.csv"
    if not os.path.exists(train_path):
        return None
    
    train_df = pd.read_csv(train_path)
    
    # Identifica la colonna target
    target_column = 'Churn' if 'Churn' in train_df.columns else train_df.columns[-1]
    
    # Rimuove le righe dove il target è NaN per evitare errori di validazione
    train_df = train_df.dropna(subset=[target_column])
    
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    from sklearn.impute import SimpleImputer
    
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
        ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    ])
    
    model.fit(X_train, y_train)
    return model
pipeline = get_trained_pipeline()

if pipeline is None:
    st.error("⚠️ Attenzione: Impossibile trovare il file `data/train_data.csv` per addestrare il modello al volo.")
else:
    st.success("✅ Modello addestrato e caricato con successo!")

    # Sidebar per l'input utente (Simulazione singolo cliente)
    st.sidebar.header("🔍 Simula un Cliente")
    
    def user_input_features():
        age = st.sidebar.slider("Età (Age)", 18, 70, 35)
        tenure = st.sidebar.slider("Anzianità contratto (Tenure - mesi)", 1, 60, 12)
        usage_freq = st.sidebar.slider("Frequenza di Utilizzo", 1, 30, 10)
        support_calls = st.sidebar.slider("Chiamate di Supporto", 0, 10, 2)
        payment_delay = st.sidebar.slider("Ritardo Pagamenti (giorni)", 0, 30, 5)
        total_spend = st.sidebar.number_input("Spesa Totale ($)", 100.0, 5000.0, 500.0)
        
        gender = st.sidebar.selectbox("Genere", ["Male", "Female"])
        sub_type = st.sidebar.selectbox("Tipo Abbonamento", ["Basic", "Standard", "Premium"])
        contract_length = st.sidebar.selectbox("Durata Contratto", ["Monthly", "Quarterly", "Annual"])
        
        data = {
            "Age": age,
            "Tenure": tenure,
            "Usage Frequency": usage_freq,
            "Support Calls": support_calls,
            "Payment Delay": payment_delay,
            "Total Spend": total_spend,
            "Gender": gender,
            "Subscription Type": sub_type,
            "Contract Length": contract_length
        }
        return pd.DataFrame(data, index=[0])

    input_df = user_input_features()

    # Layout principale a colonne
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📋 Parametri Cliente Selezionati")
        st.dataframe(input_df, use_container_width=True)

        if st.button("🚀 Predici Rischio Churn", type="primary"):
            prediction = pipeline.predict(input_df)
            prediction_proba = pipeline.predict_proba(input_df)
            
            st.subheader("Risultato della Predizione:")
            if prediction[0] == 1:
                st.error(f"⚠️ **Alto Rischio di Churn!** (Probabilità: {prediction_proba[0][1]*100:.1f}%)")
            else:
                st.success(f"✅ **Cliente Stabile / Fedele** (Probabilità Churn: {prediction_proba[0][1]*100:.1f}%)")

    with col2:
        st.subheader("💡 Informazioni sul Progetto")
        st.info("""
- **Tech Stack:** Python, Scikit-learn, Streamlit.
- **Deployment:** GitHub & Streamlit Community Cloud.
- **Focus:** Pipeline ML dinamica e riproducibile.
        """)

    # Sezione Dataset di Test
    st.markdown("---")
    st.subheader("📂 Visualizzazione rapida dataset di test")
    if os.path.exists("data/test_data.csv"):
        test_preview = pd.read_csv("data/test_data.csv")
        st.dataframe(test_preview.head(10), use_container_width=True)
    else:
        st.warning("File `test_data.csv` non trovato nella cartella data/.")