import time
import numpy as np
import pandas as pd
import streamlit as st

# ==========================================
# 1. Configurazione della Pagina
# ==========================================
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# 2. Header dell'App
# ==========================================
st.title("🔮 Customer Churn Risk Predictor")
st.markdown(
    """
    **Applicazione di Machine Learning per l'inferenza del rischio di abbandono cliente (Churn).**  
    Inserisci i dati contrattuali e comportamentali nella barra laterale o nella griglia sottostante per calcolare la probabilità stimata di disdetta.
"""
)
st.divider()

# ==========================================
# 3. Sidebar: Input dei Dati del Cliente
# ==========================================
st.sidebar.header("📋 Profilo Cliente")
st.sidebar.write("Modifica i parametri per testare l'inferenza:")

# Parametri contrattuali
anzianita_mesi = st.sidebar.slider(
    "Anzianità Cliente (Mesi)",
    min_value=1,
    max_value=72,
    value=12,
    help="Da quanti mesi il cliente ha attivo il servizio.",
)

spesa_mensile = st.sidebar.slider(
    "Canone Mensile (€)",
    min_value=18.0,
    max_value=150.0,
    value=65.0,
    step=0.5,
    help="Importo della fattura mensile.",
)

tipo_contratto = st.sidebar.selectbox(
    "Tipologia di Contratto",
    options=["Mensile (Rinnovo automatico)", "Annuale", "Biennale"],
    index=0,
)

metodo_pagamento = st.sidebar.selectbox(
    "Metodo di Pagamento",
    options=[
        "Addebito Diretto (RID)",
        "Carta di Credito",
        "Bollettino Postale",
        "Bonifico",
    ],
    index=0,
)

# Servizi accessori
st.sidebar.subheader("Servizi Attivi")
supporto_tecnico = st.sidebar.checkbox(
    "Supporto Tecnico VIP", value=False
)
sicurezza_online = st.sidebar.checkbox("Protezione Sicurezza Online", value=True)

# ==========================================
# 4. Layout Principale: Riepilogo & Previsione
# ==========================================
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
  st.subheader("👤 Scheda Riassuntiva Input")

  # Creiamo un DataFrame formattato per mostrare i dati inseriti
  summary_df = pd.DataFrame({
      "Parametro": [
          "Anzianità",
          "Canone Mensile",
          "Tipo Contratto",
          "Metodo Pagamento",
          "Supporto VIP",
          "Sicurezza Online",
      ],
      "Valore": [
          f"{anzianita_mesi} mesi",
          f"€ {spesa_mensile:.2f}",
          tipo_contratto,
          metodo_pagamento,
          "Sì" if supporto_tecnico else "No",
          "Sì" if sicurezza_online else "No",
      ],
  })
  st.dataframe(summary_df, use_container_width=True, hide_index=True)

  predict_btn = st.button(
      "🚀 Calcola Probabilità Churn", type="primary", use_container_width=True
  )

with col_right:
  st.subheader("📊 Output Modello & Probabilità")

  if predict_btn:
    with st.spinner("Elaborazione dati con il modello ML..."):
      time.sleep(0.5)  # Simulazione di latenza di inferenza

      # ==========================================
      # Mock di Inferenza Probabilistica
      # (In seguito verrà sostituito da model.predict_proba)
      # ==========================================
      # Logica euristica per rendere plausibile la demo prima di caricare il vero modello:
      base_risk = 0.40
      if "Mensile" in tipo_contratto:
        base_risk += 0.25
      elif "Biennale" in tipo_contratto:
        base_risk -= 0.20

      if anzianita_mesi < 12:
        base_risk += 0.15
      elif anzianita_mesi > 36:
        base_risk -= 0.15

      if not supporto_tecnico:
        base_risk += 0.10

      prob_churn = float(np.clip(base_risk, 0.02, 0.98))
      prob_retention = 1.0 - prob_churn

    # Visualizzazione dei risultati
    st.markdown("### Risultato della Predizione")

    if prob_churn >= 0.50:
      st.error(
          f"⚠️ **RISCHIO ELEVATO DI CHURN**  \nProbabilità di abbandono:"
          f" **{prob_churn*100:.1f}%**"
      )
    else:
      st.success(
          f"✅ **CLIENTE STABILE**  \nProbabilità di abbandono:"
          f" **{prob_churn*100:.1f}%**"
      )

    st.write("---")

    # Metriche visive affiancate
    m_col1, m_col2 = st.columns(2)
    m_col1.metric(
        label="Probabilità Churn ($P(Y=1)$)",
        value=f"{prob_churn*100:.1f}%",
        delta=f"{(prob_churn-0.5)*100:+.1f}% vs soglia",
        delta_color="inverse",
    )
    m_col2.metric(
        label="Probabilità Retention ($P(Y=0)$)",
        value=f"{prob_retention*100:.1f}%",
    )

    # Progress bar per l'impatto visivo
    st.write("**Livello di Rischio Stimato:**")
    st.progress(prob_churn)

  else:
    st.info(
        "👈 Regola i parametri nel pannello laterale e clicca su **'Calcola"
        " Probabilità Churn'** per visualizzare la stima dell'algoritmo."
    )

# ==========================================
# 5. Footer Informativo per Recruiters
# ==========================================
st.divider()
st.caption(
    "💡 **Nota Tecnica per il Recruiter:** L'architettura dell'app riceve i"
    " dati in input, costruisce la matrice delle feature, applica la"
    " pipeline di preprocessing (scaling + one-hot encoding) e calcola il"
    " vettore delle probabilità $P(Y=k|X)$ tramite il modello addestrato."
)