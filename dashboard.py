import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Dashboard Prediksi Energi",
    page_icon="⚡",
    layout="wide"
)

# --- MEMUAT DATA ---
# Menggunakan cache agar pemrosesan data lebih cepat
@st.cache_data
def load_data():
    data = pd.read_csv('CCPP.csv')
    return data

df = load_data()

# --- MEMBUAT MODEL ---
# Model ini dibuat di sini agar bisa digunakan untuk prediksi interaktif
features = ['AT', 'V', 'AP', 'RH']
X = df[features]
y = df['PE']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_multi = LinearRegression()
model_multi.fit(X_train, y_train)

# --- SIDEBAR ---
with st.sidebar:
    st.title("Final Project Big Data")
    st.info("Dashboard ini dibuat untuk memprediksi output energi (PE) dari sebuah Pembangkit Listrik Tenaga Gas dan Uap (PLTGU) berdasarkan beberapa variabel lingkungan.")
    
    st.header("Anggota Kelompok:")
    st.write("- Nazal Syamaidzar Mahendra (23.11.5547)")
    st.write("- Vendri Setyawan (23.11.5523)")

    st.header("Prediksi Interaktif")
    st.write("Geser slider untuk melihat prediksi output energi (PE):")

    # Input dari pengguna
    at_input = st.slider('Temperatur (AT)', float(df['AT'].min()), float(df['AT'].max()), float(df['AT'].mean()))
    v_input = st.slider('Vakum (V)', float(df['V'].min()), float(df['V'].max()), float(df['V'].mean()))
    ap_input = st.slider('Tekanan Udara (AP)', float(df['AP'].min()), float(df['AP'].max()), float(df['AP'].mean()))
    rh_input = st.slider('Kelembaban (RH)', float(df['RH'].min()), float(df['RH'].max()), float(df['RH'].mean()))
    
    # Prediksi
    input_data = [[at_input, v_input, ap_input, rh_input]]
    prediction = model_multi.predict(input_data)
    
    st.subheader("Hasil Prediksi:")
    st.success(f"{prediction[0]:.2f} MW")

# --- HALAMAN UTAMA ---
st.title('⚡ Dashboard Analisis & Prediksi Output Energi')

# Ringkasan Data
with st.expander("Klik untuk melihat ringkasan data mentah"):
    st.write(df.head())
    st.write("Statistik Deskriptif:")
    st.write(df.describe())

st.write("---")

# Hasil & Evaluasi Model
st.header("Hasil dan Evaluasi Model")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Perbandingan Kinerja Model")
    # Membuat tabel perbandingan
    model_comparison = pd.DataFrame({
        "Model": ["Regresi Linier Sederhana", "Regresi Linier Berganda"],
        "R-squared (R²)": [0.9083, 0.9301],
        "Mean Squared Error (MSE)": [28.91, 20.27]
    })
    st.table(model_comparison)
with col2:
    st.subheader("Kesimpulan")
    st.write(
        """
        - Model **Regresi Linier Berganda** terbukti lebih unggul.
        - Dengan R-squared **0.9301**, model ini mampu menjelaskan 93% variasi data output energi.
        - Ini menunjukkan bahwa kombinasi semua fitur lingkungan memberikan prediksi yang jauh lebih akurat.
        """
    )

st.write("---")

# Visualisasi Data
st.header('Visualisasi Data')
fig, ax = plt.subplots(1, 2, figsize=(20, 8))

# Scatter Plot
sns.scatterplot(x='AT', y='PE', data=df, alpha=0.3, ax=ax[0])
ax[0].set_title('Hubungan Temperatur (AT) vs Output Energi (PE)')

# Heatmap
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax[1])
ax[1].set_title('Heatmap Korelasi Antar Fitur')

st.pyplot(fig)