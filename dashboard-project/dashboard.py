import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Konfigurasi Halaman
st.set_page_config(
    page_title="Dashboard Prediksi Energi",
    page_icon="⚡",
    layout="wide"
)

# Judul Dashboard
st.title('⚡ Dashboard Prediksi Output Energi Pembangkit Listrik')

# Memuat data (menggunakan cache agar lebih cepat)
@st.cache_data
def load_data():
    data = pd.read_csv('CCPP.csv')
    return data

df = load_data()

# Menampilkan Ringkasan & Hasil
col1, col2 = st.columns(2)

with col1:
    st.header("Ringkasan Data")
    st.write(df.head())

with col2:
    st.header("Hasil Model Terbaik")
    st.metric(label="Akurasi Model (R-squared)", value="93.01%")
    st.write("Model Regresi Linier Berganda terbukti lebih akurat dalam memprediksi output energi.")

st.write("---")

# Menampilkan Visualisasi
st.header('Visualisasi Data Interaktif')

fig, ax = plt.subplots(1, 2, figsize=(20, 8))

# Scatter Plot
sns.scatterplot(x='AT', y='PE', data=df, alpha=0.3, ax=ax[0])
ax[0].set_title('Hubungan antara Temperatur (AT) dan Output Energi (PE)')

# Heatmap
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax[1])
ax[1].set_title('Heatmap Korelasi Antar Fitur')

st.pyplot(fig)