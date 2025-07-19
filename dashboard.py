import streamlit as st
import pandas as pd
import plotly.express as px
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
@st.cache_data
def load_data():
    data = pd.read_csv('CCPP.csv')
    return data

df = load_data()

# --- MEMBUAT MODEL ---
features = ['AT', 'V', 'AP', 'RH']
X = df[features]
y = df['PE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_multi = LinearRegression()
model_multi.fit(X_train, y_train)
y_pred = model_multi.predict(X_test)

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚡ Final Project Big Data")
    st.info(
        """
        Dashboard ini memprediksi output energi (PE) dari Pembangkit Listrik 
        berdasarkan variabel lingkungan.
        """
    )
    
    st.header("Anggota Kelompok:")
    st.write("- Nazal Syamaidzar Mahendra (23.11.5547)")
    st.write("- Vendri Setyawan (23.11.5523)")

    st.header("Prediksi Energi Interaktif")
    st.write("Geser slider untuk melihat prediksi output energi (PE) secara real-time.")

    at_input = st.slider('Temperatur (AT)', float(df['AT'].min()), float(df['AT'].max()), float(df['AT'].mean()))
    v_input = st.slider('Vakum (V)', float(df['V'].min()), float(df['V'].max()), float(df['V'].mean()))
    ap_input = st.slider('Tekanan Udara (AP)', float(df['AP'].min()), float(df['AP'].max()), float(df['AP'].mean()))
    rh_input = st.slider('Kelembaban (RH)', float(df['RH'].min()), float(df['RH'].max()), float(df['RH'].mean()))
    
    input_data = [[at_input, v_input, ap_input, rh_input]]
    prediction = model_multi.predict(input_data)
    
    st.subheader("Hasil Prediksi:")
    st.markdown(f"<h2 style='text-align: center; color: #28a745;'>{prediction[0]:.2f} MW</h2>", unsafe_allow_html=True)

# --- HALAMAN UTAMA ---
st.title('Dashboard Analisis & Prediksi Output Energi Pembangkit Listrik')

# Membuat TABS
tab1, tab2, tab3 = st.tabs(["📊 Ringkasan Project", "📈 Eksplorasi Data", "🤖 Kinerja Model"])

with tab1:
    st.header("Ringkasan Project dan Hasil Model")
    st.write(
        """
        Project ini bertujuan untuk membangun model machine learning yang dapat memprediksi output energi
        listrik (PE) berdasarkan empat variabel sensor lingkungan.
        """
    )
    
    st.subheader("Perbandingan Kinerja Model")
    model_comparison = pd.DataFrame({
        "Model": ["Regresi Linier Sederhana", "Regresi Linier Berganda"],
        "R-squared (R²)": [0.9083, 0.9301]
    })
    st.table(model_comparison.style.highlight_max(axis=0, color='#28a745'))

    with st.expander("Kamus Data (Penjelasan Fitur)"):
        st.markdown(
            """
            - **AT**: Ambient Temperature (Temperatur Lingkungan) dalam Celcius.
            - **V**: Exhaust Vacuum (Tekanan Vakum) dalam cm Hg.
            - **AP**: Ambient Pressure (Tekanan Udara) dalam milibar.
            - **RH**: Relative Humidity (Kelembaban Relatif) dalam persen.
            - **PE**: Net hourly electrical energy output (Output Energi Listrik per Jam) dalam MegaWatt. Ini adalah **target prediksi**.
            """
        )
    with st.expander("Klik untuk melihat data mentah"):
        st.dataframe(df)

with tab2:
    st.header("Eksplorasi dan Hubungan Antar Fitur")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribusi Setiap Fitur")
        feature_to_show = st.selectbox(
            'Pilih fitur untuk melihat distribusinya:',
            ('AT', 'V', 'AP', 'RH', 'PE')
        )
        fig_hist = px.histogram(df, x=feature_to_show, nbins=50, title=f'Distribusi Fitur {feature_to_show}')
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        st.subheader("Hubungan Antar Fitur (Scatter Plot)")
        fig_scatter = px.scatter(df, x='AT', y='PE', 
                                 title='Temperatur (AT) vs. Output Energi (PE)',
                                 labels={'AT': 'Temperatur (C)', 'PE': 'Output Energi (MW)'},
                                 trendline="ols")
        st.plotly_chart(fig_scatter, use_container_width=True)

with tab3:
    st.header("Analisis Kinerja Model Regresi Berganda")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Grafik Prediksi vs. Aktual")
        
        # Membuat dataframe untuk perbandingan
        results_df = pd.DataFrame({'Aktual': y_test, 'Prediksi': y_pred})
        
        fig_pred_actual = px.scatter(results_df, x='Aktual', y='Prediksi', 
                                     title='Nilai Aktual vs. Nilai Prediksi Model',
                                     labels={'Aktual': 'Output Energi Aktual (MW)', 'Prediksi': 'Output Energi Prediksi (MW)'},
                                     trendline="ols", trendline_color_override="red")
        st.plotly_chart(fig_pred_actual, use_container_width=True)
        st.caption("Garis merah menunjukkan prediksi yang sempurna. Semakin dekat titik data ke garis, semakin akurat prediksinya.")

    with col2:
        st.subheader("Pentingnya Fitur (Feature Importance)")
        
        # Membuat dataframe koefisien
        coeffs = pd.DataFrame(
            model_multi.coef_,
            features,
            columns=['Koefisien']
        ).sort_values(by='Koefisien', ascending=False)
        
        fig_coeffs = px.bar(coeffs, x=coeffs.index, y='Koefisien', 
                            title='Pengaruh Setiap Fitur Terhadap Output Energi',
                            labels={'index': 'Fitur', 'Koefisien': 'Besar Pengaruh (Koefisien)'})
        st.plotly_chart(fig_coeffs, use_container_width=True)
        st.caption("Koefisien negatif (AT dan V) berarti jika nilainya naik, output energi (PE) cenderung turun. Koefisien positif (AP dan RH) berarti sebaliknya.")