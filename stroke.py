import streamlit as st
import tensorflow as tf
import numpy as np
import joblib

# Memuat model TFLite
interpreter = tf.lite.Interpreter(model_path="stroke_model.tflite")
interpreter.allocate_tensors()

# Load scaler dari file
scaler = joblib.load("scaler.pkl")

# Melakukan prediksi
def predict(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Input tensor untuk model
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # Mengambil hasil prediksi
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Judul aplikasi Streamlit
st.title("🧠 Prediksi Risiko Stroke")

# Menambahkan form untuk input data
st.sidebar.header("📝 Masukkan Data Pasien")

# Form untuk memasukkan input
gender = st.sidebar.selectbox("Jenis Kelamin", ['Laki-laki', 'Perempuan'])
age = st.sidebar.slider("Usia", 0, 100, 25)
hypertension = st.sidebar.selectbox("Apakah memiliki hipertensi?", ['Ya', 'Tidak'])
heart_disease = st.sidebar.selectbox("Apakah memiliki penyakit jantung?", ['Ya', 'Tidak'])
ever_married = st.sidebar.selectbox("Pernah menikah?", ['Ya', 'Tidak'])
work_type = st.sidebar.selectbox("Jenis Pekerjaan", ['Swasta', 'Wiraswasta', 'Pekerja Negeri', 'Anak-anak', 'Tidak Bekerja'])
residence_type = st.sidebar.selectbox("Jenis Tempat Tinggal", ['Perkotaan', 'Pedesaan'])
avg_glucose_level = st.sidebar.slider("Rata-rata Tingkat Glukosa", 0.0, 300.0, 100.0)
bmi = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)
smoking_status = st.sidebar.selectbox("Status Merokok", ['Tidak Merokok', 'Merokok', 'Pernah Merokok'])

# Input data dalam format yang diperlukan oleh model
input_data = np.array([[gender == 'Laki-laki', age, hypertension == 'Ya', heart_disease == 'Ya', ever_married == 'Ya',
                        work_type == 'Swasta', residence_type == 'Perkotaan', avg_glucose_level, bmi, smoking_status == 'Merokok']])

# Mengubah input_data menjadi float32 sesuai model TFLite
input_data = input_data.astype(np.float32)

# Prediksi jika tombol dipencet
if st.sidebar.button("🔍 Prediksi"):
    prediction = predict(interpreter, input_data)
    prob = prediction[0][0]

    # Tentukan nilai berdasarkan probabilitas
    if prob > 0.3:
        bg_color = "#ffe6e6"
        text_color = "red"
        message = "⚠️ Pasien Diprediksi <b>TERKENA</b> Stroke"
    else:
        bg_color = "#e6ffed"
        text_color = "green"
        message = "✅ Pasien Diprediksi <b>TIDAK</b> Terkena Stroke"

    # Tampilkan hasil prediksi dengan style
    st.markdown(
        f"""
        <div style='background-color: {bg_color}; padding: 15px; border-radius: 8px'>
            <p style='color: {text_color}; font-size:18px;'>{message}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Tampilkan probabilitas biasa
    st.write(f"📊 Probabilitas Stroke: {prob:.4f}")
