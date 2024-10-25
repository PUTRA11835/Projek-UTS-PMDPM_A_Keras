import streamlit as st
import pandas as pd
import pickle
import os

model_path = r"D:\Semester 5\Kuliah\ML\Tubes"
classification_model_path = os.path.join(model_path, r'BestModel_CLF_GBT_Keras.pkl')
regression_model_path = os.path.join(model_path, r'BestModel_REG_Lasso Regression_Keras.pkl')

if os.path.exists(classification_model_path):
    with open(classification_model_path, 'rb') as f:
        clf_model = pickle.load(f)
else:
    clf_model = None
    st.error(f"Classification model not found at {classification_model_path}")

if os.path.exists(regression_model_path):
    with open(regression_model_path, 'rb') as f:
        reg_model = pickle.load(f)
else:
    reg_model = None
    st.error(f"Regression model not found at {regression_model_path}")

def main():
    st.sidebar.title("Tutorial Desain Streamlit UTS ML 24/25")
    option = st.sidebar.radio("Pilih Model:", ["Klasifikasi", "Regresi"])

    st.title(option)

    uploaded_file = st.file_uploader("Upload dataset (CSV)", type="csv")
    
    if uploaded_file is not None:
        dataset = pd.read_csv(uploaded_file)
        st.write("Dataset yang di-upload:")
        st.dataframe(dataset.head())

    yes_no_map = {"yes": 1, "no": 0}
    new_old_map = {"new": 1, "old": 0}

    if option == "Klasifikasi" and clf_model:
        st.header("Input Data untuk Klasifikasi")

        squaremeters = st.slider("Ukuran properti (squaremeters)", 0, 100000, 50000)
        numberofrooms = st.slider("Jumlah kamar (numberofrooms)", 1, 100, 10)
        hasyard = yes_no_map[st.selectbox("Apakah memiliki halaman? (hasyard)", ["yes", "no"])]
        haspool = yes_no_map[st.selectbox("Apakah memiliki kolam? (haspool)", ["yes", "no"])]
        floors = st.slider("Jumlah lantai (floors)", 1, 100, 1)
        citycode = st.number_input("Kode kota (citycode)", min_value=0, max_value=99999)
        citypartrange = st.slider("Rentang bagian kota (citypartrange)", 0, 10, 5)
        numprevowners = st.slider("Jumlah pemilik sebelumnya (numprevowners)", 0, 10, 1)
        made = st.slider("Tahun dibuat (made)", 1900, 2024, 2000)
        isnewbuilt = new_old_map[st.selectbox("Apakah baru dibangun? (isnewbuilt)", ["new", "old"])]
        hasstormprotector = yes_no_map[st.selectbox("Apakah memiliki pelindung badai? (hasstormprotector)", ["yes", "no"])]
        basement = st.number_input("Luas basement (basement)", min_value=0, max_value=10000)
        attic = st.number_input("Luas loteng (attic)", min_value=0, max_value=10000)
        garage = st.number_input("Luas garasi (garage)", min_value=0, max_value=10000)
        hasstorageroom = yes_no_map[st.selectbox("Apakah memiliki ruang penyimpanan? (hasstorageroom)", ["yes", "no"])]
        hasguestroom = st.slider("Jumlah kamar tamu (hasguestroom)", 0, 20, 1)

        input_data_clf = [[squaremeters, numberofrooms, hasyard, haspool, floors, citycode, citypartrange,
                           numprevowners, made, isnewbuilt, hasstormprotector, basement, attic, garage,
                           hasstorageroom, hasguestroom] + [0] * (21 - 16)]  # Add zero placeholders for missing features
        
        if st.button("Klasifikasi"):
            prediction_clf = clf_model.predict(input_data_clf)
            st.success(f"Kategori Properti: {prediction_clf[0]}")

    elif option == "Regresi" and reg_model:
        st.header("Input Data untuk Regresi")

        squaremeters = st.slider("Ukuran properti (squaremeters)", 0, 100000, 50000)
        numberofrooms = st.slider("Jumlah kamar (numberofrooms)", 1, 100, 10)
        hasyard = yes_no_map[st.selectbox("Apakah memiliki halaman? (hasyard)", ["yes", "no"])]
        haspool = yes_no_map[st.selectbox("Apakah memiliki kolam? (haspool)", ["yes", "no"])]
        floors = st.slider("Jumlah lantai (floors)", 1, 100, 1)
        citycode = st.number_input("Kode kota (citycode)", min_value=0, max_value=99999)
        citypartrange = st.slider("Rentang bagian kota (citypartrange)", 0, 10, 5)
        numprevowners = st.slider("Jumlah pemilik sebelumnya (numprevowners)", 0, 10, 1)
        made = st.slider("Tahun dibuat (made)", 1900, 2024, 2000)
        isnewbuilt = new_old_map[st.selectbox("Apakah baru dibangun? (isnewbuilt)", ["new", "old"])]
        hasstormprotector = yes_no_map[st.selectbox("Apakah memiliki pelindung badai? (hasstormprotector)", ["yes", "no"])]
        basement = st.number_input("Luas basement (basement)", min_value=0, max_value=10000)
        attic = st.number_input("Luas loteng (attic)", min_value=0, max_value=10000)
        garage = st.number_input("Luas garasi (garage)", min_value=0, max_value=10000)
        hasstorageroom = yes_no_map[st.selectbox("Apakah memiliki ruang penyimpanan? (hasstorageroom)", ["yes", "no"])]
        hasguestroom = st.slider("Jumlah kamar tamu (hasguestroom)", 0, 20, 1)

        input_data_reg = [[squaremeters, numberofrooms, hasyard, haspool, floors, citycode, citypartrange,
                           numprevowners, made, isnewbuilt, hasstormprotector, basement, attic, garage,
                           hasstorageroom, hasguestroom] + [0] * (21 - 16)]  # Add zero placeholders for missing features
        
        if st.button("Prediksi Harga"):
            prediction_reg = reg_model.predict(input_data_reg)
            st.success(f"Prediksi Harga Properti: {prediction_reg[0]:,.2f}")

if __name__ == "__main__":
    main()
