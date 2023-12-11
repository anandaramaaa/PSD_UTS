from audioop import minmax
import streamlit as st
import matplotlib.pyplot as plt
import IPython
import numpy as np
import scipy.stats
import seaborn as sns
import pandas as pd
import os
import pickle
import streamlit as st
import librosa
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from scipy.stats import skew, kurtosis, mode, iqr

st.set_page_config(
    page_title="Extraksi Data Audio"
)
st.write("""
Nama : Ananda Ramadana Ahmad Mulya\n
NIM : 210411100135
""")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Prediksi", "Dataset", "Ekstraksi Fitur", "Preprocessing", "Akurasi", "Reduksi PCA", "Grid Search"])

with tab1:
    st.write("""
    <h1>Prediksi Data Audio </h1>
    <br>
    """, unsafe_allow_html=True)
    def calculate_statistics(audio_path):
        x, sr = librosa.load(audio_path)

        mean = np.mean(x)
        std = np.std(x)
        maxv = np.amax(x)
        minv = np.amin(x)
        median = np.median(x)
        skewness = skew(x)
        kurt = kurtosis(x)
        q1 = np.quantile(x, 0.25)
        q3 = np.quantile(x, 0.75)
        mode_v = mode(x)[0]
        iqr = q3 - q1

        zcr = librosa.feature.zero_crossing_rate(x)
        mean_zcr = np.mean(zcr)
        median_zcr = np.median(zcr)
        std_zcr = np.std(zcr)
        kurtosis_zcr = kurtosis(zcr, axis=None)
        skew_zcr = skew(zcr, axis=None)

        n = len(x)
        mean_rms = np.sqrt(np.mean(x**2) / n)
        median_rms = np.sqrt(np.median(x**2) / n)
        skew_rms = np.sqrt(skew(x**2) / n)
        kurtosis_rms = np.sqrt(kurtosis(x**2) / n)
        std_rms = np.sqrt(np.std(x**2) / n)

        return [mean, median, mode_v, maxv, minv, std, skewness, kurt, q1, q3, iqr, mean_zcr, median_zcr, std_zcr, kurtosis_zcr, skew_zcr, mean_rms, median_rms, std_rms, kurtosis_rms, skew_rms]

    uploaded_file = st.file_uploader("Pilih file audio...", type=["wav","mp3"])
    
    scaler = st.radio(
    "Prediksi Class Data Audio",
    ('Prediksi Z-Score', 'Prediksi MinMax'))

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        if scaler == 'Prediksi Z-Score':
            st.title("Prediksi Class Data Audio Menggunakan Z-Score")

            if st.button("Cek Nilai Statistik"):
                # Simpan file audio yang diunggah
                audio_path = "audio_diunggah.wav"
                with open(audio_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Hitung statistik untuk file audio yang diunggah
                statistik = calculate_statistics(audio_path)

                results = []
                result = {
                    'Audio Mean': statistik[0],
                    'Audio Median': statistik[1],
                    'Audio Mode': statistik[2],
                    'Audio Maxv': statistik[3],
                    'Audio Minv': statistik[4],
                    'Audio Std': statistik[5],
                    'Audio Skew': statistik[6],
                    'Audio Kurtosis': statistik[7],
                    'Audio Q1': statistik[8],
                    'Audio Q3': statistik[9],
                    'Audio IQR': statistik[10],
                    'ZCR Mean': statistik[11],
                    'ZCR Median': statistik[12],
                    'ZCR Std': statistik[13],
                    'ZCR Kurtosis': statistik[14],
                    'ZCR Skew': statistik[15],
                    'RMS Energi Mean': statistik[16],
                    'RMS Energi Median': statistik[17],
                    'RMS Energi Std': statistik[18],
                    'RMS Energi Kurtosis': statistik[19],
                    'RMS Energi Skew': statistik[20],
                }
                results.append(result)
                df = pd.DataFrame(results)
                st.write(df)

                # Hapus file audio yang diunggah
                os.remove(audio_path)

            if st.button("Deteksi Audio"):

                # Memuat data audio yang diunggah dan menyimpannya sebagai file audio
                audio_path = "audio_diunggah.wav"
                with open(audio_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Menghitung statistik untuk file audio yang diunggah (gunakan fungsi calculate_statistics sesuai kebutuhan)
                audio_features = calculate_statistics(audio_path)
                results = []
                result = {
                    'Audio Mean': audio_features[0],
                    'Audio Median': audio_features[1],
                    'Audio Mode': audio_features[2],
                    'Audio Maxv': audio_features[3],
                    'Audio Minv': audio_features[4],
                    'Audio Std': audio_features[5],
                    'Audio Skew': audio_features[6],
                    'Audio Kurtosis': audio_features[7],
                    'Audio Q1': audio_features[8],
                    'Audio Q3': audio_features[9],
                    'Audio IQR': audio_features[10],
                    'ZCR Mean': audio_features[11],
                    'ZCR Median': audio_features[12],
                    'ZCR Std': audio_features[13],
                    'ZCR Kurtosis': audio_features[14],
                    'ZCR Skew': audio_features[15],
                    'RMS Energi Mean': audio_features[16],
                    'RMS Energi Median': audio_features[17],
                    'RMS Energi Std': audio_features[18],
                    'RMS Energi Kurtosis': audio_features[19],
                    'RMS Energi Skew': audio_features[20],
                }
                results.append(result)
                data_tes = pd.DataFrame(results)

                df = pd.read_csv('hasil_data.csv')

                # Memisahkan kolom target (label) dari kolom fitur
                X = df.drop(columns=['Label'])  # Kolom fitur
                y = df['Label']  # Kolom target

                # Normalisasi data menggunakan StandardScaler
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # Memisahkan data menjadi data latih dan data uji
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

                # Load the model and hyperparameters
                with open('gridsearchknnminmaxmodel.pkl', 'rb') as model_file:
                    saved_data = pickle.load(model_file)

                # Access hyperparameters
                best_n_neighbors = saved_data['hyperparameters']['best_n_neighbors']
                best_weights = saved_data['hyperparameters']['best_weights']
                best_metric = saved_data['hyperparameters']['best_metric']
                best_comp = saved_data['hyperparameters']['best_comp']

                # Melakukan PCA pada data audio yang diunggah
                pca = PCA(n_components=best_comp)

                # Memanggil metode fit dengan data pelatihan sebelum menggunakan transform
                zscore_scaler = StandardScaler()
                X_test_zscore = zscore_scaler.fit_transform(data_tes)

                X_train_pca = pca.fit_transform(X_train)
                X_test_pca = pca.transform(X_test_zscore)

                # Membuat model KNN dengan hyperparameter terbaik
                best_knn_model = KNeighborsClassifier(n_neighbors=best_n_neighbors, weights=best_weights, metric=best_metric)
                best_knn_model.fit(X_train_pca, y_train)
                    
                predicted_label = best_knn_model.predict(X_test_pca)

                # Menampilkan hasil prediksi
                st.write("Emosi Terdeteksi:", predicted_label)

                # Menghapus file audio yang diunggah
                os.remove(audio_path)

        elif scaler == 'Prediksi MinMax':
            st.title("Prediksi Class Data Audio Menggunakan MinMax")

            if st.button("Cek Nilai Statistik"):
                # Simpan file audio yang diunggah
                audio_path = "audio_diunggah.wav"
                with open(audio_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Hitung statistik untuk file audio yang diunggah
                statistik = calculate_statistics(audio_path)

                results = []
                result = {
                    'Audio Mean': statistik[0],
                    'Audio Median': statistik[1],
                    'Audio Mode': statistik[2],
                    'Audio Maxv': statistik[3],
                    'Audio Minv': statistik[4],
                    'Audio Std': statistik[5],
                    'Audio Skew': statistik[6],
                    'Audio Kurtosis': statistik[7],
                    'Audio Q1': statistik[8],
                    'Audio Q3': statistik[9],
                    'Audio IQR': statistik[10],
                    'ZCR Mean': statistik[11],
                    'ZCR Median': statistik[12],
                    'ZCR Std': statistik[13],
                    'ZCR Kurtosis': statistik[14],
                    'ZCR Skew': statistik[15],
                    'RMS Energi Mean': statistik[16],
                    'RMS Energi Median': statistik[17],
                    'RMS Energi Std': statistik[18],
                    'RMS Energi Kurtosis': statistik[19],
                    'RMS Energi Skew': statistik[20],
                }
                results.append(result)
                df = pd.DataFrame(results)
                st.write(df)

                # Hapus file audio yang diunggah
                os.remove(audio_path)

            if st.button("Deteksi Audio"):

                # Memuat data audio yang diunggah dan menyimpannya sebagai file audio
                audio_path = "audio_diunggah.wav"
                with open(audio_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Menghitung statistik untuk file audio yang diunggah (gunakan fungsi calculate_statistics sesuai kebutuhan)
                audio_features = calculate_statistics(audio_path)
                results = []
                result = {
                    'Audio Mean': audio_features[0],
                    'Audio Median': audio_features[1],
                    'Audio Mode': audio_features[2],
                    'Audio Maxv': audio_features[3],
                    'Audio Minv': audio_features[4],
                    'Audio Std': audio_features[5],
                    'Audio Skew': audio_features[6],
                    'Audio Kurtosis': audio_features[7],
                    'Audio Q1': audio_features[8],
                    'Audio Q3': audio_features[9],
                    'Audio IQR': audio_features[10],
                    'ZCR Mean': audio_features[11],
                    'ZCR Median': audio_features[12],
                    'ZCR Std': audio_features[13],
                    'ZCR Kurtosis': audio_features[14],
                    'ZCR Skew': audio_features[15],
                    'RMS Energi Mean': audio_features[16],
                    'RMS Energi Median': audio_features[17],
                    'RMS Energi Std': audio_features[18],
                    'RMS Energi Kurtosis': audio_features[19],
                    'RMS Energi Skew': audio_features[20],
                }
                results.append(result)
                data_tes = pd.DataFrame(results)

                # Load the model and hyperparameters
                with open('gridsearchknnminmaxmodel.pkl', 'rb') as model_file:
                    saved_data = pickle.load(model_file)

                df = pd.read_csv('hasil_statistik2.csv')

                # Memisahkan kolom target (label) dari kolom fitur
                X = df.drop(columns=['Label'])  # Kolom fitur
                y = df['Label']  # Kolom target

                # Normalisasi data menggunakan StandardScaler
                scaler = MinMaxScaler()
                X_scaled = scaler.fit_transform(X)

                # Memisahkan data menjadi data latih dan data uji
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

                # Access hyperparameters
                best_n_neighbors = saved_data['hyperparameters']['best_n_neighbors']
                best_weights = saved_data['hyperparameters']['best_weights']
                best_metric = saved_data['hyperparameters']['best_metric']
                best_comp = saved_data['hyperparameters']['best_comp']

                # Melakukan PCA pada data audio yang diunggah
                pca = PCA(n_components=best_comp)

                # Memanggil metode fit dengan data pelatihan sebelum menggunakan transform
                minmax_scaler = MinMaxScaler()
                X_test_minmax = minmax_scaler.fit_transform(data_tes)

                X_train_pca = pca.fit_transform(X_train)
                X_test_pca = pca.transform(X_test_minmax)

                # Membuat model KNN dengan hyperparameter terbaik
                best_knn_model = KNeighborsClassifier(n_neighbors=best_n_neighbors, weights=best_weights, metric=best_metric)
                best_knn_model.fit(X_train_pca, y_train)
                    
                predicted_label = best_knn_model.predict(X_test_pca)

                # Menampilkan hasil prediksi
                st.write("Emosi Terdeteksi:", predicted_label)


                # Menghapus file audio yang diunggah
                os.remove(audio_path)


with tab2:
    st.write("""
    <h1>Dataset </h1>
    <br>
    """, unsafe_allow_html=True)


    st.write("""<h3> Toronto emotional speech set (TESS) <h3>""", unsafe_allow_html=True)
    st.write("""<h5> A dataset for training emotion (7 cardinal emotions) classification in audio <>""", unsafe_allow_html=True)

    st.markdown("""
        <div style="text-align: justify">
            Dataset audio ini saya ambil dari kaggle yang dikumpulkan oleh University of Toronto dengan berkolaborasi bersama Eu Jin Lok sebagai author. Dataset ini di publis pada tahun 2019.\n Untuk bisa mengakses datasetnya bisa klik link :
            <a href="https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess?resource=download">Disini</a>
        </div>""", unsafe_allow_html=True)


    st.write(
        """
        <div style="text-align: justify">
            Dataset ini berisi serangkaian 200 kata target yang diucapkan dalam kalimat "Katakan kata.." oleh dua 
            aktris perempuan (berusia 26 dan 64 tahun), dan rekaman dibuat dari setiap kata tersebut dengan tujuh emosi yang berbeda yakni 
            (marah, jijik, takut, bahagia, kejutan menyenangkan, sedih, dan netral). Total ada 2800 data (berupa file audio) 
            dalam format WAV.
            Maksud dari kalimat "Katakan kata .. ", adalah kalimat ini digunakan sebagai format standar di mana dua aktris 
            perempuan diminta untuk mengucapkan 200 kata target. Dengan demikian, kalimat ini bertindak sebagai 
            pola yang memandu cara kata-kata tersebut diucapkan dalam rekaman audio. Bagian ".." dalam kalimat tersebut menunjukkan tempat 
            di mana kata target akan dimasukkan. Misalnya, jika kata target adalah "apple", kalimat yang diucapkan akan menjadi "Katakan kata apple." 
            Ini memastikan konsistensi dalam pengucapan kata-kata target selama percobaan atau penelitian.
        </div>
        """, unsafe_allow_html=True)

    

    # image = open('Folder Data Audio.png', 'rb').read()
    # st.image(image, caption='Dataset Audio', use_column_width=True)



with tab3:
    st.write("""
    <h1>Ekstraksi Fitur </h1>
    <br>
    """, unsafe_allow_html=True)

    st.write("File audio akan dicari statistiknya. Fitur-fitur yang diekstraksi termasuk statistik dari frekuensi, statistik Zero Crossing Rate (ZCR), dan RMS (Root Mean Square) energi signal dari setiap file audio.")
    

    df = pd.read_csv("hasil_statistik2.csv")
    st.write("Hasil Ekstraksi Fitur Ciri Audio : ")
    st.write(df)
    st.write("Penjelasan fitur-fitur yang ada")

    st.write("""
        <ol>
            <li>Audio Mean : Rata-rata dari nilai amplitudo dalam sinyal audio. Ini menggambarkan tingkat kekuatan sinyal audio secara keseluruhan.</li>
            <li>Audio Median : Nilai mediannya dari amplitudo dalam sinyal audio. Median adalah nilai tengah dalam distribusi data.</li>
            <li>Audio Mode : Nilai yang paling sering muncul dalam sinyal audio.</li>
            <li>Audio Maxv : Nilai amplitudo maksimum dalam sinyal audio, menunjukkan puncak tertinggi dari sinyal tersebut.</li>
            <li>Audio Minv : Nilai amplitudo minimum dalam sinyal audio, menunjukkan puncak terendah dari sinyal tersebut. </li>
            <li>Audio Std : Deviasi standar dari amplitudo dalam sinyal audio, mengukur sejauh mana nilai-nilai amplitudo tersebar dari nilai rata-rata.</li>
            <li>Audio Skew : mengukur sejauh mana distribusi amplitudo dalam suara melengkung dari distribusi yang simetris. semakin padat sebaran (mengelompok) maka standar deviasinya dan variansinya rendah, semakin renggang sebaran (menyebar) maka standar deviasinya dan variansinya tinggi.</li>
            <li>Audio Kurtosis : Mengukur tingkat ketajaman puncak dalam distribusi amplitudo. semakin kecil nilai kurtosis maka grafik semakin landai. semakin besar nilai kurtosis maka grafik semakin meruncing</li>
            <li>Audio Q1 : (Kuartil Pertama) adalah nilai yang membagi bagian bawah 25% data terendah ketika data telah diurutkan.</li>
            <li>Audio Q3 : (Kuartil Ketiga) adalah nilai yang membagi bagian atas 25% data tertinggi ketika data telah diurutkan. Secara matematis, Q3 adalah median dari setengah bagian kedua dari data.</li>
            <li>Audio IQR : adalah selisih antara Q3 dan Q1.</li>
            <li>ZCR Mean : Rata-rata dari tingkat perubahan nol-crossing rate (ZCR) dalam sinyal audio. ZCR mengukur sejauh mana sinyal audio melintasi nilai nol.</li>
            <li>ZCR Median : Median dari tingkat perubahan nol-crossing rate (ZCR) dalam sinyal audio. ZCR mengukur sejauh mana sinyal audio melintasi nilai nol.</li>
            <li>ZCR std : Standar devisiasi dari tingkat perubahan nol-crossing rate (ZCR) dalam sinyal audio. ZCR mengukur sejauh mana sinyal audio melintasi nilai nol. </li>
            <li>ZCR Kurtosis : Kurtosis dari tingkat perubahan nol-crossing rate (ZCR) dalam sinyal audio. ZCR mengukur sejauh mana sinyal audio melintasi nilai nol.</li>
            <li>ZCR Skew : Skewness dari tingkat perubahan nol-crossing rate (ZCR) dalam sinyal audio. ZCR mengukur sejauh mana sinyal audio melintasi nilai nol.</li>
            <li>RMS Energi Mean : Rata-rata dari energi root mean square (RMS) dalam sinyal audio. RMS mengukur kekuatan rata-rata sinyal audio.</li>
            <li>RMS Energi Median : Median dari energi root mean square (RMS) dalam sinyal audio. RMS mengukur kekuatan rata-rata sinyal audio.</li>
            <li>RMS Energi std : Standar Devisiasi dari energi root mean square (RMS) dalam sinyal audio. RMS mengukur kekuatan rata-rata sinyal audio.</li>
            <li>RMS Energi kurtosis : Kurtosis dari energi root mean square (RMS) dalam sinyal audio. RMS mengukur kekuatan rata-rata sinyal audio.</li>
            <li>RMS Energi skew : Skewness dari energi root mean square (RMS) dalam sinyal audio. RMS mengukur kekuatan rata-rata sinyal audio.</li>
            <li>Label : Label atau kategori yang menunjukkan emosi atau klasifikasi lain dari sinyal audio, seperti marah, senang, sedih, dll.</li>
        </ol>
    """,unsafe_allow_html=True)





with tab4:
    st.title("Preprocessing Data")
    
    scaler = st.radio(
    "Metode normalisasi data",
    ('Tanpa Normalisasi Data', 'Zscore Scaler', 'MinMax Scaler'))
    if scaler == 'Tanpa Normalisasi Data':
        st.title("Dataset Tanpa Preprocessing : ")
        df = pd.read_csv('hasil_statistik2.csv')
        df
    elif scaler == 'Zscore Scaler':
        st.title('Hasil Normalisasi Menggunakan Z-score')

        # Membaca data dari file CSV
        df = pd.read_csv('hasil_statistik2.csv')

        # Memisahkan kolom target (label) dari kolom fitur
        X = df.drop(columns=['Label'])  # Kolom fitur
        y = df['Label']  # Kolom target

        # Membagi data menjadi data training dan data testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Normalisasi Z-score pada data training
        zscore_scaler = StandardScaler()
        X_train_zscore = zscore_scaler.fit_transform(X_train)


        # Simpan hasil normalisasi data training ke dalam file pickle
        with open('zscore_scaler.pkl', 'wb') as scaler_file:
            pickle.dump(X_train_zscore, scaler_file)

        # Membuat DataFrame dari data hasil normalisasi Z-score
        normalized_data_zscore = pd.DataFrame(X_train_zscore, columns=X_train.columns)

        # Menambahkan kolom target (label) ke DataFrame hasil normalisasi Z-score
        normalized_data_zscore['Label'] = y_train

        # Menampilkan data hasil normalisasi Z-score
        print("Data Hasil Normalisasi Z-score:")
        normalized_data_zscore

    elif scaler == 'MinMax Scaler':
        st.title('Hasil Normalisasi Menggunakan Min-Max')

        # Membaca data dari file CSV
        df = pd.read_csv('hasil_statistik2.csv')

        # Memisahkan kolom target (label) dari kolom fitur
        X = df.drop(columns=['Label'])  # Kolom fitur
        y = df['Label']  # Kolom target

        # Membagi data menjadi data training dan data testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Normalisasi Min-Max Scaling pada data training
        minmax_scaler = MinMaxScaler()
        X_train_minmax = minmax_scaler.fit_transform(X_train)

        # Simpan skaler ke dalam file menggunakan pickle
        with open('minmax_scaler.pkl', 'wb') as scaler_file:
            pickle.dump(X_train_minmax, scaler_file)

        # Membuat DataFrame dari data hasil normalisasi Min-Max Scaling
        normalized_data_minmax = pd.DataFrame(X_train_minmax, columns=X_train.columns)

        # Menambahkan kolom target (label) ke DataFrame hasil normalisasi Min-Max Scaling
        normalized_data_minmax['Label'] = y_train

        # Menampilkan data hasil normalisasi Min-Max Scaling
        print("\nData Hasil Normalisasi Min-Max Scaling:")
        normalized_data_minmax



with tab5:
    st.title("Modeling KNN")

    # Membaca data dari file CSV
    df = pd.read_csv('hasil_statistik2.csv')

    # Memisahkan kolom target (label) dari kolom fitur
    X = df.drop(columns=['Label'])  # Kolom fitur
    y = df['Label']  # Kolom target

    # Membagi data menjadi data training dan data testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalisasi Z-score pada data training
    zscore_scaler = StandardScaler()
    X_train_zscore = zscore_scaler.fit_transform(X_train)

    # Normalisasi Min-Max pada data training
    minmax_scaler = MinMaxScaler()
    X_train_minmax = minmax_scaler.fit_transform(X_train)

    # Membuat dictionary untuk menyimpan akurasi untuk setiap nilai k dari 1 hingga 100
    accuracy_dict_zscore = {}
    accuracy_dict_minmax = {}

    # Melakukan loop untuk mencoba setiap nilai k dari 1 hingga 100
    for k in range(1, 101):
        # Mendefinisikan dan melatih model KNN dengan Z-score
        knn_model_zscore = KNeighborsClassifier(n_neighbors=k)
        knn_model_zscore.fit(X_train_zscore, y_train)

        # Melakukan prediksi pada data testing yang telah dinormalisasi dengan Z-score
        X_test_zscore = zscore_scaler.transform(X_test)
        y_pred_zscore = knn_model_zscore.predict(X_test_zscore)

        # Mengukur akurasi model dengan Z-score
        accuracy_zscore = accuracy_score(y_test, y_pred_zscore)
        accuracy_dict_zscore[k] = accuracy_zscore

        # Mendefinisikan dan melatih model KNN dengan Min-Max Scaling
        knn_model_minmax = KNeighborsClassifier(n_neighbors=k)
        knn_model_minmax.fit(X_train_minmax, y_train)

        # Melakukan prediksi pada data testing yang telah dinormalisasi dengan Min-Max Scaling
        X_test_minmax = minmax_scaler.transform(X_test)
        y_pred_minmax = knn_model_minmax.predict(X_test_minmax)

        # Mengukur akurasi model dengan Min-Max Scaling
        accuracy_minmax = accuracy_score(y_test, y_pred_minmax)
        accuracy_dict_minmax[k] = accuracy_minmax

    # Membuat grafik perbandingan akurasi Z-score dan Min-Max Scaling
    plt.figure(figsize=(10, 6))
    plt.plot(list(accuracy_dict_zscore.keys()), list(accuracy_dict_zscore.values()), label='Z-score')
    plt.plot(list(accuracy_dict_minmax.keys()), list(accuracy_dict_minmax.values()), label='Min-Max')
    plt.xlabel('k Value')
    plt.ylabel('Accuracy')
    plt.title('Perbandingan Akurasi nilai K menggunakan Z-score dan Min-Max Scaling')
    plt.legend()
    plt.grid(True)

    # # Menyimpan grafik ke dalam file gambar
    # plt.savefig('Perbandingan Akurasi MinMax dan Zscore.png')

    # # Menampilkan grafik menggunakan Streamlit
    # st.image('accuracy_comparison.png', use_column_width=True)

    # Menampilkan nilai k terbaik dan akurasinya untuk Z-score
    best_k_zscore = max(accuracy_dict_zscore, key=accuracy_dict_zscore.get)
    st.write(f"Best k value with Z-score : {best_k_zscore}, Accuracy : {accuracy_dict_zscore[best_k_zscore]*100:.2f}%")

    # Menampilkan nilai k terbaik dan akurasinya untuk Min-Max Scaling
    best_k_minmax = max(accuracy_dict_minmax, key=accuracy_dict_minmax.get)
    st.write(f"Best k value with Min-Max : {best_k_minmax}, Accuracy : {accuracy_dict_minmax[best_k_minmax]*100:.2f}%\n\n\n")
    st.write("<br><br><br>", unsafe_allow_html=True)
    
    st.markdown("<h2>Akurasi Model</h2>", unsafe_allow_html=True)
    scaler = st.radio("", ('Zscore', 'MinMax'))
    if scaler == 'MinMax':
        st.write(" Accuracy Score Model KNN Setelah Proses Normalisasi dengan MinMax : ")
        # Membaca data dari file CSV
        df = pd.read_csv('hasil_statistik2.csv')

        # Memisahkan kolom target (label) dari kolom fitur
        X = df.drop(columns=['Label'])  # Kolom fitur
        y = df['Label']  # Kolom target

        # Membagi data menjadi data training dan data testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Normalisasi Min-Max Scaling pada data training
        MinMax_scaler = MinMaxScaler()
        X_train_MinMax = MinMax_scaler.fit_transform(X_train)

        # Simpan skaler ke dalam file menggunakan pickle
        with open('MinMax_scaler.pkl', 'wb') as scaler_file:
            pickle.dump(MinMax_scaler, scaler_file)

        # Streamlit UI with Slider
        st.write("<h5>Slider Cek K</h5><br>", unsafe_allow_html=True)

        # Slider untuk memasukkan nilai k
        k_values = st.slider("Choose the value of k:", min_value=1, max_value=100, value=1, step=1)

        # Membuat model KNN dengan nilai k yang dipilih
        knn_model = KNeighborsClassifier(n_neighbors=k_values)
        knn_model.fit(X_train_MinMax, y_train)

        # Normalisasi Min-Max Scaling pada data testing
        X_test_MinMax = MinMax_scaler.transform(X_test)

        # Melakukan prediksi pada data testing yang telah dinormalisasi
        y_pred = knn_model.predict(X_test_MinMax)

        # Mengukur akurasi model
        accuracy = accuracy_score(y_test, y_pred)

        # Menampilkan akurasi
        st.write(f"Akurasi Model dengan k={k_values}: {accuracy:.4f}")

        # Mencari nilai k terbaik (k dengan akurasi tertinggi)
        accuracy_dict = {}
        for k in range(1, 101):
            knn_model = KNeighborsClassifier(n_neighbors=k)
            knn_model.fit(X_train_MinMax, y_train)
            y_pred = knn_model.predict(X_test_MinMax)
            accuracy = accuracy_score(y_test, y_pred)
            accuracy_dict[k] = accuracy

        # Mendapatkan nilai k terbaik (k dengan akurasi tertinggi)
        best_k = max(accuracy_dict, key=accuracy_dict.get)
        with open('MinMax_knn_model.pkl', 'wb') as file:
            pickle.dump((knn_model, best_k), file)

        # Menampilkan nilai k terbaik dan akurasinya
        st.write(f"Nilai k terbaik adalah {best_k} dengan akurasi {accuracy_dict[best_k]*100:.2f}%")

    elif scaler == "Zscore":
        st.write(" Accuracy Score Model KNN Setelah Proses Normalisasi dengan Zscore : ")
        # Membaca data dari file CSV
        df = pd.read_csv('hasil_statistik2.csv')

        # Memisahkan kolom target (label) dari kolom fitur
        X = df.drop(columns=['Label'])  # Kolom fitur
        y = df['Label']  # Kolom target

        # Membagi data menjadi data training dan data testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Normalisasi Min-Max Scaling pada data training
        Zscore_scaler = StandardScaler()
        X_train_Zscore = Zscore_scaler.fit_transform(X_train)

        # Simpan skaler ke dalam file menggunakan pickle
        with open('Zscore_scaler.pkl', 'wb') as scaler_file:
            pickle.dump(Zscore_scaler, scaler_file)

        # Streamlit UI with Slider
        st.write("<h4>Modelling</h4><br>", unsafe_allow_html=True)

        # Slider untuk memasukkan nilai k
        k_values = st.slider("Choose the value of k:", min_value=1, max_value=100, value=1, step=1)

        # Membuat model KNN dengan nilai k yang dipilih
        knn_model = KNeighborsClassifier(n_neighbors=k_values)
        knn_model.fit(X_train_Zscore, y_train)

        # Normalisasi Min-Max Scaling pada data testing
        X_test_Zscore = Zscore_scaler.transform(X_test)

        # Melakukan prediksi pada data testing yang telah dinormalisasi
        y_pred = knn_model.predict(X_test_Zscore)

        # Mengukur akurasi model
        accuracy = accuracy_score(y_test, y_pred)

        # Menampilkan akurasi
        st.write(f"Akurasi Model dengan k={k_values}: {accuracy:.4f}")

        # Mencari nilai k terbaik (k dengan akurasi tertinggi)
        accuracy_dict = {}
        for k in range(1, 101):
            knn_model = KNeighborsClassifier(n_neighbors=k)
            knn_model.fit(X_train_Zscore, y_train)
            y_pred = knn_model.predict(X_test_Zscore)
            accuracy = accuracy_score(y_test, y_pred)
            accuracy_dict[k] = accuracy

        # Mendapatkan nilai k terbaik (k dengan akurasi tertinggi)
        best_k = max(accuracy_dict, key=accuracy_dict.get)
        with open('Zscore_knn_model.pkl', 'wb') as file:
            pickle.dump((knn_model, best_k), file)

        # Menampilkan nilai k terbaik dan akurasinya
        st.write(f"Nilai k terbaik adalah {best_k} dengan akurasi {accuracy_dict[best_k]*100:.2f}%")
        



with tab6:
    st.title("Reduksi Data")
    
    st.markdown("<h4>Akurasi Model</h4>", unsafe_allow_html=True)
    scaler = st.radio("", ('ReduksiZscore', 'ReduksiMinMax'))
    if scaler == 'ReduksiMinMax':
        # Membaca data dari file CSV
        df = pd.read_csv('hasil_statistik2.csv')

        # Memisahkan kolom target (label) dari kolom fitur
        X = df.drop(columns=['Label'])  # Kolom fitur
        y = df['Label']  # Kolom target

        # Streamlit UI dengan slider untuk memilih nilai n_components dan K
        st.title('K-Nearest Neighbors Classifier with Min-Max Scaling and PCA')
        n_components = st.slider("Choose the number of PCA components :", min_value=1, max_value=20, value=16, step=1)
        k_value = st.slider("Choose the value of K :", min_value=1, max_value=100, value=9, step=1)

        # Normalisasi Min-Max Scaling pada data
        minmax_scaler = MinMaxScaler()
        X_train_minmax = minmax_scaler.fit_transform(X_train)


        # Membagi data menjadi data training dan data testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_test_minmax = minmax_scaler.transform(X_test)
        # Melakukan PCA dengan nilai n_components yang dipilih
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train_minmax)
        X_test_pca = pca.transform(X_test_minmax)

        # Mendefinisikan dan melatih model KNN dengan nilai k yang dipilih
        knn_model = KNeighborsClassifier(n_neighbors=k_value)
        knn_model.fit(X_train_pca, y_train)

        # Melakukan prediksi pada data testing yang telah direduksi oleh PCA
        y_pred = knn_model.predict(X_test_pca)

        # Mengukur akurasi model
        accuracy = accuracy_score(y_test, y_pred)

        # Menampilkan hasil akurasi
        st.write(f"Akurasi Model dengan n_components = {n_components} dan k = {k_value}: {accuracy:.4f}")

        # Menampilkan nilai k terbaik (k dengan akurasi tertinggi)
        st.write("Tabel Nilai k Terbaik untuk Setiap Komponen PCA :")
        best_k_values = []
        best_accuracies = []
        accuracy = 0
        n = 0
        k = 0

        for component in range(1, 21):
            accuracy_dict = {}
            for k in range(1, 101):
                knn_model = KNeighborsClassifier(n_neighbors=k)
                knn_model.fit(X_train_pca[:, :component], y_train)
                y_pred = knn_model.predict(X_test_pca[:, :component])
                accuracy = accuracy_score(y_test, y_pred)
                accuracy_dict[k] = accuracy

            best_k = max(accuracy_dict, key=accuracy_dict.get)
            best_accuracy = accuracy_dict[best_k]
            best_k_values.append(best_k)
            best_accuracies.append(best_accuracy)

            if best_accuracy > accuracy:
                accuracy = best_accuracy
                n = n_components
                k = best_k

        # Menampilkan tabel nilai k terbaik dan akurasi untuk setiap komponen PCA
        data = {'Component': list(range(1, 21)), 'Best K': best_k_values, 'Accuracy': best_accuracies}
        df_result = pd.DataFrame(data)
        st.write(df_result)
        st.write(f"Nilai akurasi reduksi pca pada knn dengan minmax scaler yakni {accuracy*100:.2f}% dengan k {k} pada komponen {n}")


    elif scaler == "ReduksiZscore":
        # Membaca data dari file CSV
        df = pd.read_csv('hasil_statistik2.csv')

        # Memisahkan kolom target (label) dari kolom fitur
        X = df.drop(columns=['Label'])  # Kolom fitur
        y = df['Label']  # Kolom target

        # Streamlit UI dengan slider untuk memilih nilai n_components dan K
        st.title('K-Nearest Neighbors Classifier with Z-Score Scaling and PCA')
        n_components = st.slider("Choose the number of PCA components :", min_value=1, max_value=20, value=14, step=1)
        k_value = st.slider("Choose the value of K :", min_value=1, max_value=100, value=3, step=1)

        # Normalisasi Min-Max Scaling pada data
        Zscore_scaler = StandardScaler()
        X_train_Zscore = Zscore_scaler.fit_transform(X_train)


        # Membagi data menjadi data training dan data testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_test_Zscore = Zscore_scaler.transform(X_test)
        # Melakukan PCA dengan nilai n_components yang dipilih
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train_Zscore)
        X_test_pca = pca.transform(X_test_Zscore)

        # Mendefinisikan dan melatih model KNN dengan nilai k yang dipilih
        knn_model = KNeighborsClassifier(n_neighbors=k_value)
        knn_model.fit(X_train_pca, y_train)

        # Melakukan prediksi pada data testing yang telah direduksi oleh PCA
        y_pred = knn_model.predict(X_test_pca)

        # Mengukur akurasi model
        accuracy = accuracy_score(y_test, y_pred)

        # Menampilkan hasil akurasi
        st.write(f"Akurasi Model dengan n_components = {n_components} dan k = {k_value}: {accuracy:.4f}")

        # Menampilkan nilai k terbaik (k dengan akurasi tertinggi)
        st.write("Tabel Nilai k Terbaik untuk Setiap Komponen PCA :")
        best_k_values = []
        best_accuracies = []
        accuracy = 0
        n = 0
        k = 0

        for component in range(1, 21):
            accuracy_dict = {}
            for k in range(1, 101):
                knn_model = KNeighborsClassifier(n_neighbors=k)
                knn_model.fit(X_train_pca[:, :component], y_train)
                y_pred = knn_model.predict(X_test_pca[:, :component])
                accuracy = accuracy_score(y_test, y_pred)
                accuracy_dict[k] = accuracy

            best_k = max(accuracy_dict, key=accuracy_dict.get)
            best_accuracy = accuracy_dict[best_k]
            best_k_values.append(best_k)
            best_accuracies.append(best_accuracy)

            if best_accuracy > accuracy:
                accuracy = best_accuracy
                n = n_components
                k = best_k

        # Menampilkan tabel nilai k terbaik dan akurasi untuk setiap komponen PCA
        data = {'Component': list(range(1, 21)), 'Best K': best_k_values, 'Accuracy': best_accuracies}
        df_result = pd.DataFrame(data)
        st.write(df_result)
        st.write(f"Nilai akurasi reduksi pca pada knn dengan zscore scaler yakni {accuracy*100:.2f}% dengan k {k} pada komponen {n}")
        


with tab7:
    st.title("Grid Search")
    
    st.markdown("<h4>Akurasi Model</h4>", unsafe_allow_html=True)
    scaler = st.radio("", ('Grid Search Z-Score', 'Grid Search MinMax'))
    if scaler == 'Grid Search Z-Score':
        st.title('Grid Search Metode KNN dengan Z-Score Scaler')
        # Membaca data dari file CSV
        df = pd.read_csv('hasil_statistik2.csv')

        # Memisahkan kolom target (label) dari kolom fitur
        X = df.drop(columns=['Label'])  # Kolom fitur
        y = df['Label']  # Kolom target

        # Normalisasi data menggunakan StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Memisahkan data menjadi data latih dan data uji
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Mendefinisikan parameter yang ingin diuji
        param_grid = {
            'n_neighbors': list(range(1, 101)),
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }

        # Mendefinisikan model KNN
        knn = KNeighborsClassifier()

        # Mendefinisikan Grid Search dengan model KNN dan parameter yang diuji
        grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)  # Menggunakan data latih yang belum diproses PCA

        # Menampilkan parameter terbaik
        st.write("Best Parameters:", grid_search.best_params_)

        # Menggunakan PCA dengan komponen utama terbaik
        best_n_neighbors = grid_search.best_params_['n_neighbors']
        best_weights = grid_search.best_params_['weights']
        best_metric = grid_search.best_params_['metric']

        accuracy_dict = {}
        for n_components in range(X_train.shape[1], 0, -1):
            pca = PCA(n_components=n_components)
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)

            # Membuat model KNN dengan hyperparameter terbaik
            best_knn_model = KNeighborsClassifier(n_neighbors=best_n_neighbors, weights=best_weights, metric=best_metric)
            best_knn_model.fit(X_train_pca, y_train)

            # Membuat prediksi menggunakan model terbaik
            y_pred = best_knn_model.predict(X_test_pca)

            # Mengukur akurasi model terbaik pada data uji
            grid_knn_pca = accuracy_score(y_test, y_pred)

            # Menyimpan akurasi dalam dictionary
            accuracy_dict[n_components] = grid_knn_pca

            st.write(f"Accuracy dengan {n_components} PCA components: {grid_knn_pca * 100:.2f}%")

        # Mencari nilai k terbaik
        best_comp = max(accuracy_dict, key=accuracy_dict.get)
        best_accuracy = accuracy_dict[best_comp] * 100
        st.write(f"\nBest Accuracy pada Grid Search KNN Z-Score Scaler {best_comp} PCA components: {best_accuracy:.2f}%")

        # Define and train your KNeighborsClassifier with the best hyperparameters
        final_knn_model = KNeighborsClassifier(n_neighbors=best_n_neighbors, weights=best_weights, metric=best_metric)
        X_train = X_train
        y_train = y_train
        y_test = y_test

        # Store the hyperparameters in a dictionary
        hyperparameters = {
            'best_n_neighbors': best_n_neighbors,
            'best_weights': best_weights,
            'best_metric': best_metric,
            'best_comp': best_comp
        }

        # Store the model and hyperparameters in a dictionary before pickling
        model_data = {
            'X_train': X_train,
            'y_train': y_train,
            'y_test' : y_test,
            'scaler' : X_scaled,
            'hyperparameters': hyperparameters
        }

        # Save the model and hyperparameters using pickle
        with open('gridsearchknnzscoremodel.pkl', 'wb') as model_file:
            pickle.dump(model_data, model_file)

        # Menampilkan grafik komponen PCA vs. Akurasi
        components = list(accuracy_dict.keys())
        accuracies = list(accuracy_dict.values())

        plt.figure(figsize=(10, 6))
        plt.plot(components, accuracies, marker='o', color='b', label='Akurasi')
        plt.xlabel('Jumlah Komponen PCA')
        plt.ylabel('Akurasi')
        plt.title('Akurasi vs. Jumlah Komponen PCA')
        plt.xticks(np.arange(min(components), max(components)+1, 1))
        plt.grid(True)
        plt.legend()
        st.pyplot(plt)
    elif scaler == "Grid Search MinMax":
        st.title('Grid Search Metode KNN dengan MinMax Scaler')
        # Membaca data dari file CSV
        df = pd.read_csv('hasil_statistik2.csv')

        # Memisahkan kolom target (label) dari kolom fitur
        X = df.drop(columns=['Label'])  # Kolom fitur
        y = df['Label']  # Kolom target

        # Normalisasi data menggunakan StandardScaler
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # Memisahkan data menjadi data latih dan data uji
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Mendefinisikan parameter yang ingin diuji
        param_grid = {
            'n_neighbors': list(range(1, 101)),
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }

        # Mendefinisikan model KNN
        knn = KNeighborsClassifier()

        # Mendefinisikan Grid Search dengan model KNN dan parameter yang diuji
        grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)  # Menggunakan data latih yang belum diproses PCA

        # Menampilkan parameter terbaik
        st.write("Best Parameters:", grid_search.best_params_)

        # Menggunakan PCA dengan komponen utama terbaik
        best_n_neighbors = grid_search.best_params_['n_neighbors']
        best_weights = grid_search.best_params_['weights']
        best_metric = grid_search.best_params_['metric']

        accuracy_dict = {}
        for n_components in range(X_train.shape[1], 0, -1):
            pca = PCA(n_components=n_components)
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)

            # Membuat model KNN dengan hyperparameter terbaik
            best_knn_model = KNeighborsClassifier(n_neighbors=best_n_neighbors, weights=best_weights, metric=best_metric)
            best_knn_model.fit(X_train_pca, y_train)

            # Membuat prediksi menggunakan model terbaik
            y_pred = best_knn_model.predict(X_test_pca)

            # Mengukur akurasi model terbaik pada data uji
            grid_knn_pca = accuracy_score(y_test, y_pred)

            # Menyimpan akurasi dalam dictionary
            accuracy_dict[n_components] = grid_knn_pca

            st.write(f"Accuracy dengan {n_components} PCA components : {grid_knn_pca * 100:.2f}%")

        # Mencari nilai k terbaik
        best_comp = max(accuracy_dict, key=accuracy_dict.get)
        best_accuracy = accuracy_dict[best_comp] * 100
        st.write(f"\nBest Accuracy pada Grid Search KNN MinMax Scaler {best_comp} PCA components: {best_accuracy:.2f}%")

        # Define and train your KNeighborsClassifier with the best hyperparameters
        final_knn_model = KNeighborsClassifier(n_neighbors=best_n_neighbors, weights=best_weights, metric=best_metric)
        X_train = X_train
        y_train = y_train
        y_test = y_test

        # Store the hyperparameters in a dictionary
        hyperparameters = {
            'best_n_neighbors': best_n_neighbors,
            'best_weights': best_weights,
            'best_metric': best_metric,
            'best_comp': best_comp
        }

        # Store the model and hyperparameters in a dictionary before pickling
        model_data = {
            'X_train': X_train,
            'y_train': y_train,
            'y_test' : y_test,
            'scaler' : X_scaled,
            'hyperparameters': hyperparameters
        }

        # Save the model and hyperparameters using pickle
        with open('gridsearchknnminmaxmodel.pkl', 'wb') as model_file:
            pickle.dump(model_data, model_file)

        # Menampilkan grafik komponen PCA vs. Akurasi
        components = list(accuracy_dict.keys())
        accuracies = list(accuracy_dict.values())

        plt.figure(figsize=(10, 6))
        plt.plot(components, accuracies, marker='o', color='b', label='Akurasi')
        plt.xlabel('Jumlah Komponen PCA')
        plt.ylabel('Akurasi')
        plt.title('Akurasi vs. Jumlah Komponen PCA')
        plt.xticks(np.arange(min(components), max(components)+1, 1))
        plt.grid(True)
        plt.legend()
        st.pyplot(plt)