## Domain Proyek

Prediksi harga saham merupakan tantangan signifikan dalam dunia keuangan karena sifatnya yang volatil dan dipengaruhi oleh berbagai faktor, seperti kondisi ekonomi, harga komoditas global, dan kebijakan perusahaan. PT Aneka Tambang Tbk (ANTM), sebagai salah satu perusahaan tambang terbesar di Indonesia yang bergerak di sektor nikel, emas, dan bauksit, memiliki pergerakan harga saham yang dipengaruhi oleh fluktuasi harga komoditas dan dinamika pasar domestik. Dengan meningkatnya jumlah investor ritel di Indonesia, yang mencatatkan pertumbuhan sebesar 28% pada tahun 2020 [1], kebutuhan akan alat prediksi harga saham yang akurat semakin penting untuk mendukung pengambilan keputusan investasi yang lebih terinformasi.

Masalah ini relevan karena prediksi harga saham yang akurat dapat membantu investor ritel meminimalkan risiko dan memaksimalkan keuntungan. Pendekatan machine learning, khususnya algoritma Long Short-Term Memory (LSTM), efektif untuk menangkap pola non-linear dan dependensi temporal dalam data harga saham, yang sulit dilakukan oleh metode konvensional. Penelitian sebelumnya menunjukkan bahwa LSTM mampu memberikan akurasi prediksi yang tinggi untuk harga saham [2]. Oleh karena itu, proyek ini bertujuan untuk mengembangkan model prediksi harga saham PT Antam menggunakan LSTM untuk membantu investor ritel membuat keputusan investasi yang lebih baik.

**Referensi**:
- A. N. Karmayuda, “Machine Learning Untuk Pemodelan Pergerakan Harga Saham PT Telkom Indonesia TBK,” Undergraduate thesis, Universitas Pembangunan Jaya, 2021.[1](https://eprints.upj.ac.id/id/eprint/2388/)
- L. Setiawan, D. Susanti, and R. Riaman, “Analisis Perbandingan Hasil Peramalan Harga Saham Menggunakan Model Autoregresive Integrated Moving Average dan Long Short Term Memory,” *Jurnal Matematika Integratif*, vol. 19, no. 2, pp. 223–234, 2023.[2](https://jurnal.unpad.ac.id/jmi/article/view/42164)

## Business Understanding

### Problem Statements
- **Pernyataan Masalah 1**: Harga saham PT Antam bersifat volatil dan dipengaruhi oleh faktor eksternal seperti harga komoditas global dan kebijakan domestik, Selain itu dengan adanya kasus yang baru viral terkait dengan korupsi di perusahaan tersebut pastinya menyebabkan fluktuasi yang signifikan sehingga sulit diprediksi dengan metode konvensional.
- **Pernyataan Masalah 2**: Investor ritel membutuhkan alat prediksi yang akurat untuk mengidentifikasi tren harga saham PT Antam guna mendukung keputusan investasi yang optimal.
- **Pernyataan Masalah 3**: Data historis harga saham PT Antam memiliki pola non-linear yang kompleks, yang memerlukan pendekatan machine learning untuk menangkap dependensi temporal.

### Goals
- **Jawaban Pernyataan Masalah 1**: Mengembangkan model machine learning berbasis LSTM yang mampu memprediksi harga saham PT Antam dengan akurasi tinggi meskipun terdapat volatilitas.
- **Jawaban Pernyataan Masalah 2**: Menyediakan alat prediksi berbasis LSTM yang dapat digunakan investor ritel untuk membuat keputusan investasi yang lebih terinformasi.
- **Jawaban Pernyataan Masalah 3**: Menggunakan algoritma LSTM untuk menangkap pola non-linear dan dependensi temporal dalam data historis harga saham PT Antam.

### Solution Statements
- **Solusi 1**: Menerapkan model LSTM untuk memprediksi harga saham PT Antam berdasarkan data historis, dengan metrik evaluasi Mean Absolute Percentage Error (MAPE) dan Root Mean Squared Error (RMSE) untuk mengukur akurasi prediksi.
- **Solusi 2**: Mengoptimalkan arsitektur model LSTM dengan konfigurasi dua layer LSTM dan layer Dense untuk menangkap pola harga saham dengan baik.
- **Solusi 3**: Menyediakan visualisasi prediksi harga saham untuk membantu investor ritel memahami tren harga dan membuat keputusan investasi.

## Data Understanding

Dataset yang digunakan adalah data historis harga saham PT Aneka Tambang Tbk (ANTM) dari Yahoo Finance ([https://finance.yahoo.com/quote/ANTM.JK/history](https://finance.yahoo.com/quote/ANTM.JK/history)). Data ini mencakup periode dari 23 Mei 2022 hingga 23 Mei 2025, dengan interval harian, dan terdiri dari 721 baris data. Dataset mencakup harga pembukaan, penutupan, tertinggi, terendah, volume perdagangan, dividen, dan stock split.

### Variabel-variabel pada dataset ANTM adalah sebagai berikut:
- **Open**: Harga pembukaan saham pada hari perdagangan tertentu (dalam Rupiah).
- **High**: Harga tertinggi saham pada hari perdagangan (dalam Rupiah).
- **Low**: Harga terendah saham pada hari perdagangan (dalam Rupiah).
- **Close**: Harga penutupan saham pada hari perdagangan (dalam Rupiah).
- **Volume**: Jumlah saham yang diperdagangkan pada hari tertentu.
- **Dividends**: Laba perusahaan yang dibagikan kepada pemegang saham.
- **Stock Splits**: Aksi korporasi di mana perusahaan memecah harga saham per lembar menjadi lebih rendah dengan rasio tertentu.


**Exploratory Data Analysis (EDA)**:
- **Korelasi Antar Variabel**: Heatmap korelasi menunjukkan bahwa harga penutupan (Close) memiliki korelasi kuat dengan harga pembukaan (Open), tertinggi (High), dan terendah (Low), dengan nilai korelasi mendekati 1, menandakan ketergantungan temporal yang kuat.
- **Pair Plot**: Pair plot fitur numerik (Open, High, Low, Close, Volume) menggambarkan hubungan linier antar harga saham, dengan Volume menunjukkan distribusi yang lebih bervariasi.
- **Box Plot Volume**: Analisis box plot menunjukkan adanya outlier pada volume perdagangan, yang mungkin disebabkan oleh peristiwa pasar seperti pengumuman kebijakan atau laporan keuangan perusahaan.
- **Rata-rata Harga per Tahun**: Bar plot menunjukkan tren rata-rata harga penutupan saham per tahun, dengan peningkatan signifikan pada tahun 2025 dibandingkan 2022–2024, mencerminkan volatilitas pasar.
- **Pemeriksaan Data Hilang**: Tidak ditemukan data hilang dalam dataset, sehingga integritas data terjaga.

## Data Preparation

Proses persiapan data dilakukan untuk memastikan dataset dapat digunakan secara optimal oleh model machine learning. Berikut adalah tahapan yang dilakukan:

1. **Pembersihan Data**:
   - Proses: Menghapus kolom *Dividends* dan *Stock Splits* karena tidak relevan untuk prediksi harga saham (nilainya nol untuk seluruh periode).
   - Alasan: Mengurangi noise dan kompleksitas model dengan hanya menggunakan fitur yang relevan (Open, High, Low, Close, Volume).

2. **Normalisasi Data**:
    - Proses: Menggunakan Min-Max Scaling untuk mentransformasi harga penutupan (Close) ke rentang [0,1].
    - Alasan: Normalisasi mempercepat konvergensi model LSTM dan mencegah dominasi fitur dengan skala besar.
    - **Contoh Kode**:
     ```python
     from sklearn.preprocessing import MinMaxScaler
     scaler = MinMaxScaler(feature_range=(0, 1))
     scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
     ```

3. **Pembagian Data**:
   - Proses: Membagi dataset menjadi 80% data latih dan 20% data uji.
   - Alasan: Pembagian ini memungkinkan model untuk belajar dari data historis yang cukup besar sambil menyediakan data uji untuk evaluasi.

4. **Pembuatan Jendela Waktu (Time Window)**:
   - Proses: Menggunakan jendela waktu 30 hari untuk memprediksi harga saham hari berikutnya.
   - Alasan: Jendela waktu ini memungkinkan model LSTM untuk menangkap dependensi jangka panjang dalam data deret waktu.
   - **Contoh Kode**
     ```python
      def prepare_train_df(data_series, data_input=6):
          X, y = [], []
          for i in range(len(data_series) - data_input):
              X.append(data_series[i:i + data_input].flatten())
              y.append(data_series[i + data_input][0])
          return pd.concat([pd.DataFrame(X, columns=[f'x{i+1}' for i in range(data_input)]), pd.Series(y, name='y')], axis=1)
     ```

## Modeling

Model yang digunakan dalam proyek ini adalah Long Short-Term Memory (LSTM), yang dipilih karena kemampuannya menangkap pola non-linear dan dependensi jangka panjang dalam data deret waktu.

### Model LSTM
- **Arsitektur**: Model LSTM dibangun dengan 2 layer LSTM (100 unit dan 50 unit), masing-masing diikuti oleh Dropout (0.3) untuk mencegah overfitting, dan dua layer Dense (25 unit dengan aktivasi ReLU dan 1 unit untuk output).
- **Parameter**:
  - Epoch: 100
  - Batch size: 32
  - Optimizer: Adam
  - Learning rate: 0.001
  - Callbacks: EarlyStopping (patience=15) dan ReduceLROnPlateau (factor=0.2, patience=5)
- **Kelebihan**:
  - Mampu menangkap dependensi jangka panjang dalam data deret waktu.
  - Efektif untuk data non-linear seperti harga saham.[2](https://repository.bsi.ac.id/repo/51717/PENERAPAN-DATA-MINING-DALAM-PREDIKSI-HARGA-SAHAM-DI-INDONESIA-MENGGUNAKAN-ALGORITMA-LSTM)
- **Kekurangan**:
  - Membutuhkan waktu pelatihan yang lebih lama dibandingkan model statistik.
  - Sensitif terhadap pengaturan hyperparameter.

## Evaluation

### Metrik Evaluasi
Metrik yang digunakan adalah:
- **Mean Absolute Percentage Error (MAPE)**: Mengukur persentase kesalahan rata-rata relatif terhadap nilai aktual. Formula:  
  $\text{MAPE} = \frac{1}{n} \sum_{i=1}^n \left| \frac{y_i - \hat{y}_i}{y_i} \right| \times 100$
- **Root Mean Squared Error (RMSE)**: Mengukur akar kuadrat rata-rata kesalahan kuadrat. Formula:  
  $\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$

### Hasil Proyek
- **Model LSTM**:
  - MAPE: 4.794% (akurat hingga 95.206%)
  - RMSE: 0.06510171142093855 (dalam Rupiah)
- **Penjelasan**:
  - Model LSTM menunjukkan performa yang baik dengan MAPE yang rendah, menandakan akurasi relatif yang tinggi (lebih dari 95.206% pada data uji).
  - RMSE yang relatif kecil menunjukkan bahwa model konsisten dalam memprediksi harga saham PT Antam dalam satuan Rupiah.
  - Visualisasi prediksi menunjukkan bahwa model LSTM mampu mengikuti tren harga saham aktual dengan baik, meskipun terdapat sedikit penyimpangan pada periode volatilitas tinggi.
  - Prediksi 30 hari ke depan (23 Mei 2025 hingga 21 Juni 2025) menunjukkan tren harga saham yang cenderung stabil dengan fluktuasi ringan, memberikan wawasan berharga bagi investor.

**Visualisasi Hasil**:
![Gambar 1: Perbandingan harga penutupan aktual dan prediksi menunjukkan bahwa model LSTM mampu menangkap tren jangka pendek dengan baik, dengan prediksi 30 hari ke depan menunjukkan stabilitas harga.](image.png)

**Kesimpulan**:
Model LSTM berhasil memprediksi harga saham PT Antam dengan akurasi tinggi, ditunjukkan oleh MAPE sebesar [hasil dari kode, misalnya 4.794%] dan RMSE sebesar [hasil dari kode, misalnya 0.06510171142093855 IDR] pada data uji. Model ini dapat digunakan oleh investor ritel untuk mendukung keputusan investasi, seperti menentukan waktu beli atau jual saham. Namun, perlu diperhatikan bahwa prediksi harga saham tetap memiliki risiko akibat faktor eksternal yang tidak dapat dimodelkan sepenuhnya, seperti perubahan harga komoditas global atau kebijakan pemerintah.
