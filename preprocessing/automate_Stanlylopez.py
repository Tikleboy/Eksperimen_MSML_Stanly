import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

# --- 1. FUNGSI LOAD DATA ---
def load_data(path):
    print(f"[INFO] Memuat data dari: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File dataset tidak ditemukan di: {path}")
    return pd.read_csv(path)

# --- 2. FUNGSI PREPROCESSING UTAMA ---
def preprocess_data(df):
    print("[INFO] Memulai proses pembersihan data...")
    
    # Menghapus kolom yang tidak relevan
    cols_to_drop = ['replyContent', 'repliedAt']
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    # Menghapus Missing Values
    initial_rows = len(df)
    df = df.dropna()
    print(f"   - Baris NaN dihapus: {initial_rows - len(df)}")
    
    # Menghapus Duplikat
    current_rows = len(df)
    df = df.drop_duplicates()
    print(f"   - Duplikat dihapus: {current_rows - len(df)}")

    # Feature Scaling: thumbsUpCount
    # Kita buat scaler baru setiap run (untuk preprocessing sederhana)
    scaler = StandardScaler()
    if 'thumbsUpCount' in df.columns:
        df['thumbsUpCount_scaled'] = scaler.fit_transform(df[['thumbsUpCount']])

    # Outlier Removal: IQR Method
    if 'thumbsUpCount' in df.columns:
        Q1 = df['thumbsUpCount'].quantile(0.25)
        Q3 = df['thumbsUpCount'].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        before_outlier = len(df)
        df = df[
            (df['thumbsUpCount'] >= lower_bound) &
            (df['thumbsUpCount'] <= upper_bound)
        ].copy()
        print(f"   - Outlier dihapus: {before_outlier - len(df)}")
    
    return df

# --- 3. FUNGSI LABELING SENTIMENT ---
def apply_labeling(df):
    print("[INFO] Melakukan labeling sentiment...")
    
    def sentiment_label(score):
        if score <= 2:
            return 'negative'
        elif score == 3:
            return 'neutral'
        else:
            return 'positive'

    if 'score' in df.columns:
        df['sentiment'] = df['score'].apply(sentiment_label)
    
    return df

# --- 4. FUNGSI SIMPAN DATA ---
def save_data(df, output_path):
    # Buat folder output jika belum ada
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Data bersih tersimpan di: {output_path}")

# --- BLOCK UTAMA ---
if __name__ == "__main__":
    # Mengatur path secara dinamis agar aman dijalankan di mana saja
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path Input: Naik satu folder (..) lalu masuk ke data_raw
    input_file = os.path.join(base_dir, '../data_raw/ulasan_KAI.csv')
    
    # Path Output: Di dalam folder preprocessing/data_clean
    output_file = os.path.join(base_dir, 'data_clean/ulasan_KAI_preprocessing.csv')

    try:
        # Eksekusi urutan fungsi
        df = load_data(input_file)       # 1. Load
        df_clean = preprocess_data(df)   # 2. Preprocess (Cleaning & Scaling)
        df_final = apply_labeling(df_clean) # 3. Labeling
        save_data(df_final, output_file) # 4. Save
        
    except Exception as e:
        print(f"[ERROR] Terjadi kesalahan: {e}")