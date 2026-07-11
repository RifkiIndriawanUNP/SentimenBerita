import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from datetime import datetime
import os

# Download NLTK resources
@st.cache_resource
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

download_nltk_resources()

# Initialize Sastrawi stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Load models
@st.cache_resource
def load_models():
    try:
        tfidf = joblib.load('tfidf.pkl')
        svm = joblib.load('svm.pkl')
        nb = joblib.load('naive_bayes.pkl')
        return tfidf, svm, nb
    except FileNotFoundError as e:
        st.error(f"Model file tidak ditemukan: {e}")
        st.stop()

tfidf, svm_model, nb_model = load_models()

# Preprocessing function
def preprocess_text(text):
    # Cleaning
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Case folding
    text = text.lower().strip()
    
    # Tokenizing
    tokens = word_tokenize(text)
    
    # Stopword removal
    stop_words = set(stopwords.words('indonesian'))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    
    # Stemming
    tokens = [stemmer.stem(word) for word in tokens]
    
    return ' '.join(tokens)

# Prediction function
def predict_sentiment(text, model_choice):
    processed_text = preprocess_text(text)
    
    if not processed_text.strip():
        return None, None
    
    # Transform text
    text_tfidf = tfidf.transform([processed_text])
    
    # Predict with chosen model
    if model_choice == "SVM (LinearSVC)":
        prediction = svm_model.predict(text_tfidf)[0]
        probabilities = svm_model.decision_function(text_tfidf)[0]
    else:  # Naive Bayes
        prediction = nb_model.predict(text_tfidf)[0]
        probabilities = nb_model.predict_proba(text_tfidf)[0]
    
    return prediction, probabilities

# Main app
def main():
    # Page configuration
    st.set_page_config(
        page_title="Analisis Sentimen Berita Kebijakan Pemerintah",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""
    
    # Custom CSS
    st.markdown("""
    <style>
    /* Global Styles */
    .stApp {
        background-color: #1E1E1E;
    }
    
    /* Typography */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #E53935;
        text-align: center;
        margin: 1.5rem 0 0.5rem 0;
        letter-spacing: 0.5px;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #BDBDBD;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #E53935;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #424242;
    }
    
    /* Sentiment Labels */
    .sentiment-positive {
        background-color: #1B3A1B;
        color: #66BB6A;
        font-weight: 600;
        font-size: 1.2rem;
        padding: 12px 20px;
        border-radius: 8px;
        border-left: 5px solid #4CAF50;
    }
    
    .sentiment-negative {
        background-color: #3A1B1B;
        color: #EF5350;
        font-weight: 600;
        font-size: 1.2rem;
        padding: 12px 20px;
        border-radius: 8px;
        border-left: 5px solid #E53935;
    }
    
    .sentiment-neutral {
        background-color: #3A301B;
        color: #FFCA28;
        font-weight: 600;
        font-size: 1.2rem;
        padding: 12px 20px;
        border-radius: 8px;
        border-left: 5px solid #FFC107;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #E53935;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        background-color: #C62828;
        box-shadow: 0 4px 12px rgba(229, 57, 53, 0.3);
        transform: translateY(-1px);
    }
    
    .stButton > button:active {
        background-color: #B71C1C;
        transform: translateY(0);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #252525;
        border-right: 1px solid #424242;
    }
    
    [data-testid="stSidebar"] .stMarkdown h2 {
        color: #E53935;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background-color: #E53935;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        border-bottom: 2px solid #424242;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #9E9E9E;
        font-weight: 500;
        padding: 0.5rem 1rem;
    }
    
    .stTabs [aria-selected="true"] {
        color: #E53935;
        border-bottom: 3px solid #E53935;
    }
    
    /* Dataframe */
    [data-testid="stDataFrame"] {
        border: 1px solid #424242;
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Upload box */
    [data-testid="stFileUploader"] {
        border: 2px dashed #E53935;
        border-radius: 10px;
        padding: 2rem;
        background-color: #2A1A1A;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #C62828;
        background-color: #3A2020;
    }
    
    /* Text area */
    .stTextArea textarea {
        border: 2px solid #424242;
        border-radius: 8px;
    }
    
    .stTextArea textarea:focus {
        border-color: #E53935;
        box-shadow: 0 0 0 2px rgba(229, 57, 53, 0.1);
    }
    
    /* Upload instruction box */
    .upload-instruction {
        background-color: #2A1A1A;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #E53935;
        margin-bottom: 1rem;
    }
    
    .upload-instruction p {
        margin: 0;
        color: #BDBDBD;
        font-weight: 500;
    }
    
    .upload-instruction code {
        background-color: #3A2020;
        padding: 2px 8px;
        border-radius: 4px;
        color: #E53935;
        font-weight: 600;
    }
    
    /* Sample text boxes */
    .sample-text-box {
        background-color: #2A2A2A;
        padding: 12px 15px;
        border-radius: 8px;
        margin-bottom: 12px;
        border-left: 4px solid #E53935;
        font-size: 0.9rem;
        color: #BDBDBD;
    }
    
    .sample-text-label {
        font-weight: 700;
        margin-bottom: 5px;
        font-size: 0.85rem;
    }
    
    .sample-text-label.positive {
        color: #66BB6A;
    }
    
    .sample-text-label.negative {
        color: #EF5350;
    }
    
    .sample-text-label.neutral {
        color: #FFCA28;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header Image
    if os.path.exists('ikn.png'):
        st.image('ikn.png', use_container_width=True)
    else:
        st.warning("Gambar header 'ikn.png' tidak ditemukan di folder root. Silakan tambahkan file gambar dengan nama 'ikn.png'")
    
    # Main Title
    st.markdown('<p class="main-title">Analisis Sentimen Berita Kebijakan Pemerintah Era Prabowo</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Sistem Klasifikasi Sentimen Menggunakan Machine Learning</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## Konfigurasi Model")
        
        model_choice = st.selectbox(
            "Pilih Model",
            ["SVM (LinearSVC)", "Naive Bayes"],
            help="SVM memiliki akurasi lebih tinggi (74.42%) dibanding Naive Bayes (63.95%)"
        )
        
        st.markdown("---")
        st.markdown("## Statistik Model")
        st.info(f"""
        **SVM (LinearSVC)**
        - Akurasi: 74.42%
        
        **Naive Bayes**
        - Akurasi: 63.95%
        """)
        
        st.markdown("---")
        st.markdown("## Tentang Aplikasi")
        st.markdown("""
        Aplikasi ini menganalisis sentimen berita kebijakan pemerintah era Prabowo 
        menggunakan Machine Learning.
        
        **Label Sentimen:**
        - Positif
        - Negatif
        - Netral
        """)
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["Analisis Teks", "Upload CSV", "Dataset Sample"])
    
    # ============================================================
    # TAB 1: Single Text Analysis
    # ============================================================
    with tab1:
        st.markdown('<p class="section-title">Analisis Sentimen Teks Berita</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            user_input = st.text_area(
                "Masukkan teks berita:",
                value=st.session_state.user_input,
                height=150,
                placeholder="Masukkan teks berita yang akan dianalisis sentimennya...",
                key="text_input"
            )
            
            if st.button("Analisis Sentimen", type="primary"):
                if user_input.strip():
                    with st.spinner("Menganalisis sentimen..."):
                        prediction, probabilities = predict_sentiment(user_input, model_choice)
                    
                    if prediction:
                        st.markdown("---")
                        st.markdown("### Hasil Analisis:")
                        
                        # Display prediction
                        if prediction == "Positif":
                            st.markdown('<p class="sentiment-positive">Sentimen: Positif</p>', unsafe_allow_html=True)
                        elif prediction == "Negatif":
                            st.markdown('<p class="sentiment-negative">Sentimen: Negatif</p>', unsafe_allow_html=True)
                        else:
                            st.markdown('<p class="sentiment-neutral">Sentimen: Netral</p>', unsafe_allow_html=True)
                        
                        # Confidence score
                        if model_choice == "SVM (LinearSVC)":
                            confidence = max(probabilities)
                            st.progress(float(confidence / max(probabilities)))
                            st.caption(f"Confidence Score: {confidence:.3f}")
                        else:
                            confidence = max(probabilities)
                            st.progress(float(confidence))
                            st.caption(f"Probability: {confidence:.2%}")
                else:
                    st.warning("Mohon masukkan teks berita terlebih dahulu.")
        
        with col2:
            st.markdown('<p class="section-title">Contoh Teks</p>', unsafe_allow_html=True)
            
            # Sample texts displayed below the analysis box
            st.markdown("""
            <div class="sample-text-box">
                <div class="sample-text-label positive">POSITIF</div>
                Pemerintah berhasil meningkatkan pertumbuhan ekonomi nasional melalui kebijakan hilirisasi yang mendorong investasi dan menciptakan lapangan kerja baru.
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="sample-text-box">
                <div class="sample-text-label negative">NEGATIF</div>
                Badan Gizi Nasional (BGN) merevisi sejumlah kebijakan setelah kasus korupsi menyeret para petingginya. Dari langkah-langkah yang diambil, penghentian sementara tidak masuk daftar.
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="sample-text-box">
                <div class="sample-text-label neutral">NETRAL</div>
                Pemerintah akan mengumumkan kebijakan baru terkait subsidi BBM pada rapat kabinet besok pagi di Istana Negara.
            </div>
            """, unsafe_allow_html=True)
    
    # ============================================================
    # TAB 2: CSV Upload & Batch Analysis
    # ============================================================
    with tab2:
        st.markdown('<p class="section-title">Upload File CSV untuk Batch Analysis</p>', unsafe_allow_html=True)
        
        # Upload instruction
        st.markdown("""
        <div class="upload-instruction">
            <p>
                <span style="color: #E53935; font-weight: 700;">Persyaratan File CSV:</span> 
                File harus memiliki kolom <code>content</code> 
                yang berisi teks berita yang akan dianalisis.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Pilih file CSV",
            type=['csv'],
            help="File CSV harus memiliki kolom 'content' yang berisi teks berita untuk dianalisis"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"File berhasil diupload! Jumlah data: {len(df)} baris")
                
                # Preview data
                st.markdown("### Preview Data:")
                st.dataframe(df.head(), use_container_width=True)
                
                # Select column for analysis
                text_column = None
                if 'content' in df.columns:
                    text_column = 'content'
                elif 'title' in df.columns:
                    text_column = 'title'
                else:
                    st.warning("Kolom 'content' tidak ditemukan. Silakan pilih kolom teks secara manual:")
                    text_column = st.selectbox("Pilih kolom teks:", df.columns)
                
                if text_column:
                    if st.button("Analisis Semua Data", type="primary"):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        predictions = []
                        
                        for i, row in df.iterrows():
                            text = str(row[text_column])
                            if text.strip():
                                pred, _ = predict_sentiment(text, model_choice)
                                predictions.append(pred if pred else "Netral")
                            else:
                                predictions.append("Netral")
                            
                            # Update progress
                            progress = (i + 1) / len(df)
                            progress_bar.progress(progress)
                            status_text.text(f"Memproses: {i+1}/{len(df)} ({progress*100:.1f}%)")
                        
                        df['predicted_sentiment'] = predictions
                        status_text.text("Analisis selesai!")
                        
                        # Tampilkan 10 data teratas setelah sentimen ditambahkan
                        st.markdown("---")
                        st.markdown("## 10 Data Teratas Hasil Analisis")
                        st.dataframe(df.head(10), use_container_width=True)
                        
                        # Visualizations
                        st.markdown("---")
                        st.markdown("## Visualisasi Hasil Analisis")
                        
                        # Distribution
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            sentiment_counts = df['predicted_sentiment'].value_counts()
                            
                            fig_pie = px.pie(
                                values=sentiment_counts.values,
                                names=sentiment_counts.index,
                                title="Distribusi Sentimen",
                                color=sentiment_counts.index,
                                color_discrete_map={
                                    'Positif': '#4CAF50',
                                    'Negatif': '#E53935',
                                    'Netral': '#FFC107'
                                },
                                hole=0.4
                            )
                            fig_pie.update_traces(textinfo='percent+label')
                            fig_pie.update_layout(
                                paper_bgcolor='#1E1E1E',
                                plot_bgcolor='#1E1E1E',
                                font=dict(color='#BDBDBD')
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)
                        
                        with col2:
                            fig_bar = px.bar(
                                x=sentiment_counts.index,
                                y=sentiment_counts.values,
                                title="Jumlah per Sentimen",
                                color=sentiment_counts.index,
                                color_discrete_map={
                                    'Positif': '#4CAF50',
                                    'Negatif': '#E53935',
                                    'Netral': '#FFC107'
                                },
                                labels={'x': 'Sentimen', 'y': 'Jumlah'}
                            )
                            fig_bar.update_layout(
                                showlegend=False,
                                paper_bgcolor='#1E1E1E',
                                plot_bgcolor='#1E1E1E',
                                font=dict(color='#BDBDBD')
                            )
                            st.plotly_chart(fig_bar, use_container_width=True)
                        
                        # Tentukan kolom untuk WordCloud
                        # Prioritas: stemmed_text > content + title > content > title > text_column
                        if 'stemmed_text' in df.columns:
                            wc_column = 'stemmed_text'
                        elif 'content' in df.columns and 'title' in df.columns:
                            df['text_for_wc'] = df['title'].fillna('') + ' ' + df['content'].fillna('')
                            wc_column = 'text_for_wc'
                        elif 'content' in df.columns:
                            wc_column = 'content'
                        elif 'title' in df.columns:
                            wc_column = 'title'
                        else:
                            wc_column = text_column
                        
                        # WordCloud for each sentiment
                        st.markdown("### Word Cloud per Sentimen")
                        
                        # Tampilkan info kolom yang digunakan
                        if 'stemmed_text' in df.columns:
                            st.caption("Menggunakan kolom stemmed_text untuk WordCloud")
                        elif 'text_for_wc' in df.columns:
                            st.caption("ℹMenggunakan gabungan kolom title + content untuk WordCloud")
                        else:
                            st.caption(f"ℹMenggunakan kolom {wc_column} untuk WordCloud")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        sentiments = ['Positif', 'Negatif', 'Netral']
                        
                        for col, sentiment in zip([col1, col2, col3], sentiments):
                            with col:
                                st.markdown(f"**{sentiment}**")
                                
                                # Filter data berdasarkan sentimen
                                sentiment_texts = df[df['predicted_sentiment'] == sentiment][wc_column]
                                
                                if len(sentiment_texts) > 0:
                                    # Gabungkan semua teks
                                    all_text = ' '.join(sentiment_texts.fillna('').astype(str).tolist())
                                    
                                    # Jika bukan kolom stemmed_text, bersihkan teks
                                    if wc_column != 'stemmed_text':
                                        words = all_text.split()
                                        words = [w for w in words if len(w) > 2]
                                        all_text = ' '.join(words)
                                    
                                    if all_text.strip():
                                        # Generate WordCloud
                                        wc = WordCloud(
                                            width=800,
                                            height=400,
                                            background_color='#1E1E1E',
                                            colormap='Reds',
                                            max_words=100,
                                            contour_width=1,
                                            contour_color='#E53935',
                                            min_word_length=3,
                                            collocations=False
                                        ).generate(all_text)
                                        
                                        # Tampilkan
                                        fig, ax = plt.subplots(figsize=(6, 4))
                                        ax.imshow(wc, interpolation='bilinear')
                                        ax.axis('off')
                                        fig.patch.set_facecolor('#1E1E1E')
                                        st.pyplot(fig)
                                        plt.close()
                                    else:
                                        st.caption("Tidak ada teks yang cukup")
                                else:
                                    st.caption(f"Tidak ada data sentimen {sentiment}")
                        
                        # Download results
                        st.markdown("### Download Hasil")
                        
                        # Hapus kolom bantuan sebelum download
                        df_download = df.drop(columns=['text_for_wc'], errors='ignore')
                        csv = df_download.to_csv(index=False)
                        
                        st.download_button(
                            label="Download CSV dengan Hasil Analisis",
                            data=csv,
                            file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
            except Exception as e:
                st.error(f"Error membaca file: {e}")
    
    # ============================================================
    # TAB 3: Dataset Sample
    # ============================================================
    with tab3:
        st.markdown('<p class="section-title">Informasi Dataset</p>', unsafe_allow_html=True)
        
        # Sample dataset
        sample_data = {
            'title': [
                'Program Makan Bergizi Gratis Prabowo Sukses di 100 Sekolah',
                'Harga BBM Naik, Masyarakat Menengah Bawah Mengeluh',
                'Pemerintah Evaluasi Program Bantuan Sosial 2025',
                'Investasi Asing Meningkat Berkat Kebijakan Prabowo',
                'Kritik Terhadap Penanganan Banjir di Jakarta'
            ],
            'source': ['Kompas', 'Detik', 'Tempo', 'CNBC', 'Liputan6'],
            'publish_date': ['2025-01-15', '2025-02-20', '2025-03-10', '2025-04-05', '2025-05-12']
        }
        
        sample_df = pd.DataFrame(sample_data)
        sample_df['predicted_sentiment'] = ['Positif', 'Negatif', 'Netral', 'Positif', 'Negatif']
        
        st.markdown("### Contoh Hasil Analisis Dataset:")
        st.dataframe(sample_df, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### Statistik Dataset Penelitian:")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Berita", "1.284")
        with col2:
            st.metric("Positif", "594", "45.7%")
        with col3:
            st.metric("Negatif", "470", "36.2%")
        with col4:
            st.metric("Netral", "224", "18.1%")
        
        # Source distribution
        st.markdown("### Distribusi Sumber Berita:")
        sources = ['Detik', 'Kompas', 'Tempo', 'Liputan6', 'CNBC', 'Kumparan', 'Lainnya']
        source_counts = [250, 220, 180, 160, 140, 200, 150]
        
        fig_sources = px.bar(
            x=sources,
            y=source_counts,
            title="Distribusi Sumber Berita",
            color=source_counts,
            color_continuous_scale='Reds'
        )
        fig_sources.update_layout(
            paper_bgcolor='#1E1E1E',
            plot_bgcolor='#1E1E1E',
            font=dict(color='#BDBDBD'),
            xaxis_title="Sumber",
            yaxis_title="Jumlah Berita"
        )
        st.plotly_chart(fig_sources, use_container_width=True)

if __name__ == "__main__":
    main()