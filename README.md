# Turkce-Yorum-Analizi
# Türkçe Müşteri Yorumları Sınıflandırma ve Duygu Analizi Projesi

Bu proje, müşteri yorumlarını analiz ederek hem **çoklu etiket sınıflandırması (Multi-Label Classification)** ile ilgili kategorilere ayıran hem de Hugging Face Transformers kütüphanesi ile **Duygu Analizi (Sentiment Analysis)** gerçekleştiren uçtan uca bir veri bilimi çalışmasıdır.

## Proje Kapsamı
Bu sistem; temizlenmemiş ham yorum verilerini alarak veri önişleme süreçlerinden geçirir, makine öğrenmesi modelleriyle kategorize eder ve derin öğrenme ile duygu skorlarını hesaplar.

* **Metin Önişleme (Text Preprocessing):** Linkler, etkisiz kelimeler (stopwords), noktalama işaretleri ve sayılar temizlenerek veri analize hazır hale getirilmiştir.
* **Model Karşılaştırma (Benchmarking):** Logistic Regression, Random Forest, XGBoost, LightGBM ve CatBoost modelleri 5-Fold Cross Validation ile test edilmiştir.
* **Duygu Analizi (Sentiment Analysis):** Hugging Face pipeline kullanılarak yorumların pozitif/negatif duygu durumları ve güven skorları hesaplanmıştır.

## Proje İçeriği
* **main.py:** Veriyi okuyan, modelleri eğiten, ham verileri tahminleyen ve sonuçları dışa aktaran ana script.
* **Çoklu Sınıflandırma:** Teknik Sorunlar, Fiyat ve Abonelik, Genel Memnuniyet ve İçerik Kalitesi kategorilerinde tahminleme.
* **Veri Çıktısı:** Tahminleme ve duygu analizi sonuçları birleştirilerek Excel formatında dışa aktarılmaktadır.
