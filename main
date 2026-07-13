#Utils
import re
import nltk
import pandas as pd
import time
import torch
from nltk.corpus import stopwords, wordnet
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from transformers import pipeline

# Paketler
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def preprocessing(series, **kwargs):
    stop_kelimeler = set(stopwords.words('turkish'))
    if kwargs.get('lowercase'):
        series = series.str.lower()
    if kwargs.get('remove_links'):
        series = series.str.replace(r'\n', '', regex=True)
        series = series.apply(lambda x: re.split(r'https:\/\/.*', str(x))[0])
    if kwargs.get('remove_punctuation'):
        series = series.str.replace(r'[^\w\s]', '', regex=True)
    if kwargs.get('remove_stopwords'):
        series = series.apply(lambda x: ' '.join([w for w in str(x).split() if w.lower() not in stop_kelimeler]))
    if kwargs.get('remove_numbers'):
        series = series.str.replace(r'\d+', '', regex=True)
    return series


def tr_en_char_translate(series):
    return series.str.replace('ı', 'i').str.replace('ü', 'u').str.replace('ö', 'o').str.replace('ğ', 'g').str.replace(
        'ş', 's').str.replace('ç', 'c')


def create_tfidf(series):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=2500, min_df=2)
    return vectorizer.fit_transform(series).toarray()


def get_models(X, y, classification=False, info=True):
    models = [
        ('LR', LogisticRegression()),
        ('RF', RandomForestClassifier()),
        ('XGB', XGBClassifier(verbosity=0)),
        ('LGBM', LGBMClassifier(verbose=-1)),
        ('CatBoost', CatBoostClassifier(verbose=0))
    ]
    results = []
    for name, model in models:
        if info: print(f"{name} is training...")
        if classification:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results.append([name, acc])
        else:
            score = cross_val_score(model, X, y, cv=5).mean()
            results.append([name, score])

    df_res = pd.DataFrame(results, columns=['name', 'acc_test']).sort_values(by='acc_test', ascending=False)
    print("\nAll models are done --- ordered by acc")
    print(df_res)
    return df_res


def sentiment(text_list):
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline("sentiment-analysis", device=device)

    # 10.500 satırlık veriyi metne (string) çevirip listeye aktarıyoruz
    results = classifier(text_list.astype(str).tolist())

    # Sonuçları DataFrame'e uygun formata getirme
    labels = [res['label'] for res in results]
    scores = [res['score'] for res in results]
    return pd.DataFrame({'label': labels, 'score': scores})

#Makine Öğrenmesi
import pandas as pd
from utils import *

dataframe = pd.read_excel('BushraShaabanB231306560EtiketlenmişVeriSeti.xlsx')

dataframe = dataframe[['Yorum', 'Kategori']].copy()

dataframe.head()

dataframe.Kategori.value_counts()


dataframe['teknik'] = [1 if x == 'Teknik Sorunlar' else 0 for x in dataframe.Kategori]
dataframe.teknik
dataframe['fiyat'] = [1 if x == 'Fiyat ve Abonelik' else 0 for x in dataframe.Kategori]
dataframe.fiyat
dataframe['memnuniyet'] = [1 if x == 'Genel Memnuniyet' else 0 for x in dataframe.Kategori]
dataframe.memnuniyet
dataframe['icerik'] = [1 if x == 'İçerik Kalitesi' else 0 for x in dataframe.Kategori]
dataframe.icerik

dataframe.teknik.value_counts()


#veri önişleme
dataframe['islenmis_veri'] = preprocessing(dataframe.Yorum,
                                           remove_links=True,
                                           remove_mentions=True,
                                           remove_stopwords=True,
                                           remove_hashtag=True,
                                           lowercase=True,
                                           remove_numbers=True,
                                           remove_punctuation=True)

dataframe['duygu_analizi'] = preprocessing(dataframe.Yorum,
                                           remove_links=True,
                                           remove_mentions=True,
                                           remove_hashtag=True,
                                           lowercase=True,
                                           remove_numbers=True,
                                           remove_punctuation=True)

dataframe.head()

dataframe.columns

#makine öğrenmesi
tf_idf_set = create_tfidf(dataframe['islenmis_veri'])
tf_idf_set

# Temizleme yapmadan önceki ham kelime sayısını bulalım
ham_kelimeler = " ".join(dataframe['Yorum'].astype(str)).split()
print(f"Temizleme öncesi toplam kelime sayısı: {len(ham_kelimeler)}")
print(f"Temizleme öncesi benzersiz (farklı) kelime sayısı: {len(set(ham_kelimeler))}")

get_models(tf_idf_set, dataframe.teknik, classification=True, info=True)
#en iyi sonuç RF
get_models(tf_idf_set, dataframe.fiyat, classification=True, info=True)
#en iyi sonuç RF #####
get_models(tf_idf_set, dataframe.memnuniyet, classification=True, info=True)
#en iyi sonuç CatBoost
get_models(tf_idf_set, dataframe.icerik, classification=True, info=True)
#en iyi sonuç XGB

ham_veri = pd.read_excel('BushraShaabanB231306560HamVeri.xlsx')

ham_yorumlar = ham_veri['Yorum']

ham_yorumlar

ham_veri['temizlenmiş'] = preprocessing(dataframe.Yorum,
                                           remove_links=True,
                                           remove_mentions=True,
                                           remove_stopwords=True,
                                           remove_hashtag=True,
                                           lowercase=True,
                                           remove_numbers=True,
                                           remove_punctuation=True)

ham_veri['duygu_analizi'] = preprocessing(ham_veri.Yorum,
                                           remove_links=True, remove_mentions=True, remove_hashtag=True,
                                           lowercase=True, remove_numbers=True, remove_punctuation=True)

ham_veri.head()
#ham_veri = ham_veri.dropna(subset=['duygu_analizi'])

#Makine Öğrenmesi
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorized = TfidfVectorizer(analyzer='word')
tf_idf_donusturucu = tfidf_vectorized.fit(dataframe['islenmis_veri'])

ogrenme_seti = tf_idf_donusturucu.transform(dataframe['islenmis_veri'])
ogrenme_seti_vektoru = pd.DataFrame(data=ogrenme_seti.toarray(),
                                   columns=tf_idf_donusturucu.get_feature_names_out())

tahmin_seti = tf_idf_donusturucu.transform(ham_veri['duygu_analizi'].astype(str))
tahmin_seti_vektoru = pd.DataFrame(data=tahmin_seti.toarray(),
                                   columns=tf_idf_donusturucu.get_feature_names_out())

print(tahmin_seti_vektoru)
print(ogrenme_seti_vektoru)

tf_idf_donusturucu.get_feature_names_out()

#Eğitim ve Tahminleme
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

model_teknik = RandomForestClassifier()
model_teknik.fit(ogrenme_seti_vektoru, dataframe.teknik)
ham_veri['teknik_tahmin'] = model_teknik.predict(tahmin_seti_vektoru)
print(ham_veri['teknik_tahmin'].value_counts())

model_fiyat = RandomForestClassifier()
model_fiyat.fit(ogrenme_seti_vektoru, dataframe.fiyat)
ham_veri['fiyat_tahmin'] = model_fiyat.predict(tahmin_seti_vektoru)
print(ham_veri['fiyat_tahmin'].value_counts())

model_memnuniyet = CatBoostClassifier(verbose=0)
model_memnuniyet.fit(ogrenme_seti_vektoru, dataframe.memnuniyet)
ham_veri['memnuniyet_tahmin'] = model_memnuniyet.predict(tahmin_seti_vektoru)
print(ham_veri['memnuniyet_tahmin'].value_counts())

model_icerik = XGBClassifier()
model_icerik.fit(ogrenme_seti_vektoru, dataframe.icerik)
ham_veri['icerik_tahmin'] = model_icerik.predict(tahmin_seti_vektoru)
print(ham_veri['icerik_tahmin'].value_counts())

ham_veri[['label', 'score']] = sentiment(ham_veri.duygu_analizi)

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

print(ham_veri[['Yorum', 'teknik_tahmin', 'label', 'score']].tail())

# Tüm tahmin sütunlarını ve duygu analizini bir arada görelim
print(ham_veri[['Yorum', 'teknik_tahmin', 'fiyat_tahmin', 'memnuniyet_tahmin', 'icerik_tahmin', 'label']].tail(10))

ham_veri_final = ham_veri.drop(['temizlenmiş', 'duygu_analizi'], axis=1)
print(ham_veri_final.columns)

ham_veri_final.to_excel('BushraShaabanB231306560.OdevTeslim.xlsx', index=False)
