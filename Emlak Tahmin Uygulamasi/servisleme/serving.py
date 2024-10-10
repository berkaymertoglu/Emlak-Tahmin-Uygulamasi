import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor

kat = pd.read_csv("bulundugu_kat.csv")
oda = pd.read_csv("oda_sayisi.csv")
isitma = pd.read_csv("isitma_tipi.csv")
yas = pd.read_csv("binanin_yasi.csv")
site = pd.read_csv("site_icerisinde.csv")
esya = pd.read_csv("esya_durumu.csv")
balkon = pd.read_csv("balkon_durumu.csv")
brüt_m2 = pd.read_csv("brüt_metrekare.csv")
net_m2 = pd.read_csv("net_metrekare.csv")
kat_sayisi = pd.read_csv("kat_sayisi.csv")

def balkon_durum(balkon_durumu):
    if balkon_durumu.lower() == "var":
        return 1
    elif balkon_durumu.lower() == "yok":
        return 0
    else:
        print("Hata: Geçersiz Giriş!")

def bulunulan_kat(bulundugu_kat):
    index_no = int(kat[kat["Bulunduğu Kat"] == bulundugu_kat].index.values)
    return index_no

def bina_yas(binanin_yasi):
    index_no = int(yas[yas["Binanın Yaşı"] == binanin_yasi].index.values)
    return index_no
    

def isitma_tip(isitma_tipi):
    index_no = int(isitma[isitma["Isıtma Tipi"] == isitma_tipi].index.values)
    return index_no

def oda_sayi(oda_sayisi):
    index_no = int(oda[oda["Oda Sayısı"] == oda_sayisi].index.values)
    return index_no

def esya_durum(esya_durumu):
    if esya_durumu.lower() == "eşyalı":
        return 1
    elif esya_durumu.lower() == "boş":
        return 0
    else:
        print("Hata: Geçersiz Giriş!")

def site_ici(site_icerisinde):
    if site_icerisinde.lower() == "evet":
        return 1
    elif site_icerisinde.lower() == "hayır":
        return 0
    else:
        print("Hata: Geçersiz Giriş!") 


st.set_page_config(page_title = "Emlak Tahmin Uygulaması")

tabs = ["Emlak Tahmin Uygulaması"]

page = st.sidebar.radio("Sekmeler", tabs)

if page == "Emlak Tahmin Uygulaması":
    st.markdown("<h1 style = 'text-align: center;' > Emlak Tahmin Uygulaması</h1>", unsafe_allow_html=True)


brüt_m2_1 = st.number_input("Brüt Metrekare")
net_m2_1 = st.number_input("Net Metrekare")
bina_yasi_1 = bina_yas(st.selectbox("Bina Yaşı", ["0 (Yeni)", "1", "2", "3", "4", "5-10", "11-15", "16-20", "21 Ve Üzeri"]))
bulundugu_kat_1 = bulunulan_kat(st.selectbox("Binanın Bulunduğu Kat", ["1.Kat", "2.Kat", "3.Kat","4.Kat","5.Kat","6.Kat","7.Kat","8.Kat","9.Kat","10.Kat",
                                                       "11.Kat","12.Kat","13.Kat","14.Kat","17.Kat","18.Kat","19.Kat","20.Kat","22.Kat","24.Kat",
                                                       "Bahçe Katı","Bahçe Dubleks","Bodrum Kat", "Düz Giriş (Zemin)", "Kot 1 (-1).Kat", "Kot 2 (-2).Kat","Kot 4 (-4).Kat",
                                                       "Müstakil", "Yüksek Giriş","Çatı Dubleks", "Çatı Katı"]))
kat_sayisi_1 = st.selectbox("Binanın Kat Sayısı", ["1","2","3", "4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","22","23","24","25","26","27","29","30","31","33","35","39"])
balkon_durumu_1 = balkon_durum(st.radio("Balkon Durumu", ("Var", "Yok")))
esya_durumu_1 = esya_durum(st.radio("Eşya Durumu", ["Eşyalı","Boş"]))
oda_sayisi_1 = oda_sayi(st.selectbox("Oda Sayısı", ["Stüdyo", "1 Oda", "1+1", "1.5+1", "2+1", "2.5+1", "3+1", "3.5+1", "2+2", "3+2", "4+1", "4+2","5+1", "5+2", "6+2"]))
site_icerisinde_1 = site_ici(st.radio("Site İçerisinde", ("Evet", "Hayır")))
isitma_tipi_1 = isitma_tip(st.selectbox("Isıtma Tipi", ["Kombi Doğalgaz", "Doğalgaz Sobalı", "Merkezi Doğalgaz", "Merkezi (Pay Ölçer)", "Kat Kaloriferi",
                                           "Sobalı", "Elektrikli Radyatör", "Yerden Isıtma", "Klimalı", "Kombi Fueloil", "Isıtma Yok"]))

model_sec = st.selectbox("Tahmin Modeli Seciniz", ["Decision Tree Model", "K-Nearest Neighbors Model", "Random Forest Model"])

  

def tahmin_deger(balkon_durumu, kat_sayisi, binanin_yasi, brüt_m2, bulundugu_kat, esya_durumu, isitma_tipi, net_m2, oda_sayisi, site_icerisinde):
    pred = pd.DataFrame(data =
                        {"Balkon Durumu": [balkon_durumu], "Kat Sayısı": [kat_sayisi],
                         "Binanın Yaşı": [binanin_yasi], "Brüt Metrekare": [brüt_m2],
                         "Bulunduğu Kat": [bulundugu_kat], "Eşya Durumu": [esya_durumu],
                         "Isıtma Tipi": [isitma_tipi], "Net Metrekare": [net_m2],
                         "Oda Sayısı" : [oda_sayisi], "Site İçerisinde": [site_icerisinde]})
    return pred


dt_model = joblib.load("decision_tree_model.pkl")
knn_model = joblib.load("k_nearest_neighbors_model.pkl")
rf_model = joblib.load("random_forest_model.pkl")
   
pred_value = tahmin_deger(balkon_durumu_1, kat_sayisi_1, bina_yasi_1, brüt_m2_1, bulundugu_kat_1, esya_durumu_1, isitma_tipi_1, net_m2_1, oda_sayisi_1, site_icerisinde_1)

if st.button("Tahmin Et"):
    if model_sec == "Decision Tree Model":
        tahmin = dt_model.predict(pred_value)
    elif model_sec == "K-Nearest Neighbors Model":
        tahmin = knn_model.predict(pred_value)
    elif model_sec == "Random Forest Model":
        tahmin = rf_model.predict(pred_value)
    else:
        st.error("Hata: Geçersiz Model Tipi Seçildi")
    st.success(f"Tahmin Sonucu: {tahmin[0]}")
