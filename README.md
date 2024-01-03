# Laporan Proyek Machine Learning
### Nama : Muhammad Rizal Rapeie
### Nim : 211351099
### Kelas : Malam B

## Domain Proyek

Didasari pada datset penjualan barang di sebuah supermarket, yang mana dataset nya tersedia di kaggle. Dengan dataset tersebut kta bisa coba lakukan Market basket analisys dengan apriori supaya kita bisa menentukan keterikatan suatu item pada association rules.

## Business Understanding

Algoritma Apriori adalah salah satu algoritma pada data mining untuk mencari frequent item/itemset pada transaksional database. Algoritma apriori pertama kali diperkenalkan oleh R.Agarwal dan R Srikant untuk mencari frequent tertinggi dari suatu database. 
Algoritma apriori banyak digunakan pada data transaksi atau biasa disebut market basket, misalnya sebuah swalayan memiliki market basket, dengan adanya algoritma apriori, pemilik swalayan dapat mengetahui pola pembelian seorang konsumen. 
Atau algoritma apriori juga bisa digunakan oleh restoran cepat saji dalam menentukan paket menu makanan atau minuman yang ada di restoran mereka berdasakan pola beli konsumen.

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Sulitnya menentukan diskon, paket penjualan karena dalam penjualan supermarket pola pembelian beragam

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Memudahkan menetukan diskon, paket penjualan dan stok barang dengan mempelajari pola pembelian customer

## Data Understanding
Disini saya menggunakan datasets Groceries dataset dari kaggle
[Groceries dataset](https://www.kaggle.com/datasets/heeraldedhia/groceries-dataset).

### Variabel-variabel pada Diamonds Prices Dataset adalah sebagai berikut:
- Member_number      (Unique ID setiap member)(int)
- Date               (Tanggal transaksi)(date)
- itemDescription    (Nama item yang di beli)(str)


## Data Preparation
Teknik yang digunakan untuk mengeksplorasi dan mempersiapkan datasets adalah EDA. Ini dilakukan agar mendapatkan akurasi yang lebih tinggi dengan memastikan tidak adanya data yang berulang atau data yang terpencil. <br>
Tahap pertama adalah mengupload file kaggle lalu mendownload datasets yang kita inginkan, seperti berikut :
``` bash
from google.colab import files
files.upload()
```

```bash
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!ls ~/.kaggle

!kaggle datasets download -d heeraldedhia/groceries-dataset
```
Nah, setelah mendownload datasetsnya, kita extract terlebih dahulu, lalu kita bisa menggunakannya,
``` bash
!mkdir groceries-dataset
!unzip groceries-dataset.zip -d groceries-dataset
!ls groceries-dataset
```
Memasukkan library yang akan kita gunakan selama EDA,
``` bash
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates
from mlxtend.frequent_patterns import association_rules, apriori
```
Lalu membaca file csv yang tadi telah kita extract, dan melihat 5 data paling atasnya
``` bash
df = pd.read_csv("diamonds_prices/Diamonds Prices2022.csv")
df.head()
```
Selanjutnya, mari kita melihat typedata dari masing masing kolom,
``` bash
df.info()
```
Lalu melihat nilai mean, std, min dan count, kita bisa gunakan ini,
``` bash
df.describe()
```
Lalu kita bisa cek kolom yang tersedia dalam datasets sekarang,
``` bash
df.columns
```
Kita bisa melihat top penjualan 10 item
```bash
top_item = df['itemDescription'].value_counts().nlargest(10)

plt.figure(figsize=(12, 6))
sns.countplot(x=df['itemDescription'], data=df, order=top_item.index, palette='viridis')
plt.xlabel('Nama Item')
plt.ylabel('Count')
plt.title('Top 10 Item')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

plt.show()
```
![image](https://github.com/RizalRapeie/ustad/assets/148552041/a3039a25-3ad8-4b95-afda-85fd74168e9e)
Kita juga bisa melihat 10 item dengan pembelian paling sedikit
```bash
low_item = df['itemDescription'].value_counts().nsmallest(10)

plt.figure(figsize=(12, 6))
sns.countplot(x=df['itemDescription'], data=df, order=low_item.index, palette='viridis')
plt.xlabel('Nama Item')
plt.ylabel('Count')
plt.title('10 item paling jarang dibeli')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

plt.show()
```
![image](https://github.com/RizalRapeie/ustad/assets/148552041/29bfc544-5e54-4e1b-a8b2-32acfe2cbb6b)
```
Kita juga bisa mengecek top 10 member
```bash
top_member = df['Member_number'].value_counts().nlargest(10)

plt.figure(figsize=(12, 6))
sns.countplot(x=df['Member_number'], data=df, order=top_member.index, palette='viridis')
plt.xlabel('ID Member')
plt.ylabel('Count')
plt.title('Top 10 Member')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

plt.show()
```
![image](https://github.com/RizalRapeie/ustad/assets/148552041/40baa95d-4550-4b3a-baf2-866bc7a3e18d)
Kita cek angka penjualan harian nya
```bash
plt.figure(figsize=(8,5))
sns.countplot(x='day',data=df)
plt.title('Angka Penjualan Perhari nya')
plt.show()
```
![image](https://github.com/RizalRapeie/ustad/assets/148552041/76a1a591-0816-4299-9596-7514b09c75dc)
dan penjualan perbulanya
```bash
plt.figure(figsize=(8,5))
sns.countplot(x='month',data=df)
plt.title('Angka Penjualan Perbulan nya')
plt.show()
```
![image](https://github.com/RizalRapeie/ustad/assets/148552041/44b5a41a-d94d-4e01-a894-87a1d38cdbda)



## Modeling
```bash
df["itemDescription"] = df["itemDescription"].apply(lambda item: item.lower())
df["itemDescription"] = df["itemDescription"].apply(lambda item: item.strip())
```

Kita buat colom yang hanya terdiri dari member dan item
```bash
df = df[["Member_number", "itemDescription"]].copy()
```
Kita hitung pembelian item setiap membernya
```bash
item_count = df.groupby(["Member_number", "itemDescription"])["itemDescription"].count().reset_index(name="Count")
```
Selanjutnya kita buatkan table pivot
```bash
item_count_pivot = item_count.pivot_table(index='Member_number', columns='itemDescription', values='Count', aggfunc='sum').fillna(0)
print("ukuran dataset : ", item_count_pivot.shape)
```
kita rubah table pivot nya menjadi bertipe int
```bash
item_count_pivot = item_count_pivot.astype("int32")
```
Kita lakukan encode untuk setiap item yang di hitung, jika item tersebut terbeli maka = 1 jika tidak = 0
```bash
def encode(x):
    if x <=0:
        return 0
    elif x >= 1:
        return 1

item_count_pivot = item_count_pivot.applymap(encode)
item_count_pivot.head()
```
Kita tentukan batas nilai support dan masukan apriori nya
```bash
support = 0.02
frequent_items = apriori(item_count_pivot, min_support=support, use_colnames=True)
frequent_items.sort_values("support", ascending=False).head(10)
```
![Screenshot (101)](https://github.com/RizalRapeie/ustad/assets/148552041/d9de267c-3f5b-4156-b529-ac1043aa2e90)




## Evaluation

Kita lakukan evaluasi dengan menampilakn nilai lift ratio dan nilai confidence nya
```bash
metric = "lift"
min_treshold = 1

rules = association_rules(frequent_items, metric=metric, min_threshold=min_treshold)[["antecedents","consequents","support","confidence","lift"]]
rules.sort_values('confidence', ascending=False,inplace=True)
rules.head(15)
```
![Screenshot (102)](https://github.com/RizalRapeie/ustad/assets/148552041/1dbe3be0-9d97-468a-9ca4-26e71e024d38)


## Deployment
[MBA Groceries dataset](https://mba-apri.streamlit.app/)
![image](https://github.com/RizalRapeie/ustad/assets/148552041/9214c69b-7b8b-4279-83b6-6c8d18c007bb)
