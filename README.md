# Laporan Proyek Machine Learning
### Nama : Muhammad Rizal Rapeie
### Nim : 211351099
### Kelas : Malam B

## Domain Proyek

Dengan keindahan dan ketahanannya, serta merupakan permata alami dengan nilai yang sangat tinggi. Untuk mempersiapkan masa tua kita nanti alangkah baiknya kita berinvestasi pada permata yang bernama berlian ini?? Berikut adalah applikasi web yang saya kembangkan untuk memprediksi/mengestimasi harga berlian berdasarkan spesifikasi yang anda inginkan.

## Business Understanding

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Semakin sulitnya mencari sarana investasi jangka panjang yang terjamin dan mudah untuk dicairkan maupun diwariskan.
- Dengan budget yang terbatas alangkah baiknya kita mendapatkan harga yang sesuai dengan spesifikasi berlian yang kita inginkan.
- Banyaknya penjual berlian ini memasang dengan harga yang sangat tinggi

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Dengan memudahkan kalian dalam mematok harga untuk investasi akan menjadi lebih mudah untuk mendapatkan harga berlian yang sesuai sehingga untuk membelinya pun menjadi mudah.
- Mendapatkan harga yang sesuai dengan spesifikasi yang diinginkan
- Agar tidak mendapatkan harga yang terlalu tinggi dari penjual, sehingga investasi anda tidak berujung buntung.

## Data Understanding
Disini saya menggunakan datasets Diamond Prices dari kaggle, yang berisikan 10 kolom dan 53,940 baris : <br>
Contoh: [Diamonds Prices](https://www.kaggle.com/datasets/nancyalaswad90/diamonds-prices).

### Variabel-variabel pada Diamonds Prices Dataset adalah sebagai berikut:
- carat     : merupakan jumlah karat yang ada pada berlian.
- cut       : merupakan jenis potongan berlian.
- color     : merupakan warna berlian.
- clarity   : merupakan seberapa beningnya berlian.
- depth     : tinggi dari berlian.
- table     : merupakan seberapa datar sebuah berlian
- price     : merupakan harga berlian
- x         : merupakan panjang berlian
- y         : merupakan lebar berlian
- z         : merupakan tinggi berlian


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

!kaggle datasets download -d nancyalaswad90/diamonds-prices
```
Nah, setelah mendownload datasetsnya, kita extract terlebih dahulu, lalu kita bisa menggunakannya,
``` bash
!unzip diamonds-prices.zip -d diamonds_prices
!ls diamonds_prices
```
Memasukkan library yang akan kita gunakan selama EDA,
``` bash
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
Bisa dilihat ya, di dalam datasetsnya terdapat satu kolom yang tidak diinginkan, yaitu "Unnamed: 0", mari kita hilangkan kolom itu,
``` bash
df = df.drop('Unnamed: 0', axis=1)
```
Lalu kita bisa cek kolom yang tersedia dalam datasets sekarang,
``` bash
df.columns
```
Mari kita hitung terdapat berapa nilai dari color, cut, dan clarity dari masing-masing jenisnya,
``` bash
df["color"].value_counts()
df["cut"].value_counts()
df["clarity"].value_counts()
```
Lalu kita harus memeriksa apakah di dalam datasets terdapat data yang berulang,
```bash
df[df.duplicated()]
```
Nah, terdapat 149 baris yang sama, kita bisa menghapusnya dengan ini,
``` bash
df.drop_duplicates(inplace=True)
```
Lalu mari kita lihat apakah ada kolom yang null/tidak memiliki nilai,
``` bash
sns.heatmap(df.isnull())
```
![download](https://github.com/RizalRapeie/estimasi_harga_berlian/assets/148552041/1751bbcf-ee8b-4f28-b642-4d76dd98475d)
<br>
Terlihat aman ya
<br>
Selanjutnya, melihat apakah ada data terpencil dengan menggunakan boxplot, seperti berikut
``` bash

numeric_cols = ['carat', 'depth','table', 'price', 'x', 'y', 'z']
plt.figure(figsize=(15, 15))
for i in range(7) :
    plt.subplot(3,3,i+1)

    sns.boxplot(x=df[numeric_cols[i]],color='#6DA59D')
    plt.title(numeric_cols[i])
plt.show()
```
![download](https://github.com/RizalRapeie/estimasi_harga_berlian/assets/148552041/08f922ee-af04-4206-a0d7-1766b01fb610)
<br>
Seperti yang bisa dilihat, ada beberapa data terpencil di dalamnya,
<br>
Untuk menghilangkan data-data terpencil itu kita bisa menggunkan, kode berikut
```bash
def detect_outliers(data,column):
    q1 = df[column].quantile(.25)
    q3= df[column].quantile(.75)
    IQR = q3-q1

    lower_bound = q1 - (1.5*IQR)
    upper_bound = q3 + (1.5*IQR)

    ls = df.index[(df[column]  upper_bound)]

    return l
```
Setelah kita mendeteksi data terpencil, kita lanjut dengan menghilangkannya,
``` bash
index_list = []

for column in numeric_cols:
    index_list.extend(detect_outliers(df,column))

# remove duplicated indices in the index_list and sort it
index_list = sorted(set(index_list))

df =df.drop(index_list)
after_remove = df.shape

print(f'Shape of data after remove : {after_remove}')
```
Data terpencil sudah tidak ada sekarang, ini bisa berpengaruh sekali dengan tingkat akurasi model kita, 
Selanjutnya kita akan melihat data dalam bentuk scatter plot,
``` bash
cols = ['carat', 'depth','table', 'x', 'y', 'z']
plt.figure(figsize=(18, 12))
for i in range(6) :
    plt.subplot(2,3,i+1)
    #sns.set()
    plt.scatter(df[cols[i]],df['price'],color='#679C94')
    plt.title(cols[i])
    plt.ylabel('Price',size=13)
plt.show()
```
![download](https://github.com/RizalRapeie/estimasi_harga_berlian/assets/148552041/4b29b68f-f1db-4718-b9c6-00e3b89ebd46)

Terlihat disini bahwa semakin tingginya harga maka semakin tinggi pula jumlah karat, lebar, panjang, tinggi berlian, untuk lebih jelasnya kita bisa menggunakan ini,
``` bash
kualitas = df.groupby('cut').mean().sort_values('price',ascending=False)
kualitas = kualitas[['price']].round(2)
kualitas.reset_index(inplace=True)
kualitas
```
Dari sini terlihat rata-rata berlian dengan potongan yang "Fair" itu lebih mahal dibandingkan berlian dengan potongan "Premium", selanjutnya mari lihat heatmap yang menunjukkan korelasi antar kolom,
``` bash
sns.heatmap(df.corr(),annot = True,cmap='bone')
```
![download](https://github.com/RizalRapeie/estimasi_harga_berlian/assets/148552041/0f37cff7-ae11-47f2-ad04-28d9440a49d6)
Dari ini kita bisa melihat korelasi antara depth dan table sangat buruk ya, ini bisa mempengaruhi model kita nantinya, namun hal itu tidak bisa kita apa-apakan lagi karena semua datanya sudah dilakukan pembersihan, langkah selanjutnya adalah modeling.

## Modeling
Memasukkan semua library yang nanti akan digunakan selama proses modeling
``` bash
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
```
Selanjutnya mari mengubah semua nilai "object" menjadi "int" agar nanti bisa kita olah menjadi sebuah model regressi,
``` bash
df['cut'] = df['cut'].map({'Fair':0, 'Good':1, 'Very Good':2, 'Premium':3, 'Ideal':4})
df['color'] = df['color'].map({'J':0, 'I':1, 'H':2, 'G':3, 'F':4, 'E':5, 'D':6})
df['clarity'] = df['clarity'].map({'I1':0, 'SI2':1, 'SI1':2, 'VS2':3, 'VS1':4, 'VVS2':5, 'VVS1':6, 'IF':7})
```
Lalu mari kita memasukkan fitur-fitur dan target yang kita inginkan,
``` bash
x=df[['carat', 'cut', 'color', 'clarity',
       'x', 'y', 'z']]
y=df[['price']]
```
Selanjutnya mari kita membuat fitur fiturnya terstandarisasi dengan menghilangkan nilai mean-nya dengan menggunakan scaler,
``` bash
scaler = StandardScaler()
x= scaler.fit_transform(x)
```
Selanjutnya kita akan melakukan test dan train split dengan menggunakan train_test_split dengan 20% test dan 80% train
``` bash
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=.2,shuffle=True)
```
Lalu kita akan menggunakan linearRegression untuk membuat model kita
``` bash
model = LinearRegression()

model.fit(x_train,y_train)
y_pred = model.predict(x_test)
```
Selanjutnya mari kita lihat score yang kita dapatkan,
``` bash
print(f''' Akurasi Train : {r2_score(y_train,model.predict(x_train))}
Akurasi Test : {r2_score(y_test , y_pred)}''')
```
Lumayan dapat 91%, mari coba dengan data inputan,
``` bash
data = np.array([[2, 2, 3, 3, 5, 5, 3]])
prediksi = model.predict(data)
print('Estimasi harga Berlian : ', prediksi)
```
Langkah terakhir adalah mengexport model yang telah dihasilkan dengan menggunakan pickle,
``` bash
import pickle

filename = "estimasi_harga_berlian.sav"
pickle.dump(model,open(filename,'wb'))
```
Selesai sudah.

## Evaluation
Metrik evaluasi yang saya gunakan ada F1, dan berikut adalah kodenya,
```
from sklearn.metrics import precision_recall_curve, f1_score

threshold = 3000

y_pred_binary = (y_pred > threshold).astype(int)
y_test_binary = (y_test > threshold).astype(int)

f1 = f1_score(y_test_binary, y_pred_binary)

print('F1 Score:', f1)
```
f1 score yang didapat adalah 94.9% dan itu sudah cukup besar. Yang artinya model ini memiliki performa yang seimbang antara presisi dan recall, serta keyakinan yang tinggi terhadap prediksi. Jadi tidak diragukan lagi hasil prediksi dari model ini cukup tepat.

## Deployment
[Estimasi Harga Berlian](https://estimasi-harga-berlian-rizal.streamlit.app/)
![image](https://github.com/RizalRapeie/estimasi_harga_berlian/assets/148552041/e0ed3569-6131-44a6-af94-3ff30a01c378)

