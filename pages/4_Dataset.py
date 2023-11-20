import streamlit as st
import pandas as pd
from collections import Counter
import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'vscode'
pio.templates.default = 'plotly'
import altair as alt
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")

st.title ("Подготовка датасета")

df = pd.read_csv('data/breast-cancer.csv')
#удалим столбец id
df.drop(labels=['id'] , axis=1 , inplace=True)
st.table(df.head())

st.write(f"В датасете: {df.shape[0]} строк, {df.shape[1]} столбец")

#st.write ('Кол-во пропущенных значений')
#st.table(df.isnull().sum())

st.write("Посмотрим описательную статистику:")
st.table(df.describe().T)

#Распределение данных по классам
value_counts = Counter(df['diagnosis'])

labels = list(value_counts.keys())
values = list(value_counts.values())

plt.figure(figsize=(6, 6))
plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)

plt.title('Распределение данных по классам')
st.pyplot(plt)

#перекодируем категориальную переменную
df=df.replace({"M": 1 ,"B":0})# Malligant =1  Benign=0

#Смотрим выбросы
#st.altair_chart(px.box(df))
chart = (
            alt.Chart(df).mark_boxplot()
         )
st.plotly_chart(px.box(df))

st.markdown(
        """
        При добросовесном анализе надо уточнить могут ли данные принимать такие значения или это выбросы. 
        В данном случае примем, что значения реальны.
               
    """
    )




