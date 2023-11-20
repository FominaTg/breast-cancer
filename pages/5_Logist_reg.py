import streamlit as st
import joblib
import pandas as pd
#from utils import evaluate
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data/breast-cancer.csv')
#удалим столбец id
df.drop(labels=['id'] , axis=1 , inplace=True)

# загрузка данных для тестирования моделей
st.write("Загрузите тестовую выборку (X_test)")
uploaded_file_test = st.file_uploader("Загрузите файл данных (CSV)", type=["csv"], key='test')
st.write("Загрузите файл, с которым сравниваем (y_test)")
uploaded_file_y = st.file_uploader("Загрузите файл данных (CSV)", type=["csv"], key='classes')

def evaluate(model, test_data, y_data):
    
    y_test_pred = model.predict(test_data)
    
    # st.write("Предсказание модели на текущих данных" + y_test_pred)

    st.write("РЕЗУЛЬТАТЫ НА ВЫБОРКЕ: \n===============================")
    clf_report = pd.DataFrame(classification_report(y_data, y_test_pred, output_dict=True))
    st.table(clf_report)
    st.write(f"МАТРИЦА ОШИБОК (CONFUSION MATRIX):\n{confusion_matrix(y_data, y_test_pred)}")
    st.write(f"ACCURACY ПАРАМЕТР:\n{accuracy_score(y_data, y_test_pred):.4f}")
    st.write(f"PRECISION ПАРАМЕТР:\n{precision_score(y_data, y_test_pred):.4f}")
    st.write(f"RECALL ПАРАМЕТР:\n{recall_score(y_data, y_test_pred):.4f}")
    st.write(f"F1 МЕРА:\n{f1_score(y_data, y_test_pred):.4f}")


if uploaded_file_test is not None and uploaded_file_y is not None:
    
    data_test = pd.read_csv(uploaded_file_test)
    st.table(data_test.head())
    data_y = pd.read_csv(uploaded_file_y)
    st.table(data_y.head())
    model = joblib.load('lr.pkl')
    evaluate(model, data_test, data_y)

