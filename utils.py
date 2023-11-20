# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import textwrap

import streamlit as st
import pandas as pd

from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split


def show_code(demo):
    """Showing the code of the demo."""
    show_code = st.sidebar.checkbox("Show code", True)
    if show_code:
        # Showing the code of the demo.
        st.markdown("## Code")
        sourcelines, _ = inspect.getsourcelines(demo)
        st.code(textwrap.dedent("".join(sourcelines[1:])))

def get_df():
    # чтение датасета
    df = pd.read_csv('data/breast-cancer.csv')
    #удалим столбец id
    df.drop(labels=['id'] , axis=1 , inplace=True)

    return df

#def bagging_model_load():
 #   loaded = joblib.load('bagging_model.pkl') #классическое  машинное
    #h5, keras - для нейронок
  #  return loaded


#def knn_model_load():
    #загрузка файла модели knn
 #   knn_loaded = joblib.load('knn_best_model.pkl') #классическое  машинное
    #h5, keras - для нейронок
  #  return knn_loaded

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
    st.write(f"ОТЧЕТ О КЛАССИФИКАЦИИ:\n{clf_report}")

  