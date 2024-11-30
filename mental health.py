#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install numpy --upgrade --user


# In[2]:


#pip install upgrade pip


# In[3]:


#import numpy
#print(numpy. __version__)


# In[4]:


#pip install catboost
#pip install lightgbm
#pip install perceptron
#pip install SVC
#pip install perceptron
#pip install xgboost
#pip install SVC
#pip install --upgrade scikit-learn
#pip install --upgrade --user scikit-learn
#pip install threadpoolctl
#pip install --upgrade threadpoolctl


# In[5]:


#pip install --upgrade --user threadpoolctl


# In[6]:


#pip install optuna


# In[7]:


from sklearn.linear_model import LogisticRegression, RidgeClassifierCV


# In[8]:




import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder

from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,classification_report

from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

import plotly.express as px
import plotly.graph_objects as go


#
#
#
# Create for very easily
# for i in tqdm(range(10000)):

from tqdm import tqdm


print('Carregou as bibliotecas...')


# In[9]:


from sklearn.ensemble import GradientBoostingClassifier


# In[10]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[11]:


import optuna


# In[12]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[13]:


# Com os dois imports feitos abaixo. OneHotEncoder para criar features booleanas com os campos.
# O outro import trás o decomposition do PCA ( Principal Component Analisys ) que auxilia na análise preliminar e suas correlações.

from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA


# In[14]:


from sklearn.metrics import precision_score


# In[15]:


#print(sklearn.__version__)


# In[16]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# In[17]:


final_depression_dataset_1 = pd.read_csv('final_depression_dataset_1.csv')


# In[18]:


target = 'Depression'

numerical_columns = [
    "Age", "Academic Pressure", "Work Pressure", "CGPA",
    "Study Satisfaction", "Job Satisfaction", "Work/Study Hours",
    "Financial Stress"
]

one_hot_columns = [
    "Gender", "Working Professional or Student", "City", "Family History of Mental Illness"
]

label_columns = [
    "Degree", "Profession", "Dietary Habits", "Have you ever had suicidal thoughts ?", "Sleep Duration"
]


# In[19]:


test


# In[20]:


#pip install ydata-profiling --user
#pip install -U ydata-profiling


# In[21]:


#pip install ydata-profiling==4.8.3 --user


# In[22]:


#pip install ydata-profiling


# In[23]:


from ydata_profiling import ProfileReport


# In[24]:


#print(ProfileReport)


# In[25]:


target


# In[26]:


final_depression_dataset_1.describe()


# In[27]:


final_depression_dataset_1.isnull().sum()


# In[28]:


#Uma relação à ser estudada. Pressão academica, sem profissão, CGPA, satisfação no estudos e Grau estão vazios de forma associada?
final_depression_dataset_1[ final_depression_dataset_1['Academic Pressure'].isnull() == True ]


# In[29]:


#Como não tem trabalho. Também não tem pressão no trabalho e nem satisfação no emprego.

final_depression_dataset_1[ final_depression_dataset_1['Work Pressure'].isnull() == True ]


# In[30]:


final_depression_dataset_1[ final_depression_dataset_1['CGPA'].isnull() == True ]


# In[31]:


#Avaliar se Academic Pressure e CGPA e Study Satisfaction estão interligados.
final_depression_dataset_1[ final_depression_dataset_1['Study Satisfaction'].isnull() == True ]


# In[32]:


# Profession e Work Pressure e Job Satisfaction estão interligados?
final_depression_dataset_1[ final_depression_dataset_1['Job Satisfaction'].isnull() == True ]


# In[33]:


final_depression_dataset_1.dtypes


# In[34]:


profile = ProfileReport(final_depression_dataset_1[numerical_columns], title="Relatório básico")
print(final_depression_dataset_1[numerical_columns])


# In[35]:


"""
numerical_columns
one_hot_columns
label_columns
"""


# In[36]:


enc = OneHotEncoder(handle_unknown='ignore')


# In[37]:


#result = enc.fit_transform(final_depression_dataset_1[one_hot_columns]).toarray()


# In[38]:


#print(result)


# In[39]:


dados = pd.get_dummies(final_depression_dataset_1, columns = one_hot_columns )


# In[40]:


test['Dietary Habits']


# In[41]:


dados_test = pd.get_dummies(test, columns = one_hot_columns )


# In[42]:


dados_test['Dietary Habits']


# In[43]:


dados_test = pd.concat([dados_test, test[label_columns]], axis=1)


# In[44]:


dados_test


# In[ ]:





# In[45]:


one_hot_columns


# In[46]:


final_depression_dataset_1[label_columns]


# In[47]:


final_depression_dataset_1[numerical_columns]


# In[48]:


dados[label_columns]


# In[49]:


profile


# In[50]:


dados_numericos= dados.select_dtypes(exclude=['object', 'string'])


# In[51]:


dados_numericos_test = dados_test.select_dtypes(exclude=['object', 'string'])


# In[52]:


dados_numericos.shape


# In[53]:


dados_numericos_test[dados_numericos.columns].shape


# In[54]:


label_columns


# In[55]:


dados_numericos_test


# In[56]:


dados_numericos


# In[57]:


dados_test[dados_numericos.columns]


# In[58]:


dados_test['Dietary Habits']


# In[59]:


dados_numericos.columns


# In[60]:


dados_numericos_test=pd.concat([dados_numericos_test, dados_test], axis=1)


# In[61]:


dados_test['Dietary Habits']


# In[62]:


dados_test[label_columns]


# In[63]:


dados_numericos_test


# In[64]:


dados_test


# In[65]:


dados_test[label_columns].shape


# In[66]:


dados_numericos_test.shape


# In[67]:


dados_numericos.isnull().sum()


# In[68]:


dados_numericos = dados_numericos.fillna(0)


# In[69]:


dados_test = dados_numericos_test.fillna(0)


# In[70]:


dados_numericos_test['Dietary Habits']


# In[71]:


dados_test['Dietary Habits']
dados_test


# In[72]:


dados_test


# In[73]:


scaler = StandardScaler()
data_scaled = scaler.fit_transform(dados_numericos)


# In[74]:


pca = PCA(n_components=2)


# In[75]:


pca_transformed = pca.fit_transform(data_scaled)


# In[76]:


pca_transformed


# In[77]:


pca_df = pd.DataFrame(data=pca_transformed, columns=["PC1", "PC2"])


# In[78]:


dados_numericos


# In[79]:


pca_df_final = pd.concat([pca_df, final_depression_dataset_1['Depression']], axis=1)


# In[80]:


base = pd.concat([dados_numericos, final_depression_dataset_1['Depression']], axis=1)


# In[81]:


base_test = dados_test


# In[82]:


dados_numericos


# In[83]:


dados_test


# In[84]:


base


# In[85]:


amostra = base[ base['Depression'] == 'Yes' ]


# In[86]:


amostra


# In[87]:


#Não existe caso de depressão que não tenha pressão acadêmica ou pressão no trabalho
#olhando para a coluna pressure parece que todas as colunas foram preenchidas. Independente se tem ou não depression is true.

amostra.loc[(amostra['Academic Pressure'] == 0) | (amostra['Work Pressure'] == 0), ['Academic Pressure','Work Pressure']]


# In[88]:


amostra.loc[:,['Academic Pressure','Work Pressure']]


# In[89]:


amostra[['Academic Pressure','Work Pressure']]


# In[90]:


amostra.shape


# In[91]:


#Vamos criar ,mais uma feature caso haja valor positivo em Academic Pressure ou Work Pressure este campo será positivo.
#amostra[(amostra['Academic Pressure'] == 0) | (amostra['Work Pressure'] == 0)]

amostra['pressure'] = amostra['Academic Pressure']+amostra['Work Pressure']


# In[92]:


amostra_negativa = base[base['Depression'] == 'No']


# In[93]:


amostra_negativa[(amostra_negativa['Academic Pressure'] == 0) & (amostra_negativa['Academic Pressure'] == 0)]


# In[94]:


base.loc[(base['Academic Pressure'] == 0) | (base['Work Pressure'] == 0), ['Academic Pressure','Work Pressure']]


# In[95]:


amostra[ (amostra['pressure'] == 0)  ]


# In[96]:


sum_academic_work = pd.DataFrame(amostra)


# In[97]:


mapa_correlacao = sns.heatmap(amostra.corr(), annot = True, fmt=".1f", linewidths=.1)


# In[98]:



#mapa_correlacao.fig.set_dpi(100)
#mapa_correlacao


# In[99]:


pca_df_final


# In[100]:


pca_df


# In[101]:


#pip install pyqt5 --user


# In[102]:


# Garantir que os gráficos sejam exibidos no notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# In[103]:


plt.figure(figsize=(8, 6))
plt.scatter(pca_df["PC1"], pca_df["PC2"], color='blue', edgecolor='k', s=100)
plt.title("PCA - Componentes Principais", fontsize=16)
plt.xlabel("Componente Principal 1 (PC1)", fontsize=14)
plt.ylabel("Componente Principal 2 (PC2)", fontsize=14)
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.grid(alpha=0.3)
plt.show()


# In[104]:


n_componentes = 2

sns.pairplot(
pca_df_final, vars  = ['PC1','PC2'],
hue='Depression', diag_kind="hist"
)
plt.show()


# In[105]:


n_componentes = 2

sns.pairplot(
pca_df_final, vars  = ['PC1','PC2'],
hue='Depression', diag_kind="hist")
plt.show()


# In[106]:


sns.heatmap(dados_numericos.corr())


# In[107]:


#px.pie(target)
#px.pie_chart(target)


# In[108]:


#Fase de modelagem, treino, teste e medição.


# In[109]:


dados_numericos


# In[110]:


pca_df


# In[111]:


final_depression_dataset_1[ final_depression_dataset_1['Depression'] == 'Yes' ]


# In[112]:


base


# In[113]:


base.corr()


# In[114]:


base.columns


# In[115]:


sns.heatmap(base.corr(numeric_only=True),cmap="Blues",annot=True)


# In[116]:


#Vamo olhar algumas relações
sns.jointplot(data=base, x="Work/Study Hours", y="Academic Pressure", hue="Depression")


# In[117]:


#Vamo olhar algumas relações
sns.jointplot(data=base, x="Age", y="Work Pressure", hue="Depression")


# In[118]:


#Vamo olhar algumas relações
sns.jointplot(data=base, x="Age", y="Work/Study Hours", hue="Depression")


# In[119]:


#Vamo olhar algumas relações
sns.jointplot(data=base, x="Age", y="Study Satisfaction", hue="Depression")


# In[120]:


#Vamo olhar algumas relações
sns.jointplot(data=base, x='Work Pressure', y='Financial Stress', hue="Depression")


# In[121]:


base.plot.hist(column=['Academic Pressure', 'Work Pressure', 'CGPA', 'Study Satisfaction', 'Job Satisfaction', 'Work/Study Hours','Financial Stress'])


# In[122]:


base


# In[123]:


base.loc[(base['Job Satisfaction'] != '0.0') & (base['Study Satisfaction'] != '0.0')]


# In[124]:


final_depression_dataset_1['Profession'].unique()


# In[125]:


sleep_duration = pd.get_dummies(final_depression_dataset_1['Sleep Duration'])


# In[126]:


sleep_duration


# In[127]:


#base_test = test['Sleep Duration'].unique()
#base_test = 


# In[128]:


#base_test


# In[129]:


#test['Sleep Duration'].drop_duplicates(subset=['Sleep Duration'])


# In[130]:


dados_test


# In[131]:


base_test


# In[132]:


#dados_test['Sleep Duration'] = dados_test['Sleep Duration'].unique()


# In[133]:


dados_test['Sleep Duration']


# In[134]:


#dados_test['Sleep Duration'] = dados_test.drop_duplicates(subset=['Sleep Duration'])
#dados_test['Sleep Duration']


# In[135]:


sleep_duration_test = pd.get_dummies(dados_test['Sleep Duration'])
sleep_duration_test


# In[136]:


base_test


# In[137]:


sleep_duration_test[sleep_duration_test.duplicated()==True]


# In[138]:


sleep_duration_test = sleep_duration_test.T.drop_duplicates().T


# In[139]:


sleep_duration_test[sleep_duration_test.duplicated()==True]


# In[140]:


sleep_duration_test = sleep_duration_test.loc[:, ~sleep_duration_test.columns.duplicated()]


# In[141]:


sleep_duration_test.columns


# In[142]:


base_test.columns


# In[143]:


base_test[base_test.duplicated()==True]


# In[144]:


sleep_duration_test.shape


# In[145]:


base_test.shape


# In[146]:


# Remover índices duplicados (mantendo apenas a primeira ocorrência)
base_test = base_test[~base_test.index.duplicated()]
sleep_duration_test = sleep_duration_test[~sleep_duration_test.index.duplicated()]


# In[147]:


# Verificar duplicatas no índice
print("Duplicados em base_test:", base_test.index.duplicated().any())
print("Duplicados em sleep_duration_test:", sleep_duration_test.index.duplicated().any())


# In[148]:


print("Índice de base_test:", base_test.index)
print("Índice de sleep_duration_test:", sleep_duration_test.index)


# In[149]:


#sleep_duration_test
base_test = pd.concat( [base_test, sleep_duration_test], axis=1 )


# In[150]:


base_test


# In[151]:


#dados_test


# In[152]:


#base_test = base_test.drop('Unhealthy')
#base_test


# In[153]:


#base_test = sleep_duration_test
#dados_test = pd.concat( [dados_test, sleep_duration_test], axis=1 )
base_test


# In[154]:


sleep_duration_test


# In[155]:


#base_test = pd.get_dummies(dados_test['Sleep Duration'])


# In[156]:


#base_test = pd.concat( base_test, pd.get_dummies(final_depression_dataset_1['Sleep Duration']) )


# In[157]:


base.shape


# In[158]:


sleep_duration.shape


# In[159]:


base_test.shape


# In[160]:


base['suicide'] = final_depression_dataset_1['Have you ever had suicidal thoughts ?'].replace({"Yes":1,"No":0})


# In[161]:


base_test


# In[162]:


base_test['suicide'] = test['Have you ever had suicidal thoughts ?'].replace({"Yes":1,"No":0})


# In[163]:


base_test


# In[164]:


dietarie_habits = pd.get_dummies(final_depression_dataset_1['Dietary Habits'])


# In[165]:


#dados_test['Dietary Habits']
dietarie_habits


# In[166]:


base_test


# In[167]:


dados_test_dietarie_habits = pd.get_dummies(test['Dietary Habits'])


# In[168]:


dados_test_dietarie_habits


# In[169]:


dados_test_dietarie_habits = dados_test_dietarie_habits.loc[:, ~dados_test_dietarie_habits.columns.duplicated()]


# In[170]:


dados_test_dietarie_habits.duplicated()


# In[171]:


dados_test_dietarie_habits


# In[172]:


base = pd.concat([base,dietarie_habits], axis=1)


# In[173]:


base_test = pd.concat([base_test,dados_test_dietarie_habits], axis=1)


# In[174]:


base_test


# In[175]:


#test = verifiquei que há valores diferentes na base de test da base de treinamento
test[test['Dietary Habits'] == 'Vivaan']


# In[176]:


test['Have you ever had suicidal thoughts ?']
#test[test['Have you ever had suicidal thoughts ?'] == 'Vivaan']


# In[177]:


dados_test_dietarie_habits


# In[178]:


#dados_test_dietarie_habits = dados_test_dietarie_habits.drop('Unhealthy', axis='columns')


# In[179]:


base.shape


# In[180]:


colunas = base.columns
colunas = colunas.drop('Depression')


# In[181]:


#dados_test[dados_test['Healthy'] == ]


# In[182]:


#base de teste final ajustada conforme a tabela de treino
base_test[colunas].shape


# In[183]:


base.columns


# In[184]:


#sepando dados alvo
X = base.drop('Depression', axis=1)
y = base['Depression']
#df_test = df_test.drop('Depression', axis=1)


# In[185]:


y = y.replace({"Yes":1,"No":0})


# In[186]:


from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[187]:


# Divisão do conjunto de dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[188]:


# Modelo CatBoost
cat_model = CatBoostClassifier(verbose=0)
cat_model.fit(X_train, y_train)
cat_preds = cat_model.predict(X_test)
cat_acc = accuracy_score(y_test, cat_preds)
cat_acc


# In[189]:


# Modelo LightGBM
lgb_model = LGBMClassifier()
lgb_model.fit(X_train, y_train)
lgb_preds = lgb_model.predict(X_test)
lgb_acc = accuracy_score(y_test, lgb_preds)
lgb_acc


# In[190]:


# Support Vector Machines

svc = SVC()
svc.fit(X_train, y_train)
SVC_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, y_train) * 100, 2)
acc_svc


# In[191]:


# Perceptron

perceptron = Perceptron()
perceptron.fit(X_train, y_train)
perceptron_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, y_train) * 100, 2)
acc_perceptron


# In[192]:


##
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, y_train) * 100, 2)

acc_knn


# In[193]:


##
#criterion='gini',n_estimators=1100,max_depth=5,min_samples_split=4,min_samples_leaf=5,max_features='auto',oob_score=True,random_state=42,n_jobs=-1,verbose=1

random_forest = RandomForestClassifier(criterion='log_loss', n_estimators=100, min_samples_split=4)
random_forest.fit(X_train, y_train)
random_forest_pred = random_forest.predict(X_test)

random_forest.score(X_train, y_train)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)

acc_random_forest


# In[194]:


y_train1 = y_train.replace({"Yes":1,"No":0})


# In[195]:


y_test = y_test.replace({"Yes":1,"No":0})


# In[196]:


X_test


# In[197]:


## 
xgbboost = XGBClassifier(learning_rate=1,max_depth=2)
xgbboost.fit(X_train, y_train1)
xgbboost_pred = xgbboost.predict(X_test)

xgbboost_classifier = round(xgbboost.score(X_train, y_train1) * 100, 2)

xgbboost_classifier


# In[198]:


gaussian = GaussianNB()
gaussian.fit(X_train, y_train1)
gaussian_pred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, y_train1) * 100, 2)

acc_gaussian


# In[199]:


GradientBoostingClassifierObj = GradientBoostingClassifier
GradientBoostingClassifier = GradientBoostingClassifierObj()
GradientBoostingClassifier.fit(X_train, y_train1)
GradientBoostingClassifier_pred = GradientBoostingClassifier.predict(X_test)

gradient_boosting_classifier = round(GradientBoostingClassifier.score(X_train, y_train1) * 100, 2)

gradient_boosting_classifier


# In[200]:


print(f"Acurácia CatBoost: {cat_acc}")
print(f"Acurácia LightGBM: {lgb_acc}")
print(f"Acurácia SVC: {acc_svc}")
print(f"Acurácia Perceptron: {acc_perceptron}")
print(f"Acurácia KNeighborsClassifier: {acc_knn}")
print(f"Acurácia acc_random_forest: {acc_random_forest}")
print(f"Acurácia xgbboost_classifier: {xgbboost_classifier}")
print(f"Acurácia acc_gaussian: {acc_gaussian}")
print(f"Acurácia gradient_boosting_classifier: {gradient_boosting_classifier}")


# In[201]:


## Utilizar esta métrica para medir a precisão do modelo final
## CatBoost
cat_precision_score = precision_score(y_test, cat_preds, average='macro')  
print(classification_report(y_test,cat_preds))


# In[202]:


## LightGBM
lgb_precision_score = precision_score(y_test, lgb_preds, average='macro') 
print(classification_report(y_test,lgb_preds))


# In[203]:


## SVC
SVC_precision_score = precision_score(y_test, SVC_pred, average='macro') 
print(classification_report(y_test,SVC_pred))


# In[204]:


## perceptron
perceptron_precision_score = precision_score(y_test, perceptron_pred, average='macro') 
print(classification_report(y_test,perceptron_pred))


# In[205]:


## KNeighborsClassifier
knn_precision_score = precision_score(y_test, knn_pred, average='macro')
print(classification_report(y_test,knn_pred))


# In[206]:


## random_forest_pred
random_forest_precision_score = precision_score(y_test, random_forest_pred, average='macro') 
print(classification_report(y_test,random_forest_pred))


# In[207]:


## xgbboost_pred
xgbboost_precision_score = precision_score(y_test, xgbboost_pred, average='macro') 
print(classification_report(y_test,xgbboost_pred))


# In[208]:


## acc_gaussian_pred
acc_gaussian_precision_score = precision_score(y_test, gaussian_pred, average='macro') 
print(classification_report(y_test,gaussian_pred))


# In[209]:


## gradient_boosting_classifier
gradient_boosting_classifier_precision_score = precision_score(y_test, GradientBoostingClassifier_pred, average='macro') 
print(classification_report(y_test,GradientBoostingClassifier_pred))


# In[210]:


print(f"Precision score CatBoost: {cat_precision_score}")
print(f"Precision score LightGBM: {lgb_precision_score}")
print(f"Precision score SVC: {SVC_precision_score}")
print(f"Precision score Perceptron: {perceptron_precision_score}")
print(f"Precision score KNeighborsClassifier: {knn_precision_score}")
print(f"Precision score acc_random_forest: {random_forest_precision_score}")
print(f"Precision score xgbboost_classifier: {xgbboost_precision_score}")
print(f"Precision score acc_gaussian_precision_score: {acc_gaussian_precision_score}")
print(f"Precision score gradient_boosting_classifier_precision_score: {gradient_boosting_classifier_precision_score}")


# In[211]:


f1s = cross_val_score(cat_model, X, y, cv=5, scoring="f1_macro")
print("F1-macros catboost:", f1s)
print("F1-macros catboost:", np.mean(f1s), "+-", np.std(f1s))


# In[212]:


f1s = cross_val_score(lgb_model, X, y, cv=5, scoring="f1_macro")
print("F1-macros lightboost:", f1s)
print("F1-macros lightboost:", np.mean(f1s), "+-", np.std(f1s))


# In[213]:


f1s = cross_val_score(svc, X, y, cv=5, scoring="f1_macro")
print("F1-macros svc:", f1s)
print("F1-macros svc:", np.mean(f1s), "+-", np.std(f1s))


# In[214]:


f1s = cross_val_score(perceptron, X, y, cv=5, scoring="f1_macro")
print("F1-macros perceptron:", f1s)
print("F1-macros perceptron:", np.mean(f1s), "+-", np.std(f1s))


# In[215]:


f1s = cross_val_score(knn, X, y, cv=5, scoring="f1_macro")
print("F1-macros knn:", f1s)
print("F1-macros knn:", np.mean(f1s), "+-", np.std(f1s))


# In[216]:


f1s = cross_val_score(random_forest, X, y, cv=5, scoring="f1_macro")
print("F1-macros random_forest:", f1s)
print("F1-macros random_forest:", np.mean(f1s), "+-", np.std(f1s))


# In[217]:


y = y.replace({"Yes": 1, "No": 0})


# In[218]:


f1s = cross_val_score(xgbboost, X, y, cv=5, scoring="f1_macro")
print("F1-macros:", f1s)
print("F1-macros:", np.mean(f1s), "+-", np.std(f1s))


# In[219]:


f1s = cross_val_score(gaussian, X, y, cv=5, scoring="f1_macro")
print("F1-macros gaussian:", f1s)
print("F1-macros gaussian:", np.mean(f1s), "+-", np.std(f1s))


# In[220]:


def objective2(trial):
    
    #from sklearn.ensemble import GradientBoostingClassifier
    
    params = {
    "n_estimators": trial.suggest_int("n_estimators", 100, 5000, step = 100),
    "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log = True),
    "max_depth": trial.suggest_int("max_depth", 3, 9),
    "subsample": trial.suggest_float("subsample", 0.5, 0.9, step = 0.1),
    #"max_features": trial.suggest_categorical("max_features", ["auto"]),
    "random_state": 42,
    }
    
    GradientBoostingClassifier = GradientBoostingClassifierObj(**params)
    GradientBoostingClassifier.fit(X_train, y_train1)
    GradientBoostingClassifier_pred = GradientBoostingClassifier.predict(X_test)
    gradient_boosting_classifier = round(GradientBoostingClassifier.score(X_train, y_train1) * 100, 2)

    return gradient_boosting_classifier


# In[221]:


X.columns


# In[222]:


X_test.columns


# In[223]:


#study = optuna.create_study(direction='maximize')
#study.optimize(objective2, n_trials=10)


# In[224]:


colunas = base.columns[base.columns != 'Depression']
colunas


# In[225]:


base_test[colunas]


# In[226]:


base_submission = base_test[X.columns]


# In[232]:


base_submission[perceptron.feature_names_in_].columns


# In[233]:


base_submission = base_submission.T.drop_duplicates().T


# In[280]:


base_submission[X.columns]


# In[262]:


#submission = perceptron.predict(dados_test[X_test.columns])
submission = cat_model.predict(base_submission[X.columns])


# In[263]:


submission.shape


# In[264]:


submission = np.where(submission == "Yes", 1, np.where(submission == "No", 0, submission))


# In[265]:


#pd.set_option('display.max_rows', None)


# In[281]:


submission.shape


# In[282]:


submission = submission.astype('int64')
submission


# In[268]:


base_submission[X_test.columns]


# In[283]:


base_submission.shape


# In[289]:


submission.shape


# In[284]:


dados_test.shape


# In[285]:


id = dados_test['id'].T.drop_duplicates().T


# In[286]:


id.shape


# In[293]:


id


# In[290]:


submission.shape


# In[294]:


dd = pd.DataFrame({
    "id":id['id'],
    "Depression":submission
})


# In[295]:


dd.shape


# In[296]:


dados_test.shape


# In[298]:


## Submit notebooks to the challenge

submission_final = pd.DataFrame({
        "id":id['id'],
        "Depression":submission
    })
submission_final.to_csv('submission.csv', index=False)

print(" Arquivo submission.csv pronto ")


# In[ ]:




