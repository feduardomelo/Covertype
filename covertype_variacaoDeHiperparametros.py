import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# 1. Carregar a base Covertype, porém limitando o tamanho para reduzir tempo de execução
X, y = fetch_covtype(return_X_y=True)
# Limitando para 3000 amostras (ajuste conforme sua disponibilidade de recursos)
X = X[:5000]
y = y[:5000]

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Definir os modelos de classificação
dt = DecisionTreeClassifier(random_state=42)
svm = SVC(random_state=42)
knn = KNeighborsClassifier()
mlp = MLPClassifier(random_state=42)

# 3. Definir espaços de busca de hiperparâmetros
param_distributions_dt = {
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

param_distributions_svm = {
    #'gamma': [0.001, 0.01, 0.1],
    'kernel': ['poly','rbf','sigmoid'],
    'C': [0.1,1,2,4,5,10,15,20,30,50,100,200,500,1000]

}

param_distributions_knn = {
    'n_neighbors': [3, 5, 10, 20, 30, 40, 50],
    'weights': ['uniform', 'distance'],
    'metric':  ['euclidean', 'manhattan', 'minkowski'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}

param_distributions_mlp = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate_init': [0.001, 0.01, 0.1],
    'max_iter': [200, 300, 500]
}

# 4. Número de rodadas de busca
n_runs = 10  # você pode alterar conforme necessário

best_scores_dt = []
best_scores_svm = []
best_scores_knn = []
best_scores_mlp = []
best_params = {
    'Decision Tree': [],
    'SVM': [],
    'KNN': [],
    'MLP': []
}

for i in range(n_runs):
    print(f"Início da rodada numero: {i} ")
    # Decision Tree
    search_dt = RandomizedSearchCV(
        estimator=dt,
        param_distributions=param_distributions_dt,
        n_iter=5,
        cv=3,
        random_state=42+i,
        n_jobs=-1
    )
    search_dt.fit(X_train, y_train)
    y_pred_dt = search_dt.best_estimator_.predict(X_test)
    best_scores_dt.append(accuracy_score(y_test, y_pred_dt))
    best_params['Decision Tree'].append(search_dt.best_params_)

    # SVM
    search_svm = RandomizedSearchCV(
        estimator=svm,
        param_distributions=param_distributions_svm,
        n_iter=5,
        cv=3,
        random_state=42+i,
        n_jobs=-1
    )
    search_svm.fit(X_train, y_train)
    y_pred_svm = search_svm.best_estimator_.predict(X_test)
    best_scores_svm.append(accuracy_score(y_test, y_pred_svm))
    best_params['SVM'].append(search_svm.best_params_)

    # KNN
    search_knn = RandomizedSearchCV(
        estimator=knn,
        param_distributions=param_distributions_knn,
        n_iter=5,
        cv=3,
        random_state=42+i,
        n_jobs=-1
    )
    search_knn.fit(X_train, y_train)
    y_pred_knn = search_knn.best_estimator_.predict(X_test)
    best_scores_knn.append(accuracy_score(y_test, y_pred_knn))
    best_params['KNN'].append(search_knn.best_params_)

    # MLP
    search_mlp = RandomizedSearchCV(
        estimator=mlp,
        param_distributions=param_distributions_mlp,
        n_iter=5,
        cv=3,
        random_state=42+i,
        n_jobs=-1
    )
    search_mlp.fit(X_train, y_train)
    y_pred_mlp = search_mlp.best_estimator_.predict(X_test)
    best_scores_mlp.append(accuracy_score(y_test, y_pred_mlp))
    best_params['MLP'].append(search_mlp.best_params_)

    print(f"Fim da rodada numero: {i} ")

# 5. Apresentando melhores parâmetros de cada modelo
# print(best_params)

print("Melhores parâmetros por rodada: ")

print("Decision Tree")
for idx, params in enumerate(best_params['Decision Tree']):
    print(f'Rodada {idx}: {params}, Acurácia: {best_scores_dt[idx]} ')

print("SVM")
for idx, params in enumerate(best_params['SVM']):
    print(f'Rodada {idx}: {params}, Acurácia: {best_scores_svm[idx]} ')

print("KNN")
for idx, params in enumerate(best_params['KNN']):
    print(f'Rodada {idx}: {params}, Acurácia: {best_scores_knn[idx]} ')

print("MLP")
for idx, params in enumerate(best_params['MLP']):
    print(f'Rodada {idx}: {params}, Acurácia: {best_scores_mlp[idx]} ')

# 6. Plotar a evolução do score
plt.figure(figsize=(10,6))
plt.plot(best_scores_dt, marker='o', label='Decision Tree')
plt.plot(best_scores_svm, marker='o', label='SVM')
plt.plot(best_scores_knn, marker='o', label='KNN')
plt.plot(best_scores_mlp, marker='o', label='MLP')
plt.xlabel('Rodada da Busca de Hiperparâmetros')
plt.ylabel('Acurácia no Teste')
plt.title('Evolução do Melhor Score ao longo das Rodadas')
plt.legend()
plt.grid(True)
plt.show()