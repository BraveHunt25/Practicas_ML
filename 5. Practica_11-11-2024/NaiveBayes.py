import numpy as np
import os
import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, LeaveOneOut

import numpy as np

class NaiveBayesClassifier:
    def __init__(self):
        self.class_priors = {}       # Probabilidades a priori de cada clase
        self.feature_params = {}     # Parámetros de características: media y desviación estándar (para numéricos) o probabilidades (para categóricos)
        self.classes = None          # Lista de clases

    def fit(self, X, y):
        # Calcular las probabilidades a priori (priors) de cada clase
        self.classes, class_counts = np.unique(y, return_counts=True)
        total_samples = len(y)
        
        for cls, count in zip(self.classes, class_counts):
            self.class_priors[cls] = count / total_samples  # P(C)
            
            # Filtrar los datos que corresponden a la clase actual
            cls_data = X[y == cls]
            self.feature_params[cls] = {}
            
            for col in X.columns:
                # Si el atributo es numérico, calculamos media y desviación estándar
                if np.issubdtype(X[col].dtype, np.number):
                    mean = cls_data[col].mean()
                    std = cls_data[col].std() + 1e-6  # Añadir una constante para evitar std=0
                    self.feature_params[cls][col] = {'mean': mean, 'std': std}
                else:
                    # Si el atributo es categórico, calculamos probabilidades
                    values, counts = np.unique(cls_data[col], return_counts=True)
                    total_counts = np.sum(counts)
                    self.feature_params[cls][col] = {val: (count + 1) / (total_counts + len(values)) for val, count in zip(values, counts)}

    def _calculate_probability(self, x, cls):
        log_prob = np.log(self.class_priors[cls])  # Probabilidad a priori de la clase
        
        for col, value in x.items():
            if col in self.feature_params[cls]:  # Verificar que la columna tiene parámetros calculados
                if isinstance(self.feature_params[cls][col], dict) and 'mean' in self.feature_params[cls][col]:
                    # Atributo numérico: aplicamos distribución normal
                    mean = self.feature_params[cls][col]['mean']
                    std = self.feature_params[cls][col]['std']
                    # Usamos la fórmula de la distribución normal
                    prob = (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-((value - mean) ** 2 / (2 * std ** 2)))
                    log_prob += np.log(prob + 1e-6)  # Sumar el logaritmo de la probabilidad
                else:
                    # Atributo categórico
                    prob = self.feature_params[cls][col].get(value, 1e-6)  # Laplace para valor no visto
                    log_prob += np.log(prob)
        return log_prob

    def predict(self, X):
        predictions = []
        for _, row in X.iterrows():
            class_scores = {cls: self._calculate_probability(row, cls) for cls in self.classes}
            predictions.append(max(class_scores, key=class_scores.get))  # Escoger la clase con mayor probabilidad
        return predictions        

if __name__ == '__main__':
    iris: DataFrame = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets/iris.csv"))
    wine: DataFrame = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets/wine.data.csv"))
    mushroom: DataFrame = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets/agaricus-lepiota.data.csv"))
    
    '''Con método estratificado'''
    # Para iris
    X: DataFrame = iris.drop(columns='class')
    Y: DataFrame = iris['class']
    accuracies: list = []
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=42)
    clasificador: NaiveBayesClassifier = NaiveBayesClassifier()
    clasificador.fit(X_train, Y_train)
    predicciones = clasificador.predict(X_test)
    accuracy = np.mean(predicciones == Y_test)
    accuracies.append(accuracy)
    print("Iris - Estratificado")
    conf_matrix = confusion_matrix(Y_test, predicciones)
    print("Matriz de Confusión:\n", conf_matrix)
    print(f"Precisión: {accuracy}\n")
    print(f"Precisión media (método Estratificado): {np.mean(accuracies)}\n")

    print('-'*100)
    # Para wine
    X = wine.drop(columns='Class')
    Y = wine['Class']
    accuracies: list = []
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=42)
    clasificador: NaiveBayesClassifier = NaiveBayesClassifier()
    clasificador.fit(X_train, Y_train)
    predicciones = clasificador.predict(X_test)
    accuracy = np.mean(predicciones == Y_test)
    accuracies.append(accuracy)
    print("Wine - Estratificado")
    conf_matrix = confusion_matrix(Y_test, predicciones)
    print("Matriz de Confusión:\n", conf_matrix)
    print(f"Precisión: {accuracy}\n")
    print(f"Precisión media (método Estratificado): {np.mean(accuracies)}\n")

    print('-'*100)
    # Para mushroom
    X = mushroom.drop(columns='class')
    Y = mushroom['class']
    accuracies: list = []
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=42)
    clasificador: NaiveBayesClassifier = NaiveBayesClassifier()
    clasificador.fit(X_train, Y_train)
    predicciones = clasificador.predict(X_test)
    accuracy = np.mean(predicciones == Y_test)
    accuracies.append(accuracy)
    print("Mushroom - Estratificado")
    conf_matrix = confusion_matrix(Y_test, predicciones)
    print("Matriz de Confusión:\n", conf_matrix)
    print(f"Precisión: {accuracy}\n")
    print(f"Precisión media (método Estratificado): {np.mean(accuracies)}\n")

    '''Con método de 10-fold cross-validation'''
    cross_validation: StratifiedKFold = StratifiedKFold(n_splits=10)

    print(f"10-fold cross-validation".center(20, '*'))
    # Para iris
    X: DataFrame = iris.drop(columns='class')
    Y: DataFrame = iris['class']
    accuracies = []
    matrices_confusion = []
    for train_index, test_index in cross_validation.split(X, Y):
        # Dividir los datos en conjuntos de entrenamiento y prueba
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

        # Crear y ajustar el clasificador
        clasificador = NaiveBayesClassifier()
        clasificador.fit(X_train, Y_train)

        # Hacer predicciones
        predicciones = clasificador.predict(X_test)

        # Calcular precisión (puedes usar otras métricas si deseas)
        accuracy = np.mean(predicciones == Y_test)
        accuracies.append(accuracy)

        conf_matrix = confusion_matrix(Y_test, predicciones)
        matrices_confusion.append(conf_matrix)

    print(f"Iris - 10-fold Cross-Validation\nPrecisión media: {np.mean(accuracies)}")
    print(f"Desviación estándar de la precisión: {np.std(accuracies)}")
    print("Matrices de Confusión por Fold:")
    for idx, matrix in enumerate(matrices_confusion):
        print(f"Fold {idx + 1}:\n{matrix}\n")

    # Para wine
    X = wine.drop(columns='Class')
    Y = wine['Class']
    accuracies = []
    matrices_confusion = []
    for train_index, test_index in cross_validation.split(X, Y):
        # Dividir los datos en conjuntos de entrenamiento y prueba
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

        # Crear y ajustar el clasificador
        clasificador = NaiveBayesClassifier()
        clasificador.fit(X_train, Y_train)

        # Hacer predicciones
        predicciones = clasificador.predict(X_test)

        # Calcular precisión (puedes usar otras métricas si deseas)
        accuracy = np.mean(predicciones == Y_test)
        accuracies.append(accuracy)

        conf_matrix = confusion_matrix(Y_test, predicciones)
        matrices_confusion.append(conf_matrix)

    print(f"Wine - 10-fold Cross-Validation\nPrecisión media: {np.mean(accuracies)}")
    print(f"Desviación estándar de la precisión: {np.std(accuracies)}")
    print("Matrices de Confusión por Fold:")
    for idx, matrix in enumerate(matrices_confusion):
        print(f"Fold {idx + 1}:\n{matrix}\n")

    # Para mushroom
    X = mushroom.drop(columns='class')
    Y = mushroom['class']
    accuracies = []
    matrices_confusion = []
    for train_index, test_index in cross_validation.split(X, Y):
        # Dividir los datos en conjuntos de entrenamiento y prueba
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

        # Crear y ajustar el clasificador
        clasificador = NaiveBayesClassifier()
        clasificador.fit(X_train, Y_train)

        # Hacer predicciones
        predicciones = clasificador.predict(X_test)

        # Calcular precisión (puedes usar otras métricas si deseas)
        accuracy = np.mean(predicciones == Y_test)
        accuracies.append(accuracy)

        conf_matrix = confusion_matrix(Y_test, predicciones)
        matrices_confusion.append(conf_matrix)

    print(f"Mushroom - 10-fold Cross-Validation\nPrecisión media: {np.mean(accuracies)}")
    print(f"Desviación estándar de la precisión: {np.std(accuracies)}")
    print("Matrices de Confusión por Fold:")
    for idx, matrix in enumerate(matrices_confusion):
        print(f"Fold {idx + 1}:\n{matrix}\n")

    '''Con método Leave-One-Out'''
    leave_one_out: LeaveOneOut = LeaveOneOut()
    
    print(f"Leave One Out".center(20, '*'))
    # Para iris
    X: DataFrame = iris.drop(columns='class')
    Y: DataFrame = iris['class']
    accuracies: list = []
    global_conf_matrix = np.zeros((len(np.unique(Y)), len(np.unique(Y))))  # Matriz de confusión global acumulada
    for train_index, test_index in leave_one_out.split(X, Y):
        # Dividir los datos en conjuntos de entrenamiento y prueba
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

        # Crear y ajustar el clasificador
        clasificador = NaiveBayesClassifier()
        clasificador.fit(X_train, Y_train)

        # Hacer predicción
        predicciones = clasificador.predict(X_test)

        accuracy = np.mean(predicciones == Y_test)
        accuracies.append(accuracy)

        conf_matrix = confusion_matrix([Y_test.iloc[0]], [predicciones[0]], labels=np.unique(Y))
        global_conf_matrix += conf_matrix

    print(f"Iris - Leave-One-Out\nPrecisión media: {np.mean(accuracies)}")
    print(f"Desviación estándar de la precisión: {np.std(accuracies)}")
    print("Matriz de Confusión Global Acumulada:")
    print(global_conf_matrix)
   
    # Para wine
    X = wine.drop(columns='Class')
    Y = wine['Class']
    accuracies = []
    global_conf_matrix = np.zeros((len(np.unique(Y)), len(np.unique(Y))))  # Matriz de confusión global acumulada
    for train_index, test_index in leave_one_out.split(X, Y):
        # Dividir los datos en conjuntos de entrenamiento y prueba
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

        # Crear y ajustar el clasificador
        clasificador = NaiveBayesClassifier()
        clasificador.fit(X_train, Y_train)

        # Hacer predicción
        predicciones = clasificador.predict(X_test)

        accuracy = np.mean(predicciones == Y_test)
        accuracies.append(accuracy)

        conf_matrix = confusion_matrix([Y_test.iloc[0]], [predicciones[0]], labels=np.unique(Y))
        global_conf_matrix += conf_matrix

    print(f"Wine - Leave-One-Out\nPrecisión media: {np.mean(accuracies)}")
    print(f"Desviación estándar de la precisión: {np.std(accuracies)}")
    print("Matriz de Confusión Global Acumulada:")
    print(global_conf_matrix)

    # Para mushroom
    X = mushroom.drop(columns='class')
    Y = mushroom['class']
    accuracies = []
    global_conf_matrix = np.zeros((len(np.unique(Y)), len(np.unique(Y))))  # Matriz de confusión global acumulada
    for train_index, test_index in leave_one_out.split(X, Y):
        # Dividir los datos en conjuntos de entrenamiento y prueba
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

        # Crear y ajustar el clasificador
        clasificador = NaiveBayesClassifier()
        clasificador.fit(X_train, Y_train)

        # Hacer predicción
        predicciones = clasificador.predict(X_test)

        accuracy = np.mean(predicciones == Y_test)
        accuracies.append(accuracy)

        conf_matrix = confusion_matrix([Y_test.iloc[0]], [predicciones[0]], labels=np.unique(Y))
        global_conf_matrix += conf_matrix

    print(f"Mushroom - Leave-One-Out\nPrecisión media: {np.mean(accuracies)}")
    print(f"Desviación estándar de la precisión: {np.std(accuracies)}")
    print("Matriz de Confusión Global Acumulada:")
    print(global_conf_matrix)