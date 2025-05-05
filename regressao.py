import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Carregar os dados
data = pd.read_csv('4.heart.csv')

# Codificar variáveis categóricas
le = LabelEncoder()
data['sex'] = le.fit_transform(data['sex'])
data['painType'] = le.fit_transform(data['painType'])
data['restingECG'] = le.fit_transform(data['restingECG'])
data['STslope'] = le.fit_transform(data['STslope'])
data['defectType'] = le.fit_transform(data['defectType'])

# Separar variáveis independentes (X) e dependentes (y)
X = data.drop('heartDisease', axis=1)
y = data['heartDisease']

# Dividir em dados de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar as variáveis numéricas
scaler = StandardScaler()
X_train[['age', 'restingBP', 'cholesterol', 'restingHR', 'STdepression', 'coloredVessels']] = scaler.fit_transform(X_train[['age', 'restingBP', 'cholesterol', 'restingHR', 'STdepression', 'coloredVessels']])
X_test[['age', 'restingBP', 'cholesterol', 'restingHR', 'STdepression', 'coloredVessels']] = scaler.transform(X_test[['age', 'restingBP', 'cholesterol', 'restingHR', 'STdepression', 'coloredVessels']])

# Treinar o modelo (usando Random Forest como exemplo)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)

# Avaliar o modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo: {accuracy:.2f}')
