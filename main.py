import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# 1. Carregar os dados
df = pd.read_excel('USA.xlsx')

dt = df[4530:4800]

# df = df[4430:4780]
df = df.tail(500)

# Função para calcular os gols marcados e sofridos nas últimas n partidas
def calcular_gols_ultimos_jogos(df, equipe_casa, equipe_fora, idx_atual, n=50):
    # Filtra jogos anteriores ao jogo atual (antes de idx_atual)
    jogos_casa = df[(df['Home'] == equipe_casa) & (df.index < idx_atual)][['HG', 'AG']].tail(n)
    jogos_fora = df[(df['Away'] == equipe_fora) & (df.index < idx_atual)][['AG', 'HG']].tail(n)

    # Gols do time da casa
    gols_marcados_casa = jogos_casa['HG'].sum()
    gols_sofridos_casa = jogos_casa['AG'].sum()

    # Gols do time visitante
    gols_marcados_fora = jogos_fora['AG'].sum()
    gols_sofridos_fora = jogos_fora['HG'].sum()

    return (gols_marcados_casa, gols_sofridos_casa, gols_marcados_fora, gols_sofridos_fora)

# Adicionando novas colunas para armazenar as somas de gols
df['gols_marcados_casa'] = 0
df['gols_sofridos_casa'] = 0
df['gols_marcados_fora'] = 0
df['gols_sofridos_fora'] = 0

dt['gols_marcados_casa'] = 0
dt['gols_sofridos_casa'] = 0
dt['gols_marcados_fora'] = 0
dt['gols_sofridos_fora'] = 0

# Iterando sobre cada linha (jogo) no DataFrame
for index, row in df.iterrows():
    # Time da casa e visitante
    time_casa = row['Home']
    time_fora = row['Away']

    # Calcular os gols marcados e sofridos para o time da casa e visitante
    gols_marcados_casa, gols_sofridos_casa, gols_marcados_fora, gols_sofridos_fora = calcular_gols_ultimos_jogos(df, time_casa, time_fora, index)

    # Atualizando o DataFrame com os valores calculados
    df.at[index, 'gols_marcados_casa'] = gols_marcados_casa
    df.at[index, 'gols_sofridos_casa'] = gols_sofridos_casa
    df.at[index, 'gols_marcados_fora'] = gols_marcados_fora
    df.at[index, 'gols_sofridos_fora'] = gols_sofridos_fora

# Iterando sobre cada linha (jogo) no DataFrame
for index, row in dt.iterrows():
    # Time da casa e visitante
    time_casa = row['Home']
    time_fora = row['Away']

    # Calcular os gols marcados e sofridos para o time da casa e visitante
    gols_marcados_casa, gols_sofridos_casa, gols_marcados_fora, gols_sofridos_fora = calcular_gols_ultimos_jogos(df, time_casa, time_fora, index)

    # Atualizando o DataFrame com os valores calculados
    dt.at[index, 'gols_marcados_casa'] = gols_marcados_casa
    dt.at[index, 'gols_sofridos_casa'] = gols_sofridos_casa
    dt.at[index, 'gols_marcados_fora'] = gols_marcados_fora
    dt.at[index, 'gols_sofridos_fora'] = gols_sofridos_fora

# dt = dt.tail(20)
# Exibindo o DataFrame atualizado
# print(df.info())

# 2. Preparar os dados
# Assumindo que temos colunas: 'time_casa', 'time_visitante', 'gols_casa', 'gols_visitante', 'resultado'

# Criar feature de resultado (1 para vitória do time da casa, 0 para empate ou derrota)
# df['vitoria_time'] = (df['HG'] > df['AG']).astype(int)
lb = LabelEncoder()
df['vitoria_time'] = lb.fit_transform(df['Res'])

# Codificar times como valores numéricos
le = LabelEncoder()
df['time_casa_encoded'] = le.fit_transform(df['Home'])
df['time_visitante_encoded'] = le.fit_transform(df['Away'])

# 3. Selecionar features e target
X = df[['time_casa_encoded', 'time_visitante_encoded', 'AvgCH', 'AvgCD', 'AvgCA', 'gols_marcados_casa', 'gols_sofridos_casa', 'gols_marcados_fora', 'gols_sofridos_fora']]
y = df['vitoria_time']

# Normalizar as features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Balanceamento das classes com SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# 4. Dividir dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# 5. Treinar o modelo
# model = RandomForestClassifier(max_features=None, max_depth=20, min_samples_split=2, n_estimators=100, min_samples_leaf=1, bootstrap=True, criterion='gini', random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 6. Fazer previsões e avaliar o modelo
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Acurácia do modelo: {accuracy*100:.2f}%")
print("Relatório de classificação:\n", classification_report(y_test, predictions))

# # 7. Usar o modelo para prever próximos jogos
# # Exemplo: prever resultado para time_casa_id=1 vs time_visitante_id=2
# novo_jogo = [[1, 2]]
# previsao = model.predict(novo_jogo)
# print(f"Previsão para o novo jogo: {'Vitória do time da casa' if previsao[0] == 1 else 'Empate ou vitória do time visitante'}")

# 1. Criar um dicionário de mapeamento
time_mapping = dict(zip(le.transform(le.classes_), le.classes_))
res_mapping = dict(zip(lb.transform(lb.classes_), lb.classes_))

# 2. Função para decodificar
def decodificar_time(codigo):
    return time_mapping.get(codigo, "Time desconhecido")
def decodificar_res(codigo):
    return res_mapping.get(codigo, "Res desconhecido")

# 3. Aplicar a decodificação
df['time_casa_decodificado'] = df['time_casa_encoded'].map(decodificar_time)
df['time_visitante_decodificado'] = df['time_visitante_encoded'].map(decodificar_time)

# Realizar Cross-Validation para ajustar o modelo
cv_scores = cross_val_score(model, X_resampled, y_resampled, cv=5)
print("Acurácia média com Cross-Validation:", cv_scores.mean())

# Tuning de hiperparâmetros usando GridSearchCV
# param_grid = {
#     'n_estimators': [50, 100, 200, 300, 500],         # Número de árvores na floresta
#     'max_depth': [5, 10, 20, 30, None],               # Profundidade máxima de cada árvore (None significa sem limite)
#     'min_samples_split': [2, 5, 10, 15],              # Número mínimo de amostras necessárias para dividir um nó
#     'min_samples_leaf': [1, 2, 4, 10],                # Número mínimo de amostras necessárias em uma folha
#     'max_features': ['auto', 'sqrt', 'log2', None],   # Número de features a serem consideradas para encontrar a melhor divisão
#     'bootstrap': [True, False],                       # Se deve ou não usar o bootstrap para amostragem das árvores
#     'criterion': ['gini', 'entropy'],                 # Critério de divisão usado nas árvores
#     'class_weight': [None, 'balanced', 'balanced_subsample']  # Ajuste de pesos para classes desbalanceadas
# }
# grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
# grid_search.fit(X_train, y_train)

# print("Melhores parâmetros:", grid_search.best_params_)
# print("Melhor acurácia após GridSearch:", grid_search.best_score_)

# dfe = df.tail(20)

# print(f"{dfe['time_casa_decodificado']} : {dfe['time_casa_encoded']}")

"""
  Atletico-MG = 2
  Bahia = 3
  Criciuma = 7
  Fortaleza = 12
  Sao Paulo = 17
  Internacional = 14
  Flamengo RJ = 10
  Vasco = 18
  Criciuma = 7
  Cruzeiro = 8
  Cuiaba = 9
  Botafogo RJ = 4
  Cruzeiro = 8
  Gremio = 13
  Corinthians = 6
  Athletico-PR = 0
  Bragantino = 5
  Fluminense = 11
  Juventude = 15
  Vitoria = 19
"""

# print(dt)

# 4. Exemplo de uso na previsão
novo_jogo = [[16, 3, 2.17, 3.01, 3.80, 13, 12, 17, 18]]
previsao = model.predict(novo_jogo)
time_casa = decodificar_time(novo_jogo[0][0])
time_visitante = decodificar_time(novo_jogo[0][1])
print(decodificar_res(previsao[0]))
# print(f"Previsão para o jogo {time_casa} vs {time_visitante}: {'Vitória do time da casa' if previsao[0] == 1 else 'Vitória do time visitante'}")
if previsao[0] == 2:
  print(f"Previsão para o jogo {time_casa} vs {time_visitante}: Vitória do time da casa")
elif previsao[0] == 0:
  print(f"Previsão para o jogo {time_casa} vs {time_visitante}: Vitória do time de fora")
else:
  print(f"Previsão para o jogo {time_casa} vs {time_visitante}: Empate")

# 5. Visualizar algumas linhas do DataFrame para confirmar
# print(df[['Home', 'time_casa_encoded', 'time_casa_decodificado',
#           'Away', 'time_visitante_encoded', 'time_visitante_decodificado']].iloc[-50:])