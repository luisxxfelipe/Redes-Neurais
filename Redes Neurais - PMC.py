import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def normalizar_valores(coluna):
    min_val = np.min(coluna)
    max_val = np.max(coluna)
    return (coluna - min_val) / (max_val - min_val)

def funcao_ativacao_sigmoide(x):
    return 1 / (1 + np.exp(-x))

def derivada_sigmoide(x):
    return x * (1 - x)

def erro_medio_quadratico(d, y):
    return np.sum((d - y) ** 2)

def calcular_variancia(erro):
    return np.var(erro)

# Passo forward
def passo_forward(X, W1, W2):
    z1 = np.dot(W1, X) 
    a1 = funcao_ativacao_sigmoide(z1)
    a1 = np.insert(a1, 0, -1)  # Adiciona o bias na saída da camada oculta
    z2 = np.dot(W2, a1) 
    a2 = funcao_ativacao_sigmoide(z2)
    return a1, a2

# Passo backward
def passo_backward(X, y, a1, a2, W1, W2, n):
    dz2 = a2 - y
    dW2 = np.dot(dz2[:, np.newaxis], a1[np.newaxis, :])  
    dz1 = np.dot(W2[:, 1:].T, dz2) * derivada_sigmoide(a1[1:])  
    dW1 = np.dot(dz1[:, np.newaxis], X[np.newaxis, :])
    W2 -= n * dW2
    W1 -= n * dW1
    return W1, W2

def aplicar_criterio_saida(saidas):
    saidas_pos = []
    for saida in saidas:
        saida_pos = []
        for valor in saida:
            if valor >= 0.5:
                saida_pos.append(1)
            else:
                saida_pos.append(0)
        saidas_pos.append(saida_pos)
    return np.array(saidas_pos)

# Treinamento
def pmc_treinamento(arquivo_treinamento, n, e, w1, w2):
    df = pd.read_excel(arquivo_treinamento)
    
    x1_norm = normalizar_valores(df['x1'])
    x2_norm = normalizar_valores(df['x2'])
    x3_norm = normalizar_valores(df['x3'])
    x4_norm = normalizar_valores(df['x4'])
    df_norm = pd.DataFrame({
        'x1': x1_norm,
        'x2': x2_norm,
        'x3': x3_norm,
        'x4': x4_norm,
        'd1': df['d1'],
        'd2': df['d2'],
        'd3': df['d3']
    })  
    
    X = np.vstack([df_norm['x1'], df_norm['x2'], df_norm['x3'], df_norm['x4']])
    d = np.vstack([df_norm['d1'], df_norm['d2'], df_norm['d3']])
    
    erro_anterior = float('inf')
    erros_por_epoca = []
    variancias_por_epoca = []
    epocas = 0
    
    while True:
        erros = []
        
        for amostra in range(len(X[0])):
            Xb = np.hstack([-1, X[:, amostra]])
            y1, y2_saida = passo_forward(Xb, w1, w2)
            erro = erro_medio_quadratico(d[:, amostra], y2_saida)
            erros.append(erro)
            w1, w2 = passo_backward(Xb, d[:, amostra], y1, y2_saida, w1, w2, n)
        
        erro_atual = np.mean(erros)
        variancia_atual = calcular_variancia(erros)
        erro = abs(erro_atual - erro_anterior)
        erro_anterior = erro_atual
        erros_por_epoca.append(erro)
        variancias_por_epoca.append(variancia_atual)
        if erro < e:
            break
        epocas += 1
    return w1, w2, epocas, erros_por_epoca, variancias_por_epoca, erros

# Validação
def pmc_validacao(pesos1, pesos2, arquivo_validacao):
    df_val = pd.read_excel(arquivo_validacao)
    
    x1_norm = normalizar_valores(df_val['x1'])
    x2_norm = normalizar_valores(df_val['x2'])
    x3_norm = normalizar_valores(df_val['x3'])
    x4_norm = normalizar_valores(df_val['x4'])
    df_val_norm = pd.DataFrame({
        'x1': x1_norm,
        'x2': x2_norm,
        'x3': x3_norm,
        'x4': x4_norm
    })
    
    X_val = np.vstack([df_val_norm['x1'], df_val_norm['x2'], df_val_norm['x3'], df_val_norm['x4']])
    
    saidas = []
    for amostra in range(len(X_val[0])):
        Xb = np.hstack([-1, X_val[:, amostra]])
        y1, y2_saida = passo_forward(Xb, pesos1, pesos2)
        saidas.append(y2_saida)
    
    saidas = np.array(saidas).T
    saidas_pos = aplicar_criterio_saida(saidas)
    
    valores_esperados = np.vstack([df_val['d1'], df_val['d2'], df_val['d3']])
    erro = erro_medio_quadratico(valores_esperados, saidas_pos)
    variancia_total = calcular_variancia(saidas_pos - valores_esperados)
    return saidas_pos, erro, variancia_total

# Definindo neurônios
n_camada_entrada = 4
n_camada_escondida = 15
n_camada_saida = 3

# Inicialização dos pesos
w1 = np.random.random([n_camada_escondida, n_camada_entrada + 1]) 
w2 = np.random.random([n_camada_saida, n_camada_escondida + 1]) 

# Parâmetros de treinamento
n = 0.1 # Taxa de aprendizado
e = 1e-6  # Erro desejado

# Chamando a função de treinamento
pesos1, pesos2, epocas, erros_por_epoca, variancias_por_epoca, teste = pmc_treinamento('PP04_dados_treinamento.xls', n, e, w1, w2)

print(f'Treinamento concluído em {epocas} épocas com erro médio de {erros_por_epoca[-1]} e variância de {variancias_por_epoca[-1]}')

# Chamando a função de validação
saidas, erro_validacao, variancia_validacao = pmc_validacao(pesos1, pesos2, 'PP04_dados_validacao.xls')

# Exibindo as saídas
for i, saida in enumerate(saidas.T):
    print(f'Amostra {i + 1}: y1 = {saida[0]}, y2 = {saida[1]}, y3 = {saida[2]}')

print(f'Erro na validação: {erro_validacao}')
print(f'Variância na validação: {variancia_validacao}')

plt.plot(range(len(erros_por_epoca)), erros_por_epoca)
plt.xlabel('Épocas')
plt.ylabel('Erro')
plt.title('Erro ao longo das épocas')
plt.show()
