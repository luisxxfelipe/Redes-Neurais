import numpy as np
import matplotlib.pyplot as plt

# Função para ler os dados de um arquivo .txt
def load_data(filename):
    data = np.genfromtxt(filename, delimiter=None, skip_header=1)
    X = data[:, :-1]
    y = data[:, -1]
    return X, y

# Função de ativação Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada da função de ativação Sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Inicialização de pesos
def initialize_weights(input_size, hidden_size, output_size):
    W1 = np.random.randn(input_size, hidden_size)
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size)
    b2 = np.zeros((1, output_size))
    return W1, b1, W2, b2

# Propagação para frente
def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    return Z1, A1, Z2, A2

# Cálculo do erro quadrático médio
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Treinamento da rede neural
def train_model(X, y, hidden_size, epochs, learning_rate):
    input_size = X.shape[1]
    output_size = 1
    W1, b1, W2, b2 = initialize_weights(input_size, hidden_size, output_size)
    losses = []

    for epoch in range(epochs):
        # Propagação para frente
        Z1, A1, Z2, A2 = forward_propagation(X, W1, b1, W2, b2)
        
        # Cálculo do erro
        loss = mean_squared_error(y, A2)
        losses.append(loss)
        
        # Propagação para trás
        dZ2 = A2 - y.reshape(-1, 1)
        dW2 = np.dot(A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        
        dA1 = np.dot(dZ2, W2.T)
        dZ1 = dA1 * sigmoid_derivative(A1)
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        
        # Atualização dos pesos
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        
    return W1, b1, W2, b2, losses

# Função para fazer previsões
def predict(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_propagation(X, W1, b1, W2, b2)
    return A2

# Função para avaliar o modelo
def evaluate_model(X, y, W1, b1, W2, b2):
    y_pred = predict(X, W1, b1, W2, b2)
    error_rel = np.mean(np.abs((y - y_pred) / y))
    variance = np.var(y_pred - y)
    return error_rel, variance

# Carregar dados de treinamento e validação
x_train, y_train = load_data('C:\\Users\\Luis\\OneDrive\\Documentos\\Codigo redes neurais\\PP03_dados-treinamento.txt')
x_test, y_test = load_data('C:\\Users\\Luis\\OneDrive\\Documentos\\Codigo redes neurais\\PP03_dados-validacao.txt')

# Normalização dos dados
x_train = x_train / np.max(x_train, axis=0)
y_train = y_train / np.max(y_train)
x_test = x_test / np.max(x_test, axis=0)
y_test = y_test / np.max(y_test)

# Treinamento do modelo
hidden_size = 10
epochs = 1000
learning_rate = 0.1
histories = []
weights = []

for i in range(5):
    W1, b1, W2, b2, history = train_model(x_train, y_train, hidden_size, epochs, learning_rate)
    histories.append(history)
    weights.append((W1, b1, W2, b2))

# Plotar os gráficos de erro quadrático médio
for i, history in enumerate(histories):
    plt.plot(history, label=f'Training {i+1}')
plt.xlabel('Épocas')
plt.ylabel('Erro Quadrático Médio')
plt.legend()
plt.show()

# Avaliação do modelo
errors = []
variances = []

for i, (W1, b1, W2, b2) in enumerate(weights):
    error_rel, var = evaluate_model(x_test, y_test, W1, b1, W2, b2)
    errors.append(error_rel)
    variances.append(var)

# Exibir resultados
for i in range(5):
    print(f'Training {i+1} - Erro Relativo Médio: {errors[i]:.4f}, Variância: {variances[i]:.4f}')

# Melhor configuração
best_index = np.argmin(errors)
print(f'Melhor configuração: Training {best_index+1}')
