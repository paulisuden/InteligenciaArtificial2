import torch
from torch import nn
#manejan el dataset y los mini-lotes (batches):
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np

# ================================
# 1. Cargar los datos
# ================================
data = pd.read_csv("winequality-red.csv") 
X = data.drop("quality", axis=1).values
y = data["quality"].values

# Normalización y preprocesamiento
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Carga y separación de variables
X = data.drop("quality", axis=1).values
y = data["quality"].values

# Diagnostico:
data.isnull().sum()   # chequear nulos
data.dtypes           # tipos de columnas
data["quality"].value_counts()  # distribución de clases


# Re-etiquetar clases a 0..N-1 (clave para CrossEntropy)
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Normalizar: Escalado de features
scaler = StandardScaler()
X = scaler.fit_transform(X) # media 0, varianza 1

# Convertir a tensores de PyTorch
X = torch.tensor(X, dtype=torch.float32) #X como float32 (reales).
y = torch.tensor(y, dtype=torch.long) # y como long (enteros de clase). Es imprescindible para CrossEntropyLoss.

# Dataset y split en train/test
#TensorDataset empareja cada feature X[i] con su etiqueta y[i].
dataset = TensorDataset(X, y)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
#random_split divide aleatoriamente en 80% train y 20% test.
train_ds, test_ds = random_split(dataset, [train_size, test_size])

# DataLoaders (mini-batches)
#Entregan los datos por lotes
batch_size = 32
#shuffle=True en train para mezclar muestras en cada época (mejor generalización):
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=batch_size)

# ================================
# 2. Definir el modelo
# ================================
#MLP con 3 capas ocultas y ReLU
""" 
Tres capas ocultas con ReLU como activación.

La última capa no lleva Softmax: CrossEntropyLoss ya aplica log_softmax internamente y espera logits.

output_dim = cantidad de clases (p. ej., 6).

"""
class MLP(nn.Module):
    def __init__(self, input_dim, hidden1=128, hidden2=64, hidden3=32, output_dim=10):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, hidden3),
            nn.ReLU(),
            nn.Linear(hidden3, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

input_dim = X.shape[1]   # 11 características
# Output dimension correcto
output_dim = len(np.unique(y))
model = MLP(input_dim=X.shape[1], output_dim=output_dim)

# ================================
# 3. Definir función de pérdida y optimizador
# ================================
"""
CrossEntropyLoss compara logits vs. índices de clase (0..N-1).
Adam actualiza los pesos combinando momento y RMSProp; lr=0.001 suele ser buen inicio.
"""
criterion = nn.CrossEntropyLoss() #Clasificacion multiclase
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ================================
# 4. Entrenamiento
# ================================
"""
-- Ciclo por épocas y por batches.
-- Forward → Loss → Zero grad → Backward → Step es el patrón estándar.
-- Se imprime la pérdida promedio por época como señal de aprendizaje.
"""
def train_model(model, train_loader, criterion, optimizer, epochs=50):
    for epoch in range(epochs):
        total_loss = 0.0
        model.train()                       # modo entrenamiento (activa dropout/bn si hubiera)
        for xb, yb in train_loader:
            preds = model(xb)               # forward -> logits [batch, n_classes]
            loss = criterion(preds, yb)     # calcula pérdida del lote

            optimizer.zero_grad()           # limpia gradientes acumulados
            loss.backward()                 # backprop: d(loss)/d(params)
            optimizer.step()                # actualiza parámetros

            total_loss += loss.item()
        if (epoch+1)%3== 0:
          print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")


train_model(model, train_loader, criterion, optimizer, epochs=200)

# ================================
# 5. Evaluación
# ================================
"""
-- model.eval() y torch.no_grad() hacen la evaluación eficiente.
-- argmax convierte logits en predicciones de clase.
-- Se reporta accuracy simple.
"""
def evaluate(model, test_loader):
    model.eval()                    # modo evaluación (desactiva dropout/bn)
    correct, total = 0, 0
    with torch.no_grad():           # sin gradientes (más rápido y menos memoria)
        for xb, yb in test_loader:
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)  # clase con mayor logit
            correct += (preds == yb).sum().item()
            total   += yb.size(0)
    acc = correct / total
    print(f"Accuracy: {acc:.2%}")


evaluate(model, test_loader)
