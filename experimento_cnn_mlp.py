import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Dados
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1,28,28,1).astype('float32') / 255
x_test  = x_test.reshape(-1,28,28,1).astype('float32') / 255
y_train_cat = to_categorical(y_train, 10)
y_test_cat  = to_categorical(y_test, 10)

# CNN
cnn = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history_cnn = cnn.fit(x_train, y_train_cat, batch_size=128, epochs=30, validation_split=0.2, verbose=1)
_, acc_cnn = cnn.evaluate(x_test, y_test_cat, verbose=0)
print(f'Acurácia CNN: {acc_cnn:.4f}')

# MLP
mlp = Sequential([
    Flatten(input_shape=(28,28,1)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
mlp.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history_mlp = mlp.fit(x_train, y_train_cat, batch_size=128, epochs=30, validation_split=0.2, verbose=1)
_, acc_mlp = mlp.evaluate(x_test, y_test_cat, verbose=0)
print(f'Acurácia MLP: {acc_mlp:.4f}')

# Matrizes de confusão
def plot_cm(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.savefig(f'{title}.png', dpi=300)
    plt.show()

y_pred_cnn = np.argmax(cnn.predict(x_test), axis=1)
y_pred_mlp = np.argmax(mlp.predict(x_test), axis=1)
plot_cm(y_test, y_pred_cnn, 'Matriz_Confusao_CNN')
plot_cm(y_test, y_pred_mlp, 'Matriz_Confusao_MLP')

# Curvas de aprendizado
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history_cnn.history['loss'], label='CNN treino')
plt.plot(history_cnn.history['val_loss'], label='CNN val')
plt.plot(history_mlp.history['loss'], label='MLP treino')
plt.plot(history_mlp.history['val_loss'], label='MLP val')
plt.title('Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history_cnn.history['accuracy'], label='CNN treino')
plt.plot(history_cnn.history['val_accuracy'], label='CNN val')
plt.plot(history_mlp.history['accuracy'], label='MLP treino')
plt.plot(history_mlp.history['val_accuracy'], label='MLP val')
plt.title('Acurácia')
plt.legend()
plt.tight_layout()
plt.savefig('comparacao_curvas.png', dpi=300)
plt.show()