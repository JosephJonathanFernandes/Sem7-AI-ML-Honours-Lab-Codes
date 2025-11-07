import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Dataset
text = "hello world"
chars = sorted(list(set(text)))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

X, y = [], []
for i in range(len(text) - 3):
    seq = text[i:i + 3]
    target = text[i + 3]
    X.append([char_to_idx[c] for c in seq])
    y.append(char_to_idx[target])

X = np.array(X)
y = np.array(y)

# One-hot encoding
X_onehot = np.eye(len(chars))[X]
y_onehot = np.eye(len(chars))[y]

# Model
model = Sequential([
    SimpleRNN(32, input_shape=(3, len(chars))),
    Dense(len(chars), activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_onehot, y_onehot, epochs=200, verbose=0)

# Prediction
pred = model.predict(X_onehot)
for i, p in enumerate(pred):
    print(text[i:i+3], "->", idx_to_char[np.argmax(p)])
