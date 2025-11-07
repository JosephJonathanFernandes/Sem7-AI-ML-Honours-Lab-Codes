from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Step 1: Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Step 2: Preprocess data
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Step 3: Build CNN model
cnn = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Step 4: Model summary
cnn.summary()

# Step 5: Compile model
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 6: Train model
cnn.fit(x_train, y_train, epochs=3, batch_size=128, validation_data=(x_test, y_test))

# Step 7: Evaluate on test data
test_loss, test_acc = cnn.evaluate(x_test, y_test)
print("\nTest Accuracy: {:.2f}%".format(test_acc * 100))
