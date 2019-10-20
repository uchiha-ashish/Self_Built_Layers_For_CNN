import keras
num_classes=185
x_train = []
y_train = []
IMG_SIZE=224

for features,label in training_data:
    x_train.append(features)
    y_train.append(label)
print(np.shape(x_train[0].reshape(-1, IMG_SIZE, IMG_SIZE, 3)))

x_train = np.array(x_train).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
x_train=np.asarray(x_train)
y_train = keras.utils.to_categorical(y_train, num_classes)


x_test = []
y_test = []
IMG_SIZE=224

for features,label in test_data:
    x_test.append(features)
    y_test.append(label)
print(np.shape(x_test[0].reshape(-1, IMG_SIZE, IMG_SIZE, 3)))

x_test = np.array(x_test).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
x_test=np.asarray(x_test)
y_test = keras.utils.to_categorical(y_test, num_classes)
