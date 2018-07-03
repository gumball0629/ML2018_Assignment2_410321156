import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
# 使用抽象的Keras後端來編寫代碼
from keras import backend as K
# 'th' 設置圖像的維度順序
K.set_image_dim_ordering('th')

# 將隨機數生成器初始化為常數
seed = 7
numpy.random.seed(seed)

# 讀檔
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 大小標準化 [pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

# 將值標準化為[0,1]
X_train = X_train / 255
X_test = X_test / 255

# 將y轉換成one-hot encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

def baseline_model():
    # 模型 Sequential() add(self, layer) 將模型疊起
    
    model = Sequential()
    # 2D卷積層
    model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    # MaxPooling2D 空間數據的最大池化操作
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Dropout 訓練期間每次更新時將輸入單位的一小部分隨機設置為20%，這有助於防止過度擬合
    model.add(Dropout(0.2))
    # 將2D矩陣數據轉換為稱為Flatten的矢量的圖層。
    model.add(Flatten())
    # Dense() 連接層
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # compile() For a multi-class classification problem
    # adam 梯度下降
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = baseline_model()
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
# 評估模型的損失值和度量值
scores = model.evaluate(X_test, y_test, verbose=0)
print("Error: %.2f%%" % (100-scores[1]*100))