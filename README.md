## 機器學習導論HW2
### Handwritten Character Recognition 

----------------------
#### 1.Downloading Dataset

我使用的是 keras 中的 mnist.load_data()

<pre><code>
(X_train, y_train), (X_test, y_test) = mnist.load_data()
</code></pre>
----------------------
#### 2.Preprocessing Character Images
<pre><code>
# 大小標準化 [pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

# 將值標準化為[0,1]
X_train = X_train / 255
X_test = X_test / 255
</code></pre>
----------------------
### 3.Choosing a Classifier and Training It
Sequential() 作為模型，方法是使用 add() 將模型疊起

Conv2D() 是一個叫做Convolution2D的捲積層，Conv2D([pixels][width][height])
<pre><code>
 model = Sequential()
 # 2D卷積層
 model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
</code></pre>

<pre><code>
# MaxPooling2D 空間數據的最大池化操作
model.add(MaxPooling2D(pool_size=(2, 2)))
# Dropout 訓練期間每次更新時將輸入單位的一小部分隨機設置為20%，這有助於防止過度擬合
model.add(Dropout(0.2))
# 將2D矩陣數據轉換為稱為Flatten的矢量的圖層。
model.add(Flatten())
</code></pre>

compile(loss,optimizer,metrics)

loss: 損失函數

categorical_crossentropy: 多類分類問題

optimizer: 優化器，訓練方法

adam: 梯度下降

metrics: 網絡性能的指標,一般是用 'accuracy'

<pre><code>
# adam 梯度下降
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
</code></pre>
----------------------
### 4.Evaluating the Performance 

評估模型的損失值和度量值
<pre><code>
scores = model.evaluate(X_test, y_test, verbose=0)
</code></pre>

output:

![image](ML2018_Assignment2_410321156/output.PNG)

----------------------

