## 機器學習導論HW2
### Handwritten Character Recognition 

----------------------
#### 1.Downloading Dataset
我使用的是 keras 中的 mnist.load_data()

#### 2.Preprocessing Character Images
<pre><code>
# 大小標準化 [pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

# 將值標準化為[0,1]
X_train = X_train / 255
X_test = X_test / 255
</code></pre>
