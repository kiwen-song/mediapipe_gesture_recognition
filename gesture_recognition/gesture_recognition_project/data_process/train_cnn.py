import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# 读取数据
data = pd.read_csv('collected_data_cnn/data_cnn.csv')
X = data.iloc[:, :-1].values  # 特征（21个点 * 3维）
y = data.iloc[:, -1].values   # 标签

# one-hot 编码标签
y = tf.keras.utils.to_categorical(y - 1, num_classes=10)

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建模型
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(63,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')  # 10 个手势类别
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), batch_size=16)

# 保存模型
model.save('model_cnn/gesture_model_cnn.h5')
print("模型已保存为 gesture_model_cnn.h5")
