---

title: 'Deep Learning - tf.Keras를 이용한 간단한 신경망 작성 실습 ( Fashion Mnist data)'

categories: ['Data Science', 'Deep Learning']

tags: 
- 딥러닝

use_math: true

toc: true
toc_label: "My Table of Contents"
toc_icon: "cog"

---

### Keras Sequential 신경망 작성 순서:

신경망의 이론을 공부해보았고, keras library를 이용해 이를 코드로 구현해본다.
코드 작성의 순서는 아래와 같다.
0. seed 고정
1. keras library 호출
2. 신경망 Sequential model 구조정의
3. model compile로 학습 방향성 정의
4. model fit 진행

### keras Sequential model 작성

```python

# keras library 제공 데이터 fashion_mnist 불러오기
from tensorflow.keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# 정규화
x_train = x_train/255
x_test = x_test/255

# keras models 라이브러리 호출
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# keras model 구조 정의
model = Sequential()
model.add(Flatten( input_shape = (28,28)))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

# keras compile 정의
model.compile(optimizer= 'adam', loss = 'sparse_categorical_crossentropy', metrics= ['accuracy'])

# keras model 학습진행, epochs =10 > 데이터 전체를 10번 학습한다.
model.fit(x_train, y_train, epochs = 10, validation_data=(x_test, y_test))
```

```
[output]
Epoch 1/10
1875/1875 [==============================] - 4s 2ms/step - loss: 0.5340 - accuracy: 0.8096 - val_loss: 0.4585 - val_accuracy: 0.8336
Epoch 2/10
1875/1875 [==============================] - 3s 2ms/step - loss: 0.3948 - accuracy: 0.8567 - val_loss: 0.4242 - val_accuracy: 0.8464
Epoch 3/10
1875/1875 [==============================] - 3s 2ms/step - loss: 0.3641 - accuracy: 0.8665 - val_loss: 0.3852 - val_accuracy: 0.8591
Epoch 4/10
1875/1875 [==============================] - 3s 2ms/step - loss: 0.3410 - accuracy: 0.8738 - val_loss: 0.3759 - val_accuracy: 0.8642
Epoch 5/10
1875/1875 [==============================] - 3s 2ms/step - loss: 0.3233 - accuracy: 0.8809 - val_loss: 0.3814 - val_accuracy: 0.8662
Epoch 6/10
1875/1875 [==============================] - 3s 2ms/step - loss: 0.3115 - accuracy: 0.8845 - val_loss: 0.3695 - val_accuracy: 0.8694
Epoch 7/10
1875/1875 [==============================] - 3s 2ms/step - loss: 0.2970 - accuracy: 0.8905 - val_loss: 0.3591 - val_accuracy: 0.8717
Epoch 8/10
1875/1875 [==============================] - 3s 2ms/step - loss: 0.2891 - accuracy: 0.8924 - val_loss: 0.3532 - val_accuracy: 0.8741
Epoch 9/10
1875/1875 [==============================] - 3s 2ms/step - loss: 0.2787 - accuracy: 0.8960 - val_loss: 0.3766 - val_accuracy: 0.8613
Epoch 10/10
1875/1875 [==============================] - 3s 2ms/step - loss: 0.2740 - accuracy: 0.8980 - val_loss: 0.3491 - val_accuracy: 0.8772
<keras.callbacks.History at 0x7ff7a614a7d0>
```
신경망의 가장 일반적인 제작을 진행하는 방법이고, matplotlib.pyplot을 이용해 아래와 같이 이미지를 볼 수 있다. 

```python
import matplotlib.pyplot as plt
for i in range(9):
  plt.subplot(3,3,i+1)
  plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))
  plt.axis('off')
plt.show()
```

![](https://images.velog.io/images/dlskawns/post/d4e2013c-5048-45bc-b32a-c12fa00af2db/image.png)

해당 데이터는 60000개로, 모두 다른 이미지를 갖고 있는데, 너무 많으므로 9개만 추려본 case이다.
