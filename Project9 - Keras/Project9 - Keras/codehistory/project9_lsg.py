# project9 이순규

# 에이스 조의 묻혀가고픈 이순규
# Boston Housing Price 데이터 설명

# 총 506개의 행과 12개의 열, 마지막 열인 MEDV(집값)이 종속변수로 설정.
# 데이터명 : Boston Housing Price (보스턴 주택 가격 데이터).
# 데이터설명 : 보스턴 시의 주택 가격에 대한 데이터
# 레코드수 : 506 개
# 필드개수 :  14 개
# 보스턴 주택 데이터는 여러 개의 측정지표들 (예를 들어, 범죄율, 학생/교사 비율 등)을 포함한,
# 보스턴 인근의 주택 가격의 중앙값(median value)이다. 이 데이터 집합은 12개의 변수를 포함하고 있다.

# 각 변수 설명

# CRIM - 자치시(town)별 1인당 범죄율.
# ZN - 25,000 평방피트를 초과하는 거주지역의 비율.
# INDUS - 비소매상업지역이 점유하고 있는 토지의 비율
# CHAS - 찰스 강 더미 변수 (= 1 강 경계 경우; 0 그렇지 않으면)
# NOX10ppm - 당 농축 일산화질소
# RM - 주택 1가구당 평균 방의 개수
# AGE - 1940년 이전에 건축된 소유주택의 비율
# DIS - 5개의 보스턴 직업센터까지의 접근성 지수
# TAX - 10,000달러당 재산세율
# PTRATIO - 자치시(town)별 학생/교사 비율
# B - (Bk-0.62)^2, 여기서 Bk는 자치시별 흑인의 비율
# LSTAT - 모집단의 하위계층의 비율(%)
# MEDV - 본인 소유의 주택가격(중앙값) (단위: $1,000)

# 분석을 위한 패키지 및 데이터 불러오기

################################################################################

# 1. keras package 내 Boston Housing Price 데이터셋과 함수를 이용 다중회귀분석을 수행하시오.

# 데이터 가져오기
from keras.datasets import boston_housing
(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

# 데이터 확인
print(train_data.shape, train_labels.shape)
print(test_data.shape, test_labels.shape)

# 데이터 프레임으로 변환
import pandas as pd
print(train_labels[:5])
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
df = pd.DataFrame(train_data, columns=column_names)
df.head()

# 학습 데이터 생성
import numpy as np
order = np.argsort(np.random.random(train_labels.shape))
train_data = train_data[order]
train_labels = train_labels[order]

# 데이터 표준화
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std

mean = test_data.mean(axis=0)
std = test_data.std(axis=0)
test_data = (test_data - mean) / std

# 모델 생성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(13,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

from tensorflow.keras.optimizers import Adam
model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics=['mae'])

from keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=20)
history = model.fit(train_data, train_labels, epochs=500, validation_split=0.2, callbacks=[early_stop])
# 321번째에서 학습모듈 종료

# 학습결과 시각화
import matplotlib.pyplot as plt

plt.plot(history.history['mae'], label='train mae')
plt.plot(history.history['val_mae'], label='val mae')
plt.xlabel('epoch')
plt.ylabel('mae [$1,000]')
plt.legend(loc='best')
plt.ylim([0, 5])

# 모델 회귀예측 확인
test_loss, test_mae = model.evaluate(test_data, test_labels)
print(np.round(test_labels[:10]))
test_predictions = model.predict(test_data[:10]).flatten()
print(np.round(test_predictions))

##############################################################################
# 데이터 가져오기
from keras.datasets import boston_housing
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

# 데이터와 타겟 구분
train_data.shape
test_data.shape
train_targets

# 데이터 준비
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std

# 모델 구성
from keras import models
from keras import layers

def build_model(): # 동일한 모델을 여러 번 생성할 것이므로 함수를 만들어 사용합니다.
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

# k-겹 교차 검증
import numpy as np

k = 4

num_epochs = 100
all_mae_histories = []
for i in range(k):
    print('처리중인 폴드 #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]  #검증 데이터 준비: k번째 분할    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate(  # 훈련 데이터 준비: 다른 분할 전체
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)

model = build_model()  # 케라스 모델 구성(컴파일 포함)
history = model.fit(partial_train_data, partial_train_targets,  # 모델 훈련(verbose=0이므로 훈련 과정이 출력되지 않습니다.)
                    validation_data=(val_data, val_targets),
                    epochs=num_epochs, batch_size=1, verbose=0)
mae_history = history.history['val_mean_absolute_error']
all_mae_histories.append(mae_history)

# 2. keras package 내 Boston Housing Price 데이터셋 대상으로 기존의 R 함수를 이용하여 다중회귀분석을 실행하고 (1)번의 결과와 비교하시오
# project9_lsg.R 과 비교

# 3. ‘Deep Learning with R’ 교재 내 5.2 Using convnets with small datasets 섹션에서 개와 고양이의 이미지 분류 문제에서 다음 조건을 고려하여 실행하시오.

# 1) 다음의 데이터를 사용하여 분류 문제를 실행하고 시각화하시오.


# 2) 분류의 정확성을 높이기 위한 방법과 사용자가 조정할 수 있는 parameter는 무엇인지 기술하시오.
