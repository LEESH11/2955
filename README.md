# 인용하는 방법
---
>> 인용 안에 인용문은
>>> 계속해서 추가 가능하다.
>>>
# GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification

# 예시 모델 정의
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)

# 예시 데이터 생성
x_train, y_train = make_classification(n_samples=100, n_features=20, random_state=42)

# 파라미터 그리드 설정
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1.0]
}

# GridSearchCV 설정
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')

# 모델 학습
grid_search.fit(x_train, y_train)

# 결과 출력
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.2f}")
