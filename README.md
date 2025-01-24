# 응급실 환자 사망 예측 모델 프로젝트

## 개요

이 프로젝트는 응급실 환자의 초기 생체 징후를 기반으로 사망 위험을 예측하는 머신러닝 모델을 구현합니다.

## 주요 기능

- 환자의 생체 징후 데이터 기반 사망 위험도 예측
- 예측 결과에 대한 SHAP 기반 설명력 제공 
- 수치형/범주형 변수의 분포 시각화
- REST API 엔드포인트 제공

## 기술 스택

- Python 3.8+
- XGBoost
- Flask
- Pandas
- NumPy
- SHAP
- Scikit-learn

## 설치 방법

1. 필요한 패키지 설치:

~~
pip install -r requirements.txt
~~

2. 모델 파일 다운로드:
- `.pkl` 훈련한 파일을 프로젝트 루트 디렉토리에 위치

## 사용 방법

### 1. 오프라인 데이터 준비

~~
python offline_data_preparation.py
~~

이 스크립트는:
- 수치형 변수의 KDE 분포를 계산하여 JSON으로 저장
- 범주형 변수의 빈도를 계산하여 JSON으로 저장

### 2. 서버 실행

~~
python app.py
~~

서버는 기본적으로 http://localhost:5001 에서 실행됩니다.

### 3. API 엔드포인트

#### 예측 요청
- POST `/predict`
- 요청 예시:
~~
{
  "Age": 70,
  "Sexuality": "M", 
  "Response": "A",
  "SBP": 120,
  "DBP": 80,
  "Pulse": 70,
  "Breath": 15,
  "Temperature": 36.5,
  "SpO2": 98
}
~~

- 응답 예시:
~~
{
  "survival_rate": 95.2,
  "shap_values": {
    "PTMIHIBP": 0.123,
    ...
  },
  "userVals": {
    "numeric": {...},
    "categorical": {...}
  }
}
~~

#### 분포 데이터 요청
- GET `/kde_data/<var_name>` : 수치형 변수의 KDE
- GET `/bar_data/<var_name>` : 범주형 변수의 빈도

## 모델 평가

모델의 성능 지표:
- AUROC: 0.977
- AUPRC: 1.000
- 민감도: 0.999
- 특이도: 0.615
- 정확도: 0.999

자세한 평가 결과는 `evaluation.ipynb`에서 확인할 수 있습니다.

## 프로젝트 구조

```
.
├── app.py                     # Flask 서버 
├── offline_data_preparation.py # 데이터 전처리
├── evaluation.py              # 모델 평가
├── train.py                   # 모델 학습
├── best_model_41.pkl          # 학습된 모델
├── kde_data/                  # 수치형 변수 분포
└── bar_data/                  # 범주형 변수 분포
```


## 라이선스

MIT License

## 기여 방법

1. Fork the Project
2. Create your Feature Branch
3. Commit your Changes
4. Push to the Branch
5. Open a Pull Request
