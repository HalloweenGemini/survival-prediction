from flask import Flask, render_template, request, jsonify, send_from_directory
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import json
import shap 

app = Flask(__name__)
# 폴더 경로
KDE_FOLDER = 'kde_data'
BAR_FOLDER = 'bar_data'

NUMERIC_VARS = ["PTMIHIBP","PTMILOBP","PTMIPULS","PTMIBRTH","PTMIBDHT","PTMIVOXS"]
CATEGORICAL_VARS = ["PTMIBRTD","PTMISEXX","PTMIRESP"]


################################################
# (4) 유틸: 나이-> PTMIBRTD
################################################
def age_to_ptmibrtd(age_val: float) -> int:
    # 예: 1~26 코드
    if age_val <1: return 1
    elif age_val <5: return 2
    elif age_val <10: return 3
    elif age_val <15: return 4
    elif age_val <20: return 5
    elif age_val <25: return 6
    elif age_val <30: return 7
    elif age_val <35: return 8
    elif age_val <40: return 9
    elif age_val <45: return 10
    elif age_val <50: return 11
    elif age_val <55: return 12
    elif age_val <60: return 13
    elif age_val <65: return 14
    elif age_val <70: return 15
    elif age_val <75: return 16
    elif age_val <80: return 17
    elif age_val <85: return 18
    elif age_val <90: return 19
    elif age_val <95: return 20
    elif age_val <100: return 21
    elif age_val <105: return 22
    elif age_val <110: return 23
    elif age_val <115: return 24
    elif age_val <120: return 25
    else: return 26

################################################
# (5) 유틸: 성별 M/F -> PTMISEXX(1/2)
################################################
def sex_to_ptmisexx(sex_str: str) -> int:
    return 1 if sex_str=='M' else 2

################################################
# (6) 유틸: AVPU -> 0,1,2,3
################################################
resp_map = {'A':0, 'V':1, 'P':2, 'U':3}
def resp_to_code(resp_str: str) -> int:
    return resp_map.get(resp_str, 0)


# 1) 모델 로드
model = joblib.load('best_model_41.pkl')  # 학습된 XGBoost 모델

# 2) SHAP explainer 준비 (TreeExplainer)
explainer = shap.TreeExplainer(model)

# # 2) 데이터 로드 (분포 시각화용)
# try:
#     df = pd.read_csv('D:\Dropbox\SNUH_OS\99. 연구\JE20210901_PTMI1719_Selected_CSV.csv')
# except:
#     df = None

# 가령 df에는 아래 컬럼이 있다고 가정 (사용자 상황에 맞게 변경)
#   'Age', 'Sexuality', 'Response', 'SBP', 'DBP', 'Pulse', 'Breath', 'Temperature', 'SpO2'
#   'Outcome' (Survival/Death) -> 실제 예측 대상?

# (3) 학습 시 사용한 열 이름(순서)
#     enable_categorical=True + 문자열범주(PTMIRESP=A/V/P/U 등)
################################################
FEAT_NAMES = [
    "PTMIHIBP",  # 수축기혈압
    "PTMILOBP",  # 이완기혈압
    "PTMIPULS",  # 맥박
    "PTMIBRTH",  # 호흡수
    "PTMIBDHT",  # 체온
    "PTMIVOXS",  # 산소포화도
    "PTMIBRTD",  # 연령코드(1~26)
    "PTMISEXX",  # 성별(1=남,2=여)
    "PTMIRESP"   # 반응(A,V,P,U)
]

@app.route('/')
def index():
    """
    메인 페이지: templates/index.html
    """
    return render_template('index.html')

@app.route('/kde_data/<var_name>', methods=['GET'])
def get_kde(var_name):
    """
    수치형 변수를 받아서, 해당 KDE json 파일을 반환
    예: /kde_data/PTMIHIBP  => kde_data/kde_ptmihibp.json
    """
    var_name_upper = var_name.upper()
    if var_name_upper not in NUMERIC_VARS:
        return jsonify({'error': f"{var_name} is not numeric variable."}), 400

    filename = f'kde_{var_name.lower()}.json'
    filepath = os.path.join(KDE_FOLDER, filename)
    if not os.path.exists(filepath):
        return jsonify({'error':f"KDE file not found for {var_name}"}), 404

    with open(filepath, 'r') as f:
        data = json.load(f)
    return jsonify(data)

@app.route('/bar_data/<var_name>', methods=['GET'])
def get_bar(var_name):
    """
    범주형 변수를 받아서, 해당 bar json 파일을 반환
    예: /bar_data/PTMIBRTD => bar_data/bar_ptmibrtd.json
    """
    var_name_upper = var_name.upper()
    if var_name_upper not in CATEGORICAL_VARS:
        return jsonify({'error': f"{var_name} is not categorical variable."}), 400

    filename = f'bar_{var_name.lower()}.json'
    filepath = os.path.join(BAR_FOLDER, filename)
    if not os.path.exists(filepath):
        return jsonify({'error':f"Bar file not found for {var_name}"}), 404

    with open(filepath, 'r') as f:
        data = json.load(f)
    return jsonify(data)

# @app.route('/summary', methods=['GET'])
# def summary():
#     if df is None:
#         return jsonify({'error': 'No data.'})

#     # 1) describe 결과
#     desc = df[["Age","SBP","DBP","Pulse","Breath","Temperature","SpO2"]].describe()  
#     #   Age, SBP 등 원하는 변수만 describe
#     desc_json = desc.to_dict()  
#     # → 예) {
#     #       "Age": {"count":..., "mean":..., "std":..., "min":..., "25%":..., ...},
#     #       "SBP": {...},
#     #       ...
#     #     }

#     # 2) 추가로, 각 컬럼의 raw 배열(또는 샘플)도 넘겨줄 수 있음:
#     #    굳이 전부 넘기는 대신, .sample(), .dropna() 등을 활용해 사이즈를 줄이는 방법도.
#     dist_data = {}
#     for col in ["Age","SBP","DBP","Pulse","Breath","Temperature","SpO2"]:
#         dist_data[col] = df[col].dropna().sample(n=5000, replace=False, random_state=42).tolist() \
#                             if len(df[col].dropna())>5000 \
#                             else df[col].dropna().tolist()

#     return jsonify({
#         'describe': desc_json,
#         'dist_data': dist_data
#     })


# @app.route('/distributions', methods=['GET'])
# def distributions():
#     """
#     분포 시각화를 위해 각 변수의 값들을 JSON으로 넘겨준다.
#     """
#     if df is None:
#         return jsonify({'error': 'No data available for distribution.'})

#     # 연속형 변수들만 우선 예시 (Age, SBP, DBP, Pulse, Breath, Temperature, SpO2)
#     # 범주형(Sexuality, Response)도 막대그래프 등으로 가능하지만 여기서는 간단히 예시
#     numeric_cols = ['Age','SBP','DBP','Pulse','Breath','Temperature','SpO2']
#     dist_data = {}
#     for col in numeric_cols:
#         if col in df.columns:
#             series = df[col].dropna() 
#             dist_data[col] = series.tolist()

#     return jsonify(dist_data)

@app.route('/predict', methods=['POST'])
def predict():
    """
    사용자 입력 -> 모델 예측
    """
    data = request.json
    # e.g. {
    #  "Age":70, "Sexuality":"M", "Response":"A",
    #  "SBP":120, "DBP":80, "Pulse":70, "Breath":15, "Temperature":36.5, "SpO2":98
    # }

    # 1) 입력 파싱
    age_val = float(data.get('Age',0))
    sbp = float(data.get('SBP',0))       # PTMIHIBP
    dbp = float(data.get('DBP',0))       # PTMILOBP
    pulse= float(data.get('Pulse',0))    # PTMIPULS
    breath=float(data.get('Breath',0))   # PTMIBRTH
    temp  =float(data.get('Temperature',0)) # PTMIBDHT
    spo2 = float(data.get('SpO2',0))     # PTMIVOXS
    sex_str= data.get('Sexuality','M')   # PTMISEXX
    resp_str=data.get('Response','A')    # PTMIRESP

    # 2) 변환
    ptmibrtd = age_to_ptmibrtd(age_val)
    ptmisexx = sex_to_ptmisexx(sex_str)
    ptmiresp = resp_to_code(resp_str)    # 0,1,2,3

    # 3) X_input 순서 (학습 시와 동일)
    X_input_list = [
        sbp,       # PTMIHIBP
        dbp,       # PTMILOBP
        pulse,     # PTMIPULS
        breath,    # PTMIBRTH
        temp,      # PTMIBDHT
        spo2,      # PTMIVOXS
        ptmibrtd,  # PTMIBRTD
        ptmisexx,  # PTMISEXX
        ptmiresp    # PTMIRESP(int)
    ]

    dinput = xgb.DMatrix(
        [X_input_list],
        enable_categorical=True,
        feature_names=FEAT_NAMES
    )
 
    # 예측 확률 (생존 확률) 뽑는다고 가정
    # model.predict_proba(...) 중 양성(1)에 해당하는 확률
    proba = model.predict(dinput, validate_features=False)[0]
    survival_rate = round(proba*100, 2)

    # 5) SHAP
    shap_vals_2d = explainer.shap_values(dinput)  # shape=(1, 9) (가능하다면)
    shap_vals_1d = shap_vals_2d[0]
    shap_dict = {}
    for i,feat in enumerate(FEAT_NAMES):
        shap_dict[feat] = float(shap_vals_1d[i])
    print(shap_vals_1d)


    # 6) userVals
    userVals = {
        "numeric": {
            "PTMIHIBP": sbp,
            "PTMILOBP": dbp,
            "PTMIPULS": pulse,
            "PTMIBRTH": breath,
            "PTMIBDHT": temp,
            "PTMIVOXS": spo2
        },
        "categorical": {
            "PTMIBRTD": ptmibrtd,
            "PTMISEXX": ptmisexx,
            "PTMIRESP": ptmiresp
        }
    }

    return jsonify({
        "survival_rate": survival_rate,
        "shap_values": shap_dict,
        "userVals": userVals
    })

if __name__ == '__main__':
    app.run(debug=True, port=5001, host='0.0.0.0')
