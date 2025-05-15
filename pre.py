# 导入所需的库
import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib

# 加载训练好的随机森林模型
model = joblib.load('rf.pkl')
# 加载测试数据集（假设已经存在）
X_test = pd.read_csv("x_test.csv")

# 定义特征名称对应数据集中的列名
feature_names = X_test.columns[:7].tolist()  # 假设只使用前三个特征

# Streamlit 用户界面
st.title("跌倒预测器")  # 设置网页标题

# 用户输入
age= st.number_input("年龄:", min_value=60, max_value=105, value=60)
gender = st.selectbox("性别:", options=[0, 1], format_func=lambda x: "男" if x == 1 else "女")
waist = st.number_input("腰围(cm):", min_value=57.0, max_value=115.0, value=97.0,step=0.1)
sleep = st.number_input("夜晚睡眠时长(h):", min_value=1.0, max_value=12.0, value=8.0, step=0.1)
lgrip = st.number_input("左手握力(kg):", min_value=1.0, max_value=60.0, value=10.0, step=0.1)
depression = st.selectbox("抑郁情况:", options=[1, 2, 3, 4], format_func=lambda x: f" {x}级")
diabete = st.selectbox("糖尿病:", options=[0, 1], format_func=lambda x: "是" if x == 1 else "否")


# 将用户输入组合成特征向量
feature_values=[age, gender, waist, sleep, lgrip, depression, diabete]
features = np.array([feature_values])

# 当用户点击 "Predict" 按钮时执行预测
if st.button("Predict"):
    # 预测类别 (0: 无跌倒风险，1: 有跌倒风险)
    predicted_class = model[0].predict(features)[0]
    # 预测类别的概率
    predicted_proba = model[0].predict_proba(features)[0]
    # 显示预测结果
    st.write(f"**Predicted class:** {predicted_class} (1: High Risk of Fall, 0: Low Risk of Fall)")
    st.write(f"**Prediction probabilities:** {predicted_proba}")
    
    # 根据预测结果生成建议
    probability = predicted_proba[predicted_class] * 100
    if predicted_class == 1:
        advice = (f"According to our model, you have a high risk of falling. "
                  f"The model predicts that your probability of falling is {probability:.1f}%. "
                  "It's advised to take extra precautions, such as using assistive devices, "
                  "improving home safety, and consulting with a healthcare provider.")
    else:
        advice = (f"According to our model, you have a low risk of falling. "
                  f"The model predicts that your probability of not falling is {probability:.1f}%. "
                  "However, maintaining balance and strength exercises can further reduce the risk. "
                  "Please continue regular check-ups with your healthcare provider.")
    # 显示建议
    st.write(advice)
    
    # SHAP 解释
    st.subheader("SHAP Force plot Explanation")



    # 创建 SHAP 解释器，基于树模型（如随机森林）
    explainer_shap = shap.TreeExplainer(model[0])
    # 计算 SHAP 值，用于解释模型的预测
    shap_values = explainer_shap(pd.DataFrame([feature_values],columns=feature_names))
    # 使用 Matplotlib 绘图
    if predicted_class == 1:
        shap.force_plot(explainer_shap.expected_value[1], shap_values[1].values[:, 1], matplotlib=True, show=True,feature_names=feature_names)
    else:
        shap.force_plot(explainer_shap.expected_value[0], shap_values[0].values[:, 0], matplotlib=True, show=True,feature_names=feature_names)
    plt.savefig("shap_force_plot.png",bbox_inches='tight',dpi=300)
    st.image("shap_force_plot.png",caption='SHAP Force Plot Explanation')