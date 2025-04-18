# 导入所需的库
import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载训练好的随机森林模型
model = joblib.load('RF.pkl')
# 加载测试数据集（假设已经存在）
X_test = pd.read_csv("x_test.csv")

# 定义特征名称对应数据集中的列名
feature_names = X_test.columns[:3]  # 假设只使用前三个特征

# Streamlit 用户界面
st.title("跌倒预测器")  # 设置网页标题

# 用户输入
gender = st.selectbox("性别:", options=[0, 1], format_func=lambda x: "男" if x == 1 else "女")
depression = st.selectbox("抑郁情况:", options=[1, 2, 3, 4], format_func=lambda x: f"等级 {x}")
waist = st.number_input("腰围(cm):", min_value=57, max_value=115, value=97)

# 将用户输入组合成特征向量
features = np.array([[gender, depression, waist]])

# 当用户点击 "Predict" 按钮时执行预测
if st.button("Predict"):
    # 预测类别 (0: 无跌倒风险，1: 有跌倒风险)
    predicted_class = model.predict(features)[0]
    # 预测类别的概率
    predicted_proba = model.predict_proba(features)[0]
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
    explainer_shap = shap.Explainer(model, X_test)
    # 计算 SHAP 值，用于解释模型的预测
    shap_values = explainer_shap(features)
    # 使用 Matplotlib 绘图
    if predicted_class == 1:
        shap.force_plot(explainer_shap.expected_value[1], shap_values.values[0, :, 1], pd.DataFrame(features, columns=feature_names), matplotlib=True)
    else:
        shap.force_plot(explainer_shap.expected_value[0], shap_values.values[0, :, 0], pd.DataFrame(features, columns=feature_names), matplotlib=True)