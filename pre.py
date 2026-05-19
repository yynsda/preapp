# 导入所需的库
import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib
from openai import OpenAI
from matplotlib import font_manager
import os



def fun_shap():

    # SHAP 解释
    st.subheader("SHAP Force plot Explanation")

    # 创建 SHAP 解释器，基于树模型（如随机森林）
    explainer_shap = shap.Explainer(model)
    # 计算 SHAP 值，用于解释模型的预测
    shap_values = explainer_shap(pd.DataFrame(features, columns=feature_names))

    # 力图
    #shap.force_plot(explainer_shap.expected_value[1], shap_values[0].values[:, 1], matplotlib=True, show=True,
                        feature_names=feature_names)
    #plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=300)
    #st.image("shap_force_plot.png", caption='SHAP Force Plot Explanation')
    #st.write("shap_values.values.shape =", shap_values.values.shape)

    # ---- 决策图（Decision Plot） ----
    #class_idx = 1#predicted_class  # 0或1
   # sv = shap_values.values[:, :, class_idx]  # shape (1, 9)
   # print("decision_plot input shape:", sv.shape)
    #plt.figure()
    #shap.decision_plot(
        #explainer_shap.expected_value[class_idx],
        #sv,feature_names=feature_names )
   # plt.savefig("shap_decision_plot.png", bbox_inches='tight', dpi=300)
   # st.image("shap_decision_plot.png", caption='SHAP Decision Plot')
    # ---瀑布图
    # 生成单样本、单类别的 SHAP Explanation
    single_sample_shap = shap.Explanation(
        values=shap_values.values[0, :, class_idx],  # shape (n_features,)
        base_values=explainer_shap.expected_value[class_idx],  # base value for该类
        data=features[0],  # 输入原始特征（长度n_features）
        feature_names=feature_names
    )

    # 绘制 waterfall
    plt.figure()
    shap.plots.waterfall(single_sample_shap, max_display=9)
    plt.savefig("shap_waterfall_plot.png", bbox_inches='tight', dpi=300)
    st.image("shap_waterfall_plot.png", caption="SHAP Waterfall Plot")
def chat():
    client = OpenAI(
        api_key="sk-705c1e1bea26422a973f9e196dfc3ba5",
        base_url="https://api.deepseek.com"
    )

    # 3. 请求 DeepSeek API
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一位专业的老年健康管理师，擅长用通俗中文给出实用健康建议。"},
            {"role": "user", "content": prompt},
        ]
    )

    # 4. 显示建议
    advice = response.choices[0].message.content
    st.write("💡 **个性化健康建议：**")
    st.markdown(advice)
feature_zh_map = {
    "tijian_lgrip": "Left Hand Grip (kg)",
    "renkou_age": "Age",
    "shenghuo_sleep": "Night Sleep Duration (h)",
    "jingshen_yiyu": "Depression Scale",
    "shenghuo_wushui": "Nap Duration (h)",
    "tijian_height": "Height (cm)",
    "tijian_weight": "Weight (kg)",
    "shenti_zhanli": "Sit-to-Stand Test Level",
    "renkou_gender": "Gender"
}

# 定义特征名称对应数据集中的列名
df = pd.read_csv('data3.csv')
feature_names = df.columns[:9].tolist()   #
feature_names = [feature_zh_map.get(f, f) for f in feature_names]

# 加载训练好的随机森林模型
model = joblib.load('rf.pkl')
scaler = joblib.load('scaler.pkl')
# Streamlit 用户界面
st.title("跌倒预测器")  # 设置网页标题

# 用户输入
# 第一行
col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("年龄", min_value=60, max_value=105, value=60)
with col2:
    gender = st.selectbox("性别", options=[0, 1], format_func=lambda x: "男" if x == 1 else "女")
with col3:
    lgrip = st.number_input("左手握力(kg)", min_value=1.0, max_value=60.0, value=10.0, step=0.1)

# 第二行
col4, col5, col6 = st.columns(3)
with col4:
    height = st.number_input("身高(cm)", min_value=100.0, max_value=220.0, value=170.0, step=0.1)
with col5:
    weight = st.number_input("体重(kg)", min_value=30.0, max_value=150.0, value=65.0, step=0.1)
with col6:
    sleep = st.number_input("夜晚睡眠时长(h)", min_value=1.0, max_value=12.0, value=8.0, step=0.1)

# 第三行
col7, col8, col9 = st.columns(3)
with col7:
    wushui = st.number_input("午睡时长(h)", min_value=0.0, max_value=5.0, value=0.5, step=0.1)
with col8:
    yiyu = st.number_input("抑郁量表 (CESD-10 得分)", min_value=0, max_value=30, value=10, step=1)
with col9:
    zhanli = st.selectbox("久坐站立测试", options=[1, 2, 3, 4], format_func=lambda x: f"第 {x} 级")

# 将用户输入组合成特征向量
feature_values = [lgrip, age, sleep, yiyu, wushui, height, weight, zhanli, gender]
features = np.array([feature_values])

# 拆分数值型与分类特征
features_num = features[:, :7]       # 前7列，需标准化
features_cat = features[:, 7:]       # 后2列，原始（离散型）
# 标准化
features_num_scaled = scaler.transform(features_num)
# 合并
features = np.hstack((features_num_scaled, features_cat))

# 当用户点击 "Predict" 按钮时执行预测
if st.button("Predict"):
    predicted_class = model.predict(features)[0]
    predicted_proba = round(model.predict_proba(features)[0][1], 3)


if st.button("生成个性化建议"):
    predicted_class = model.predict(features)[0]
    predicted_proba = round(model.predict_proba(features)[0][1], 3)
    # 1. 组合用户所有输入特征，推荐用中英文，越详细越好
    user_info = (
        f"年龄: {age}岁，性别: {'男' if gender == 1 else '女'}，左手握力: {lgrip}kg，"
        f"身高: {height}cm，体重: {weight}kg，夜间睡眠: {sleep}小时，午睡: {wushui}小时，"
        f"抑郁量表: {yiyu}分，久坐站立测试: {zhanli}级。"
    )

    # 2. 用英文或中文给LLM完整描述问题
    prompt = (
        f"已知某老年人特征如下：{user_info}\n"
        f"模型预测该用户的跌倒风险为：{'高' if predicted_class == 1 else '低'} "
        f"（概率约为 {predicted_proba * 100:.1f}%）。\n"
        f"请你作为专业健康管理师，基于以上信息和风险预测，给出详细、实用、个性化的健康建议（包含运动建议、居家环境、心理、饮食、家属提醒等）。请用中文作答。"
    )#背景rag，专用知识库，临床资料 
    # 3. 调用函数
    chat()
fun_shap()



