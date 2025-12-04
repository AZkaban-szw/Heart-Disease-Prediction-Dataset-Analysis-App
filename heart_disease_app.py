# 导入必要的库
# Import necessary libraries
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

# 定义中英文双语字典，用于界面文本的国际化
# Define bilingual dictionary for internationalization of interface text
LANG = {
    "zh": {
        "page_title": "心脏病风险预测系统",
        "sidebar_title": "心脏病风险分析系统",
        "nav_data": "数据概览",
        "nav_eda": "探索性数据分析",
        "nav_predict": "个人风险预测",
        "nav_eval": "模型性能评估",
        "data_loaded": "数据已自动平衡：总样本 {total:,} 条 | 有心脏病 {pos:,} 条（占比 {pct:.1f}%）",
        "avg_age_title": "有/无心脏病患者的平均年龄对比",
        "age_x": "平均年龄（岁）",
        "age_no": "无心脏病",
        "age_yes": "有心脏病",
        "age_summary": "有心脏病患者平均年龄 **{yes:.1f}岁**，比无心脏病患者高 **{diff:.1f}岁**",
        "age_sample": "样本量：无心脏病 {n_no:,} 人 | 有心脏病 {n_yes:,} 人",
        "bmi": "BMI对比",
        "sbp": "收缩压对比",
        "chol": "总胆固醇对比",
        "gender": "性别分布",
        "smoking": "吸烟情况",
        "activity": "运动频率",
        "hypertension_pie": "高血压患者心脏病比例",
        "predict_title": "个人心脏病风险预测",
        "age": "年龄",
        "height": "身高(cm)",
        "weight": "体重(kg)",
        "your_bmi": "您的BMI",
        "gender_label": "性别",
        "smoking_label": "吸烟",
        "alcohol_label": "饮酒频率",
        "activity_label": "运动水平",
        "diet_label": "饮食习惯",
        "stress_label": "压力水平",
        "hypertension_label": "是否有高血压",
        "diabetes_label": "是否有糖尿病",
        "hyperlipidemia_label": "是否有高血脂",
        "family_history_label": "是否有家族史",
        "predict_btn": "开始预测",
        "prob_label": "患病概率",
        "high_risk": "高风险！建议尽快就医",
        "low_risk": "低风险！继续保持健康生活",
        "eval_title": "模型性能评估",
        "accuracy": "测试集准确率",
        "feature_importance_title": "心脏病风险因素重要性排行榜（Top 10）",
        "feature_importance_desc": "基于逻辑回归系数绝对值排序，数值越大表示该因素对预测结果影响越大",
        "select_viz": "选择图形",
        "male": "男",
        "female": "女",
        "smoking_never": "从不吸烟",
        "smoking_former": "已戒烟",
        "smoking_current": "目前吸烟",
        "alcohol_none": "不饮酒",
        "alcohol_low": "少量",
        "alcohol_moderate": "适量",
        "alcohol_high": "大量",
        "activity_sedentary": "很少运动",
        "activity_moderate": "适度运动",
        "activity_active": "经常运动",
        "diet_healthy": "健康",
        "diet_average": "一般",
        "diet_unhealthy": "不健康",
        "stress_low": "低",
        "stress_medium": "中",
        "stress_high": "高",
        "no_hd": "无心脏病",
        "yes_hd": "有心脏病",
    },
    "en": {
        "page_title": "Heart Disease Risk Prediction System",
        "sidebar_title": "Heart Disease Risk Analysis System",
        "nav_data": "Data Overview",
        "nav_eda": "Exploratory Data Analysis",
        "nav_predict": "Personal Risk Prediction",
        "nav_eval": "Model Performance Evaluation",
        "data_loaded": "Data auto-balanced: Total {total:,} samples | Heart Disease {pos:,} ({pct:.1f}%)",
        "avg_age_title": "Average Age Comparison: With/Without Heart Disease",
        "age_x": "Average Age (years)",
        "age_no": "No Heart Disease",
        "age_yes": "Heart Disease",
        "age_summary": "Patients with heart disease: average **{yes:.1f} years**, **{diff:.1f} years** older",
        "age_sample": "Sample size: No HD {n_no:,} | With HD {n_yes:,}",
        "bmi": "BMI Comparison",
        "sbp": "Systolic BP Comparison",
        "chol": "Total Cholesterol Comparison",
        "gender": "Gender Distribution",
        "smoking": "Smoking Status",
        "activity": "Physical Activity Level",
        "hypertension_pie": "Heart Disease Rate in Hypertensive Patients",
        "predict_title": "Personal Heart Disease Risk Prediction",
        "age": "Age",
        "height": "Height (cm)",
        "weight": "Weight (kg)",
        "your_bmi": "Your BMI",
        "gender_label": "Gender",
        "smoking_label": "Smoking",
        "alcohol_label": "Alcohol Intake",
        "activity_label": "Physical Activity",
        "diet_label": "Diet",
        "stress_label": "Stress Level",
        "hypertension_label": "Has Hypertension?",
        "diabetes_label": "Has Diabetes?",
        "hyperlipidemia_label": "Has Hyperlipidemia?",
        "family_history_label": "Family History?",
        "predict_btn": "Start Prediction",
        "prob_label": "Risk Probability",
        "high_risk": "High Risk! Seek medical attention promptly",
        "low_risk": "Low Risk! Keep up the healthy lifestyle",
        "eval_title": "Model Performance Evaluation",
        "accuracy": "Test Set Accuracy",
        "feature_importance_title": "Top 10 Risk Factors for Heart Disease",
        "feature_importance_desc": "Ranked by absolute value of logistic regression coefficients",
        "select_viz": "Select Visualization",
        "male": "Male",
        "female": "Female",
        "smoking_never": "Never",
        "smoking_former": "Former",
        "smoking_current": "Current",
        "alcohol_none": "None",
        "alcohol_low": "Low",
        "alcohol_moderate": "Moderate",
        "alcohol_high": "High",
        "activity_sedentary": "Sedentary",
        "activity_moderate": "Moderate",
        "activity_active": "Active",
        "diet_healthy": "Healthy",
        "diet_average": "Average",
        "diet_unhealthy": "Unhealthy",
        "stress_low": "Low",
        "stress_medium": "Medium",
        "stress_high": "High",
        "no_hd": "No Heart Disease",
        "yes_hd": "Heart Disease",
    }
}

# 设置页面配置：标题、布局和侧边栏初始状态
# Set page configuration: title, layout and initial sidebar state
st.set_page_config(page_title="心脏病风险预测系统", layout="wide", initial_sidebar_state="expanded")

# 在侧边栏添加标题
# Add title to sidebar
st.sidebar.markdown("# 心脏病风险分析系统")

# 在侧边栏添加语言选择器
# Add language selector to sidebar
lang = st.sidebar.radio("Language / 语言", ["中文", "English"], index=0)

# 根据选择的语言加载对应的文本字典
# Load corresponding text dictionary based on selected language
txt = LANG["zh"] if lang == "中文" else LANG["en"]

# 在侧边栏添加分隔线
# Add separator in sidebar
st.sidebar.markdown("---")

# 在侧边栏创建两列用于显示合作院校的logo
# Create two columns in sidebar to display partner institutions' logos
col1, col2 = st.sidebar.columns(2)
with col1:
    st.image("WUT-Logo.png", width=100)  # 显示武汉理工大学logo
    st.caption("武汉理工大学")          # 显示学校名称
with col2:
    st.image("efrei-logo.png", width=100)  # 显示法国EFREI巴黎logo
    st.caption("EFREI Paris")             # 显示学校名称

# 在侧边栏添加作者信息
# Add author information in sidebar
st.sidebar.markdown("### Author")
st.sidebar.markdown("Ziwei Shan")

# 在侧边栏添加GitHub链接
# Add GitHub link in sidebar
st.sidebar.markdown("### GitHub")
st.sidebar.markdown("[github.com/AZkaban-szw/Heart-Disease-Prediction-Dataset-Analysis-App](https://github.com/AZkaban-szw/Heart-Disease-Prediction-Dataset-Analysis-App)")

# 在侧边栏添加课程信息
# Add course information in sidebar
st.sidebar.markdown("### Course")
st.sidebar.markdown("Data Visualization 2025")

# 在侧边栏添加教授信息
# Add professor information in sidebar
st.sidebar.markdown("### Professor")
st.sidebar.markdown("Mano Mathew")

# 使用Streamlit缓存装饰器缓存数据加载函数，避免重复加载
# Use Streamlit cache decorator to cache data loading function to avoid repeated loading
@st.cache_data
def load_data():
    # 读取CSV格式的心脏病数据集
    # Read CSV format heart disease dataset
    df = pd.read_csv("synthetic_heart_disease_dataset.csv")
    
    # 分离有心脏病和无心脏病的样本
    # Separate samples with and without heart disease
    df_pos = df[df["Heart_Disease"] == 1]
    df_neg = df[df["Heart_Disease"] == 0].sample(n=len(df_pos), random_state=42)
    
    # 平衡数据集并打乱顺序
    # Balance dataset and shuffle order
    df_balanced = pd.concat([df_pos, df_neg], ignore_index=True)
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    return df_balanced

# 加载平衡后的数据集
# Load balanced dataset
df_raw = load_data()

# 计算总样本数和有心脏病的样本数
# Calculate total number of samples and number of samples with heart disease
total = len(df_raw)
pos_count = len(df_raw[df_raw["Heart_Disease"] == 1])

# 显示数据加载成功的信息，包含样本统计
# Display data loading success message with sample statistics
st.success(txt["data_loaded"].format(total=total, pos=pos_count, pct=pos_count/total*100))

# 创建用于可视化的数据副本，并转换分类变量为当前语言
# Create a copy of data for visualization and convert categorical variables to current language
df_plot = df_raw.copy()
df_plot["Gender"] = df_plot["Gender"].map({"Male": txt["male"], "Female": txt["female"]})
df_plot["Smoking"] = df_plot["Smoking"].map({"Never": txt["smoking_never"], "Former": txt["smoking_former"], "Current": txt["smoking_current"]})
df_plot["Physical_Activity"] = df_plot["Physical_Activity"].map({
    "Sedentary": txt["activity_sedentary"],
    "Moderate": txt["activity_moderate"],
    "Active": txt["activity_active"]
})
df_plot["心脏病状态"] = df_plot["Heart_Disease"].map({0: txt["no_hd"], 1: txt["yes_hd"]})

# 准备用于模型训练的数据
# Prepare data for model training
df_train = df_raw.copy()
le_dict = {}  # 用于存储各个分类变量的编码器

# 需要编码的分类变量列表
# List of categorical variables that need encoding
cat_cols = ["Gender", "Smoking", "Alcohol_Intake", "Physical_Activity", "Diet", "Stress_Level"]

# 对每个分类变量进行标签编码
# Perform label encoding for each categorical variable
for col in cat_cols:
    le = LabelEncoder()
    df_train[col] = le.fit_transform(df_train[col].astype(str))
    le_dict[col] = le  # 保存编码器以便后续预测使用

# 分离特征和目标变量
# Separate features and target variable
X = df_train.drop("Heart_Disease", axis=1)
y = df_train["Heart_Disease"]

# 将数据集分为训练集和测试集（80%训练，20%测试）
# Split dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 使用Streamlit缓存装饰器缓存模型训练函数，避免重复训练
# Use Streamlit cache decorator to cache model training function to avoid repeated training
@st.cache_resource
def train_model():
    # 创建逻辑回归模型并训练
    # Create and train logistic regression model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

# 训练模型
# Train the model
model = train_model()

# 计算模型在测试集上的准确率
# Calculate model accuracy on test set
acc = accuracy_score(y_test, model.predict(X_test))

# 在侧边栏添加功能导航选项
# Add function navigation options in sidebar
option = st.sidebar.radio("功能导航 / Navigation", [
    txt["nav_data"], txt["nav_eda"], txt["nav_predict"], txt["nav_eval"]
])

# 设置页面标题
# Set page title
st.title("心脏病预测数据集分析系统" if lang == "中文" else "Heart Disease Prediction Analysis System")

# 根据用户选择的导航选项显示相应内容
# Display corresponding content based on user's navigation selection
if option == txt["nav_data"]:
    # 数据概览页面
    # Data overview page
    st.header(txt["nav_data"])
    
    # 创建两列布局：左侧显示数据表格，右侧显示心脏病分布饼图
    # Create two-column layout: data table on left, heart disease distribution pie chart on right
    col1, col2 = st.columns([3, 1])
    with col1:
        # 显示数据集的前10行
        # Display first 10 rows of dataset
        st.dataframe(df_raw.head(10), use_container_width=True)
    with col2:
        # 创建并显示心脏病分布饼图
        # Create and display heart disease distribution pie chart
        fig = px.pie(values=df_raw["Heart_Disease"].value_counts(),
                     names=[txt["no_hd"], txt["yes_hd"]],
                     color_discrete_sequence=["#00CC96", "#FF4444"], hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
        
elif option == txt["nav_eda"]:
    # 探索性数据分析页面
    # Exploratory Data Analysis page
    st.header(txt["nav_eda"])
    
    # 提供可视化选项供用户选择
    # Provide visualization options for users to choose from
    viz = st.selectbox(txt["select_viz"], [
        txt["avg_age_title"], txt["bmi"], txt["sbp"], txt["chol"],
        txt["gender"], txt["smoking"], txt["activity"], txt["hypertension_pie"]
    ])
    
    # 根据用户选择显示相应的可视化图表
    # Display corresponding visualization chart based on user selection
    if viz == txt["avg_age_title"]:
        # 计算有/无心脏病患者的平均年龄
        # Calculate average age of patients with/without heart disease
        grouped = df_plot.groupby("心脏病状态")["Age"].mean().round(1)
        count = df_plot["心脏病状态"].value_counts()
        age_no = grouped.get(txt["no_hd"], 0)
        age_yes = grouped.get(txt["yes_hd"], 0)
        n_no = count.get(txt["no_hd"], 0)
        n_yes = count.get(txt["yes_hd"], 0)
        
        # 创建平均年龄对比条形图
        # Create average age comparison bar chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=[txt["no_hd"], txt["yes_hd"]],
            x=[age_no, age_yes],
            orientation='h',
            text=[f"{age_no}岁" if lang=="中文" else f"{age_no} yrs", f"{age_yes}岁" if lang=="中文" else f"{age_yes} yrs"],
            textposition="outside",
            marker_color=["#00CC96", "#FF4444"],
        ))
        fig.update_layout(
            title=txt["avg_age_title"],
            xaxis_title=txt["age_x"],
            yaxis_title="",
            showlegend=False,
            height=450,
            xaxis=dict(range=[0, age_yes + 15]),
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=140, r=60, t=80, b=60)
        )
        
        # 显示年龄对比的统计信息和图表
        # Display statistical information and chart for age comparison
        diff = age_yes - age_no
        st.success(txt["age_summary"].format(yes=age_yes, diff=diff))
        st.caption(txt["age_sample"].format(n_no=n_no, n_yes=n_yes))
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz == txt["bmi"]:
        # 创建BMI对比箱线图
        # Create BMI comparison box plot
        fig = px.box(df_plot, x="心脏病状态", y="BMI", color="心脏病状态",
                     color_discrete_sequence=["#00CC96", "#FF4444"])
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz == txt["sbp"]:
        # 创建收缩压对比箱线图
        # Create systolic blood pressure comparison box plot
        fig = px.box(df_plot, x="心脏病状态", y="Systolic_BP", color="心脏病状态",
                     color_discrete_sequence=["#00CC96", "#FF4444"])
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz == txt["chol"]:
        # 创建总胆固醇对比箱线图
        # Create total cholesterol comparison box plot
        fig = px.box(df_plot, x="心脏病状态", y="Cholesterol_Total", color="心脏病状态",
                     color_discrete_sequence=["#00CC96", "#FF4444"])
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz == txt["gender"]:
        # 创建性别分布直方图
        # Create gender distribution histogram
        fig = px.histogram(df_plot, x="Gender", color="心脏病状态", barmode="group",
                           color_discrete_sequence=["#00CC96", "#FF4444"])
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz == txt["smoking"]:
        # 创建吸烟情况分布直方图
        # Create smoking status distribution histogram
        fig = px.histogram(df_plot, x="Smoking", color="心脏病状态", barmode="group",
                           color_discrete_sequence=["#00CC96", "#FF4444"])
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz == txt["activity"]:
        # 创建运动水平分布直方图
        # Create physical activity level distribution histogram
        fig = px.histogram(df_plot, x="Physical_Activity", color="心脏病状态", barmode="group",
                           color_discrete_sequence=["#00CC96", "#FF4444"])
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz == txt["hypertension_pie"]:
        # 筛选出有高血压的患者
        # Filter out patients with hypertension
        hyper = df_plot[df_plot["Hypertension"] == 1]
        if len(hyper) > 0:
            # 创建高血压患者中心脏病比例的饼图
            # Create pie chart of heart disease proportion among hypertensive patients
            fig = px.pie(hyper, names="心脏病状态", color_discrete_sequence=["#00CC96", "#FF4444"])
            st.plotly_chart(fig, use_container_width=True)
            
elif option == txt["nav_predict"]:
    # 个人风险预测页面
    # Personal risk prediction page
    st.header(txt["predict_title"])
    
    # 创建输入表单的布局
    # Create layout for input form
    col1, col2 = st.columns(2)
    with col1:
        # 收集基本信息输入
        # Collect basic information inputs
        age = st.slider(txt["age"], 18, 100, 55)
        height = st.number_input(txt["height"], 140, 220, 170)
        weight = st.number_input(txt["weight"], 40, 150, 70)
        gender = st.radio(txt["gender_label"], [txt["male"], txt["female"]])
        
    with col2:
        # 收集生活习惯相关输入
        # Collect lifestyle-related inputs
        smoking = st.selectbox(txt["smoking_label"], [txt["smoking_never"], txt["smoking_former"], txt["smoking_current"]])
        alcohol = st.selectbox(txt["alcohol_label"], [txt["alcohol_none"], txt["alcohol_low"], txt["alcohol_moderate"], txt["alcohol_high"]])
        activity = st.selectbox(txt["activity_label"], [txt["activity_sedentary"], txt["activity_moderate"], txt["activity_active"]])
        diet = st.selectbox(txt["diet_label"], [txt["diet_healthy"], txt["diet_average"], txt["diet_unhealthy"]])
        
    col3, col4 = st.columns(2)
    with col3:
        # 收集健康状况相关输入
        # Collect health status-related inputs
        stress = st.selectbox(txt["stress_label"], [txt["stress_low"], txt["stress_medium"], txt["stress_high"]])
        hypertension = st.checkbox(txt["hypertension_label"])
        diabetes = st.checkbox(txt["diabetes_label"])
        
    with col4:
        # 收集更多健康状况相关输入
        # Collect more health status-related inputs
        hyperlipidemia = st.checkbox(txt["hyperlipidemia_label"])
        family_history = st.checkbox(txt["family_history_label"])
    
    # 计算并显示BMI
    # Calculate and display BMI
    bmi = round(weight / ((height/100)**2), 1)
    st.metric(txt["your_bmi"], bmi)
    
    # 当用户点击预测按钮时执行预测
    # Perform prediction when user clicks the prediction button
    if st.button(txt["predict_btn"], type="primary"):
        # 将用户输入的文本值映射回原始数据集中使用的值
        # Map user input text values back to those used in original dataset
        gender_map = {txt["male"]: "Male", txt["female"]: "Female"}
        smoking_map = {txt["smoking_never"]: "Never", txt["smoking_former"]: "Former", txt["smoking_current"]: "Current"}
        alcohol_map = {txt["alcohol_none"]: "None", txt["alcohol_low"]: "Low", txt["alcohol_moderate"]: "Moderate", txt["alcohol_high"]: "High"}
        activity_map = {txt["activity_sedentary"]: "Sedentary", txt["activity_moderate"]: "Moderate", txt["activity_active"]: "Active"}
        diet_map = {txt["diet_healthy"]: "Healthy", txt["diet_average"]: "Average", txt["diet_unhealthy"]: "Unhealthy"}
        stress_map = {txt["stress_low"]: "Low", txt["stress_medium"]: "Medium", txt["stress_high"]: "High"}
        
        # 组织用户输入的数据为字典
        # Organize user input data into a dictionary
        input_data = {
            "Age": age, "Height": height, "Weight": weight, "BMI": bmi,
            # 对于未收集的生理指标使用默认值
            "Systolic_BP": 120, "Diastolic_BP": 80, "Heart_Rate": 75, "Blood_Sugar_Fasting": 90, "Cholesterol_Total": 200,
            "Gender": gender_map[gender],
            "Smoking": smoking_map[smoking],
            "Alcohol_Intake": alcohol_map[alcohol],
            "Physical_Activity": activity_map[activity],
            "Diet": diet_map[diet],
            "Stress_Level": stress_map[stress],
            "Hypertension": int(hypertension),
            "Diabetes": int(diabetes),
            "Hyperlipidemia": int(hyperlipidemia),
            "Family_History": int(family_history),
            "Previous_Heart_Attack": 0,  # 默认为无既往心梗史
        }
        
        # 将输入数据转换为DataFrame
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # 对分类变量进行编码，与训练模型时的编码方式保持一致
        # Encode categorical variables consistently with model training
        for col, le in le_dict.items():
            if col in input_df.columns:
                val = input_df.iloc[0][col]
                # 处理未见类别的情况
                if val not in le.classes_:
                    val = le.classes_[0]
                input_df[col] = le.transform([val])[0]
        
        # 使用模型预测心脏病风险概率
        # Use model to predict heart disease risk probability
        prob = model.predict_proba(input_df[X.columns])[0][1]
        
        # 显示预测结果
        # Display prediction result
        st.metric(txt["prob_label"], f"{prob:.1%}")
        if prob > 0.5:
            st.error(txt["high_risk"])
        else:
            st.success(txt["low_risk"])
            st.balloons()  # 低风险时显示庆祝气球
            
elif option == txt["nav_eval"]:
    # 模型性能评估页面
    # Model performance evaluation page
    st.header(txt["eval_title"])
    
    # 显示模型准确率
    # Display model accuracy
    st.metric(txt["accuracy"], f"{acc:.1%}")
    st.info("数据已平衡，评估结果更真实可靠" if lang == "中文" else "Data is balanced, evaluation results are more reliable")
    
    # 显示特征重要性排行榜
    # Display feature importance ranking
    st.markdown(f"### {txt['feature_importance_title']}")
    st.caption(txt["feature_importance_desc"])
    
    # 计算并处理特征重要性
    # Calculate and process feature importance
    importance = np.abs(model.coef_[0])
    imp_df = pd.DataFrame({"特征": X.columns, "重要性": importance}).sort_values("重要性", ascending=False).head(10)
    
    # 根据语言设置特征显示名称
    # Set feature display names based on language
    if lang == "中文":
        name_map = {
            'Diet': '饮食',
            'Cholesterol_Total': '总胆固醇',
            'Stress_Level': '压力水平',
            'Diabetes': '糖尿病',
            'Hypertension': '高血压',
            'Smoking': '吸烟',
            'Physical_Activity': '身体活动',
            'Age': '年龄',
            'Alcohol_Intake': '饮酒',
            'Systolic_BP': '收缩压',
            'Previous_Heart_Attack': '既往心梗',
            'Family_History': '家族史'
        }
    else:
        name_map = {
            'Diet': 'Diet',
            'Cholesterol_Total': 'Total Cholesterol',
            'Stress_Level': 'Stress Level',
            'Diabetes': 'Diabetes',
            'Hypertension': 'Hypertension',
            'Smoking': 'Smoking',
            'Physical_Activity': 'Physical Activity',
            'Age': 'Age',
            'Alcohol_Intake': 'Alcohol Intake',
            'Systolic_BP': 'Systolic BP',
            'Previous_Heart_Attack': 'Previous Heart Attack',
            'Family_History': 'Family History'
        }
    imp_df["显示名称"] = imp_df["特征"].map(name_map).fillna(imp_df["特征"])
    
    # 创建并显示特征重要性条形图
    # Create and display feature importance bar chart
    fig = go.Figure(go.Bar(
        x=imp_df["重要性"],
        y=imp_df["显示名称"],
        orientation='h',
        marker_color='#FF4444',
        text=imp_df["重要性"].round(4),
        textposition='outside'
    ))
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        height=520,
        xaxis_title="重要性得分（越高越关键）" if lang == "中文" else "Importance Score (Higher = More Critical)",
        template="simple_white",
        margin=dict(l=200)
    )
    st.plotly_chart(fig, use_container_width=True)

# 在侧边栏添加重要提示信息
# Add important notes in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("重要提示")
st.sidebar.markdown("Important Notes")
st.sidebar.markdown("1. 该测试结果仅供参考，不构成医疗诊断")
st.sidebar.markdown("1.The test results are for reference only and do not constitute a medical diagnosis.")
st.sidebar.markdown("2. 仅供学习用途，不具备医疗建议参考价值")
st.sidebar.markdown("2.For educational purposes only, and have no reference value for medical advice.")
st.sidebar.markdown("3. 如有健康疑虑，请咨询专业医护人员")
st.sidebar.markdown("3.If you have health concerns, please consult professional medical staff.")