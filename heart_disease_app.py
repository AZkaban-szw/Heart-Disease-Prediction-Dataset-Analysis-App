import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# ---------------------- 双语文本配置 ----------------------
LANG_DICT = {
    "zh": {
        "app_title": "❤️ 心脏病预测数据集分析App",
        "sidebar_nav": "📌 功能导航",
        "modules": ["数据概览", "探索性分析（EDA）", "心脏病风险预测", "模型性能评估"],
        "data_overview": "📊 数据基础信息",
        "data_head": "数据前5行",
        "data_stats": "数据集基本统计",
        "data_info": "数据结构信息",
        "target_dist": "目标变量分布",
        "eda_title": "🔍 探索性数据分析（EDA）",
        "eda_types": ["单变量分布", "特征相关性热力图", "特征与目标变量关联", "散点图分析", "小提琴图分析"],
        "select_feat": "选择要分析的特征",
        "cat_feat": "分类特征",
        "num_feat": "数值特征",
        "mean": "均值",
        "corr_heatmap": "特征相关性热力图",
        "target_corr": "特征与目标变量的关联",
        "dist_compare": "分布对比",
        "scatter_title": "双变量散点图（带回归线）",
        "violin_title": "小提琴图（分布密度）",
        "predict_title": "🔮 心脏病风险预测工具",
        "input_info": "请输入以下健康信息（带 * 为必填项）",
        "age": "* 年龄",
        "height": "* 身高(cm)",
        "weight": "* 体重(kg)",
        "bmi": "BMI（自动计算）",
        "hypertension": "* 高血压",
        "diabetes": "* 糖尿病",
        "hyperlipidemia": "* 高血脂",
        "family_history": "* 家族病史",
        "prev_heart_attack": "* 既往心脏病史",
        "systolic_bp": "* 收缩压(mmHg)",
        "diastolic_bp": "* 舒张压(mmHg)",
        "heart_rate": "* 心率(bpm)",
        "blood_sugar": "* 空腹血糖(mg/dL)",
        "cholesterol_total": "* 总胆固醇(mg/dL)",
        "smoking": "* 吸烟状态",
        "alcohol": "* 饮酒量",
        "physical_activity": "* 体力活动",
        "diet": "* 饮食类型",
        "stress_level": "* 压力水平",
        "missing_feat": "⚠️ 缺少以下特征的输入组件：",
        "feat_tip": "请在代码的「心脏病风险预测」模块中，为上述特征添加对应的输入组件（number_input/selectbox）",
        "predict_btn": "📊 开始预测",
        "pred_result": "预测结果",
        "risk_pos": "⚠️ 预测结果：存在心脏病风险",
        "risk_neg": "✅ 预测结果：无心脏病风险",
        "risk_prob": "风险概率",
        "model_desc": "📋 模型说明",
        "model_type": "模型类型：逻辑回归",
        "test_acc": "测试集准确率",
        "medical_tip": "注：该预测仅为数据分析演示，不构成医学诊断依据！",
        "model_eval": "📈 模型性能评估报告",
        "core_metrics": "核心指标",
        "class_metrics": "分类指标详情",
        "conf_matrix": "混淆矩阵",
        "true_label": "真实",
        "pred_label": "预测",
        "model_note": "模型说明",
        "train_data": "训练数据占比：80%（测试集20%）",
        "process_strategy": "处理策略：分类特征LabelEncoder编码",
        "scenario": "适用场景：心脏病风险初步筛查演示",
        "usage_tip": "💡 使用提示：",
        "path_tip": "1. 请先确保数据集路径正确",
        "target_tip": "2. 目标列名需与代码中 target_col 一致",
        "input_tip": "3. 预测模块需补充所有特征的输入组件",
        "tool_tip": "4. 本App仅用于数据分析演示，非医学工具",
        "distribution_by": "{feature} 按 {target} 的分布"  # 新增中文标题模板
    },
    "en": {
        "app_title": "❤️ Heart Disease Prediction Dataset Analysis App",
        "sidebar_nav": "📌 Function Navigation",
        "modules": ["Data Overview", "Exploratory Data Analysis (EDA)", "Heart Disease Risk Prediction", "Model Performance Evaluation"],
        "data_overview": "📊 Basic Data Information",
        "data_head": "First 10 Rows of Data",
        "data_stats": "Basic Data Statistics",
        "data_info": "Data Structure Information",
        "target_dist": "Target Variable Distribution",
        "eda_title": "🔍 Exploratory Data Analysis (EDA)",
        "eda_types": ["Univariate Distribution", "Feature Correlation Heatmap", "Feature-Target Correlation", "Scatter Plot Analysis", "Violin Plot Analysis"],
        "select_feat": "Select Feature to Analyze",
        "cat_feat": "Categorical Feature",
        "num_feat": "Numerical Feature",
        "mean": "Mean",
        "corr_heatmap": "Feature Correlation Heatmap",
        "target_corr": "Feature-Target Correlation",
        "dist_compare": "Distribution Comparison",
        "scatter_title": "Bivariate Scatter Plot (with Regression Line)",
        "violin_title": "Violin Plot (Distribution Density)",
        "predict_title": "🔮 Heart Disease Risk Prediction Tool",
        "input_info": "Please Enter Health Information (* Required)",
        "age": "* Age",
        "height": "* Height(cm)",
        "weight": "* Weight(kg)",
        "bmi": "BMI (Auto-Calculated)",
        "hypertension": "* Hypertension",
        "diabetes": "* Diabetes",
        "hyperlipidemia": "* Hyperlipidemia",
        "family_history": "* Family History",
        "prev_heart_attack": "* Previous Heart Attack",
        "systolic_bp": "* Systolic BP(mmHg)",
        "diastolic_bp": "* Diastolic BP(mmHg)",
        "heart_rate": "* Heart Rate(bpm)",
        "blood_sugar": "* Fasting Blood Sugar(mg/dL)",
        "cholesterol_total": "* Total Cholesterol(mg/dL)",
        "smoking": "* Smoking Status",
        "alcohol": "* Alcohol Intake",
        "physical_activity": "* Physical Activity",
        "diet": "* Diet Type",
        "stress_level": "* Stress Level",
        "missing_feat": "⚠️ Missing Input Components for Features:",
        "feat_tip": "Please add corresponding input components (number_input/selectbox) for the above features in the 'Heart Disease Risk Prediction' module",
        "predict_btn": "📊 Start Prediction",
        "pred_result": "Prediction Result",
        "risk_pos": "⚠️ Prediction Result: Heart Disease Risk Exists",
        "risk_neg": "✅ Prediction Result: No Heart Disease Risk",
        "risk_prob": "Risk Probability",
        "model_desc": "📋 Model Description",
        "model_type": "Model Type: Logistic Regression",
        "test_acc": "Test Set Accuracy",
        "medical_tip": "Note: This prediction is for data analysis demonstration only, not a medical diagnosis!",
        "model_eval": "📈 Model Performance Evaluation Report",
        "core_metrics": "Core Metrics",
        "class_metrics": "Classification Metrics Details",
        "conf_matrix": "Confusion Matrix",
        "true_label": "True",
        "pred_label": "Pred",
        "model_note": "Model Notes",
        "train_data": "Training Data Ratio: 80% (Test Set 20%)",
        "process_strategy": "Processing Strategy: LabelEncoder for Categorical Features",
        "scenario": "Application: Preliminary Heart Disease Risk Screening Demo",
        "usage_tip": "💡 Usage Tips:",
        "path_tip": "1. Ensure the dataset path is correct",
        "target_tip": "2. Target column name must match 'target_col' in the code",
        "input_tip": "3. Add input components for all features in the prediction module",
        "tool_tip": "4. This App is for data analysis only, not a medical tool",
        "distribution_by": "Distribution of {feature} by {target}"  # 新增英文标题模板
    }
}

# 页面配置 + 语言选择
st.set_page_config(
    page_title="Heart Disease Prediction App",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.sidebar.header("🌐 Language / 语言")
lang = st.sidebar.radio("Select Language", ["中文", "English"], index=0)
lang_code = "zh" if lang == "中文" else "en"
text = LANG_DICT[lang_code]

# ---------------------- 数据加载与预处理 ----------------------
@st.cache_data
def load_data(lang_code):
    dataset_path = "synthetic_heart_disease_dataset.csv"  # 替换为你的数据集路径
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        err_msg = "未找到数据集！" if lang_code == "zh" else "Dataset not found!"
        st.error(f"{err_msg} Please ensure '{dataset_path}' is in the same folder.")
        st.stop()
    
    target_col = "Heart_Disease"  # 确保与你的数据集目标列一致
    
    # 修复目标变量双语映射（关键修复点）
    target_col_bilingual = {
        "zh": "心脏病状态",
        "en": "Heart Disease"
    }[lang_code]
    
    if target_col not in df.columns:
        st.error(f"数据集缺少目标列 '{target_col}'！" if lang_code == "zh" else f"Dataset missing target column '{target_col}'!")
        st.stop()
    
    return df, target_col, target_col_bilingual

# 关键修复：将 lang_code 作为参数传递给 load_data
df, target_col, target_col_bilingual = load_data(lang_code)

def preprocess_data(df, target_col, lang_code):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # 识别分类特征
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    yes_no_cols = []
    for col in X.columns:
        if col not in cat_cols:
            unique_vals = set(X[col].dropna().unique())
            if unique_vals.issubset({'Yes', 'No', 'yes', 'no', 'Y', 'N', 'y', 'n'}):
                yes_no_cols.append(col)
    cat_cols = list(set(cat_cols + yes_no_cols))
    num_cols = [col for col in X.columns if col not in cat_cols]
    
    # 特征名双语映射（完善映射关系）
    feat_name_bilingual = {
        "Age": {"zh": "年龄", "en": "Age"},
        "Gender": {"zh": "性别", "en": "Gender"},
        "Weight": {"zh": "体重(kg)", "en": "Weight(kg)"},
        "Height": {"zh": "身高(cm)", "en": "Height(cm)"},
        "BMI": {"zh": "BMI", "en": "BMI"},
        "Hypertension": {"zh": "高血压", "en": "Hypertension"},
        "Diabetes": {"zh": "糖尿病", "en": "Diabetes"},
        "Hyperlipidemia": {"zh": "高血脂", "en": "Hyperlipidemia"},
        "Family_History": {"zh": "家族病史", "en": "Family History"},
        "Previous_Heart_Attack": {"zh": "既往心脏病史", "en": "Previous Heart Attack"},
        "Systolic_BP": {"zh": "收缩压(mmHg)", "en": "Systolic BP(mmHg)"},
        "Diastolic_BP": {"zh": "舒张压(mmHg)", "en": "Diastolic BP(mmHg)"},
        "Heart_Rate": {"zh": "心率(bpm)", "en": "Heart Rate(bpm)"},
        "Blood_Sugar_Fasting": {"zh": "空腹血糖(mg/dL)", "en": "Fasting Blood Sugar(mg/dL)"},
        "Cholesterol_Total": {"zh": "总胆固醇(mg/dL)", "en": "Total Cholesterol(mg/dL)"},
        "Smoking": {"zh": "吸烟状态", "en": "Smoking Status"},
        "Alcohol_Intake": {"zh": "饮酒量", "en": "Alcohol Intake"},
        "Physical_Activity": {"zh": "体力活动", "en": "Physical Activity"},
        "Diet": {"zh": "饮食类型", "en": "Diet Type"},
        "Stress_Level": {"zh": "压力水平", "en": "Stress Level"}
    }
    
    # 补充未匹配的特征名
    for col in X.columns:
        if col not in feat_name_bilingual:
            feat_name_bilingual[col] = {"zh": col, "en": col}
    
    # 保存分类特征取值
    cat_feat_values = {}
    for col in cat_cols:
        unique_vals = df[col].unique()
        unique_vals = [str(val) if pd.isna(val) else val for val in unique_vals]
        cat_feat_values[col] = unique_vals
    
    # 编码分类特征
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_col = df[col].fillna("nan")
        le.fit(df_col)
        le_dict[col] = le
    
    # 编码目标变量
    if y.dtype == "object" or y.dtype == "category":
        le_y = LabelEncoder()
        y = le_y.fit_transform(y)
        le_dict["target"] = le_y
    
    return X, y, cat_cols, num_cols, cat_feat_values, le_dict, feat_name_bilingual

# 数据预处理
X, y, cat_cols, num_cols, cat_feat_values, le_dict, feat_name_bilingual = preprocess_data(df, target_col, lang_code)

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------- 数据编码函数 ----------------------
def encode_data(data, cat_cols, num_cols, le_dict):
    data_encoded = data.copy()
    for col in cat_cols:
        if col in le_dict:
            data_encoded[col] = data_encoded[col].fillna("nan")
            known_classes = set(le_dict[col].classes_)
            data_encoded[col] = data_encoded[col].apply(
                lambda x: x if x in known_classes else "unknown"
            )
            if "unknown" not in known_classes:
                le_dict[col].classes_ = np.append(le_dict[col].classes_, "unknown")
            data_encoded[col] = le_dict[col].transform(data_encoded[col]).astype(int)
    
    for col in num_cols:
        if col in data_encoded.columns:
            data_encoded[col] = pd.to_numeric(data_encoded[col], errors='coerce')
            if col in X_train.columns:
                mean_val = X_train[col].astype(float).mean()
                data_encoded[col] = data_encoded[col].fillna(mean_val)
            data_encoded[col] = data_encoded[col].astype(float)
    
    for col in data_encoded.columns:
        if data_encoded[col].dtype == 'object':
            le = LabelEncoder()
            data_encoded[col] = data_encoded[col].fillna("nan")
            le.fit(data_encoded[col])
            data_encoded[col] = le.transform(data_encoded[col]).astype(int)
    
    return data_encoded

# 对训练集和测试集编码
X_train_encoded = encode_data(X_train, cat_cols, num_cols, le_dict)
X_test_encoded = encode_data(X_test, cat_cols, num_cols, le_dict)

# ---------------------- 模型训练 ----------------------
@st.cache_resource
def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    return model

model = train_model(X_train_encoded, y_train)
y_pred = model.predict(X_test_encoded)
acc = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, output_dict=True)

# ---------------------- 界面模块 ----------------------
st.title(text["app_title"])
st.sidebar.header(text["sidebar_nav"])
option = st.sidebar.selectbox(text["modules"][0], text["modules"])

# 1. 数据概览
if option == text["modules"][0]:
    st.header(text["data_overview"])
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.subheader(text["data_head"])
        df_display = df.rename(columns={k: v[lang_code] for k, v in feat_name_bilingual.items()})
        st.dataframe(df_display.head(10), use_container_width=True, height=300)
        
        st.subheader(text["data_stats"])
        st.dataframe(df.describe(include="all").round(2), use_container_width=True, height=300)
    
    with col2:
        st.subheader(text["data_info"])
        buf = io.StringIO()
        df.info(buf=buf)
        data_info = buf.getvalue()
        st.text(data_info)
        
        st.subheader(text["target_dist"])
        target_count = df[target_col].value_counts()
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # 字体配置
        if lang_code == "zh":
            plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
            plt.rcParams["axes.unicode_minus"] = False
        
        sns.countplot(x=target_col, data=df, ax=ax, palette="viridis", edgecolor="black")
        ax.set_title(f"{text['target_dist']} - {target_col_bilingual}", fontsize=12)
        ax.set_xlabel(target_col_bilingual)
        ax.set_ylabel("数量" if lang_code == "zh" else "Count")
        for i, v in enumerate(target_count.values):
            ax.text(i, v + 5, str(v), ha="center", va="bottom", fontsize=10)
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)

# 2. 探索性分析（EDA）
elif option == text["modules"][1]:
    st.header(text["eda_title"])
    eda_type = st.selectbox("Select EDA Type", text["eda_types"])
    
    # 字体配置
    if lang_code == "zh":
        plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
        plt.rcParams["axes.unicode_minus"] = False
    else:
        plt.rcParams["font.family"] = ["Arial", "Helvetica", "sans-serif"]
    
    # 1. 单变量分布
    if eda_type == text["eda_types"][0]:
        st.subheader(text["select_feat"])
        feat_type = st.radio("", [text["cat_feat"], text["num_feat"]])
        
        if feat_type == text["cat_feat"] and cat_cols:
            selected_feat = st.selectbox(text["cat_feat"], cat_cols)
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.countplot(x=selected_feat, data=df, ax=ax, palette="Set2", edgecolor="black")
            feat_name = feat_name_bilingual[selected_feat][lang_code]
            ax.set_title(f"{feat_name} - {text['dist_compare']}", fontsize=12)
            ax.set_xlabel(feat_name, fontsize=10)
            ax.set_ylabel("数量" if lang_code == "zh" else "Count", fontsize=10)
            plt.xticks(rotation=45, fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
        
        elif feat_type == text["num_feat"] and num_cols:
            selected_feat = st.selectbox(text["num_feat"], num_cols)
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(df[selected_feat], kde=True, ax=ax, color="skyblue", edgecolor="black")
            feat_name = feat_name_bilingual[selected_feat][lang_code]
            ax.set_title(f"{feat_name} - {text['dist_compare']}", fontsize=12)
            ax.set_xlabel(feat_name, fontsize=10)
            ax.set_ylabel("密度" if lang_code == "zh" else "Density", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)
    
    # 2. 特征相关性热力图
    elif eda_type == text["eda_types"][1]:
        st.subheader(text["corr_heatmap"])
        if num_cols:
            corr_df = df[num_cols + [target_col]].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                corr_df, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax,
                annot_kws={"fontsize": 9}
            )
            ax.set_title(text["corr_heatmap"], fontsize=14)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("当前数据集无数值特征，无法生成相关性热力图" if lang_code == "zh" else "No numerical features in the dataset, cannot generate correlation heatmap")
    
    # 3. 特征与目标变量关联（核心修复部分）
    elif eda_type == text["eda_types"][2]:
        st.subheader(text["target_corr"])
        selected_feat = st.selectbox(text["select_feat"], X.columns)
        feat_name = feat_name_bilingual[selected_feat][lang_code]
        target_name = target_col_bilingual  # 使用正确的目标变量名称
        
        fig, ax = plt.subplots(figsize=(8, 4))
        if selected_feat in cat_cols:
            sns.countplot(x=selected_feat, hue=target_col, data=df, ax=ax, palette="Set1", edgecolor="black")
            # 使用语言模板生成标题（关键修复）
            ax.set_title(text["distribution_by"].format(feature=feat_name, target=target_name), fontsize=12)
            ax.set_xlabel(feat_name, fontsize=10)
            ax.set_ylabel("数量" if lang_code == "zh" else "Count", fontsize=10)
            ax.legend(title=target_name, labels=["无" if lang_code == "zh" else "No", "有" if lang_code == "zh" else "Yes"])
        else:
            sns.boxplot(x=target_col, y=selected_feat, data=df, ax=ax, palette="Set1", medianprops={"color": "black"})
            # 使用语言模板生成标题（关键修复）
            ax.set_title(text["distribution_by"].format(feature=feat_name, target=target_name), fontsize=12)
            ax.set_xlabel(target_name, fontsize=10)
            ax.set_ylabel(feat_name, fontsize=10)
            ax.set_xticklabels(["无" if lang_code == "zh" else "No", "有" if lang_code == "zh" else "Yes"])
        
        plt.xticks(rotation=45, fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
    
    # 4. 散点图分析（修复颜色显示问题）
    elif eda_type == text["eda_types"][3]:
        st.subheader(text["scatter_title"])
        if len(num_cols) >= 2:
            feat1 = st.selectbox("选择第一个特征" if lang_code == "zh" else "Select First Feature", num_cols)
            feat2 = st.selectbox("选择第二个特征" if lang_code == "zh" else "Select Second Feature", num_cols, index=1)
            feat1_name = feat_name_bilingual[feat1][lang_code]
            feat2_name = feat_name_bilingual[feat2][lang_code]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 修复：使用hue参数简化颜色映射
            scatter = sns.scatterplot(
                x=feat1, 
                y=feat2, 
                hue=target_col, 
                data=df, 
                ax=ax,
                palette={0: "blue", 1: "red"},
                s=60,
                alpha=0.7
            )
            
            # 添加回归线（使用全部数据）
            sns.regplot(
                x=feat1, 
                y=feat2, 
                data=df, 
                ax=ax, 
                scatter=False, 
                color="black", 
                line_kws={"linestyle": "--", "alpha": 0.7}
            )
            
            # 设置标题和标签
            if lang_code == "zh":
                title = f"{feat1_name} 与 {feat2_name} 的散点图（按 {target_col_bilingual} 分组）"
                legend_labels = ["无心脏病", "有心脏病"]
            else:
                title = f"Scatter Plot of {feat1_name} vs {feat2_name} (Grouped by {target_col_bilingual})"
                legend_labels = ["No Heart Disease", "Heart Disease"]
            
            ax.set_title(title, fontsize=12)
            ax.set_xlabel(feat1_name, fontsize=10)
            ax.set_ylabel(feat2_name, fontsize=10)
            
            # 修复图例标签
            handles, _ = scatter.get_legend_handles_labels()
            ax.legend(handles, legend_labels, title=target_col_bilingual)
            
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("当前数据集的数值特征不足2个，无法生成散点图" if lang_code == "zh" else "Not enough numerical features (need at least 2) to generate scatter plot")
    
    # 5. 小提琴图分析
    elif eda_type == text["eda_types"][4]:
        st.subheader(text["violin_title"])
        if num_cols:
            selected_feat = st.selectbox(text["select_feat"], num_cols)
            feat_name = feat_name_bilingual[selected_feat][lang_code]
            
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.violinplot(
                x=target_col, y=selected_feat, data=df, ax=ax,
                palette="Set1", inner="quartile", linewidth=1
            )
            ax.set_title(f"{feat_name} 按 {target_col_bilingual} 的分布密度" if lang_code == "zh" else f"Distribution Density of {feat_name} by {target_col_bilingual}", fontsize=12)
            ax.set_xlabel(target_col_bilingual, fontsize=10)
            ax.set_ylabel(feat_name, fontsize=10)
            ax.set_xticklabels(["无" if lang_code == "zh" else "No", "有" if lang_code == "zh" else "Yes"])
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("当前数据集无数值特征，无法生成小提琴图" if lang_code == "zh" else "No numerical features in the dataset, cannot generate violin plot")

# 3. 心脏病风险预测
elif option == text["modules"][2]:
    st.header(text["predict_title"])
    st.subheader(text["input_info"])
    
    input_data = {}
    
    # 数值特征输入
    num_feat_input = {
        "Age": {"label": text["age"], "min": 0.0, "max": 120.0, "step": 1.0},
        "Height": {"label": text["height"], "min": 50.0, "max": 250.0, "step": 1.0},
        "Weight": {"label": text["weight"], "min": 20.0, "max": 200.0, "step": 0.1},
        "Systolic_BP": {"label": text["systolic_bp"], "min": 60.0, "max": 250.0, "step": 1.0},
        "Diastolic_BP": {"label": text["diastolic_bp"], "min": 40.0, "max": 150.0, "step": 1.0},
        "Heart_Rate": {"label": text["heart_rate"], "min": 30.0, "max": 200.0, "step": 1.0},
        "Blood_Sugar_Fasting": {"label": text["blood_sugar"], "min": 40.0, "max": 400.0, "step": 1.0},
        "Cholesterol_Total": {"label": text["cholesterol_total"], "min": 100.0, "max": 500.0, "step": 1.0}
    }
    for feat, params in num_feat_input.items():
        if feat in X.columns:
            input_data[feat] = st.number_input(
                params["label"],
                min_value=params["min"],
                max_value=params["max"],
                step=params["step"],
                key=feat
            )
    
    # 核心分类特征输入
    core_cat_feats = {
        "Hypertension": {"label": text["hypertension"]},
        "Diabetes": {"label": text["diabetes"]},
        "Hyperlipidemia": {"label": text["hyperlipidemia"]},
        "Family_History": {"label": text["family_history"]},
        "Previous_Heart_Attack": {"label": text["prev_heart_attack"]},
        "Gender": {"label": "性别" if lang_code == "zh" else "Gender"}
    }
    for feat, params in core_cat_feats.items():
        if feat in X.columns:
            options = cat_feat_values.get(feat, ["Yes", "No"])
            input_data[feat] = st.selectbox(params["label"], options=options, key=feat)
    
    # 其他分类特征输入
    other_cat_feats = {
        "Smoking": {"label": text["smoking"]},
        "Alcohol_Intake": {"label": text["alcohol"]},
        "Physical_Activity": {"label": text["physical_activity"]},
        "Diet": {"label": text["diet"]},
        "Stress_Level": {"label": text["stress_level"]}
    }
    for feat, params in other_cat_feats.items():
        if feat in X.columns:
            options = cat_feat_values.get(feat, ["Low", "Medium", "High"])
            input_data[feat] = st.selectbox(params["label"], options=options, key=feat)
    
    # 自动计算BMI
    if "Height" in input_data and "Weight" in input_data and input_data["Height"] > 0:
        bmi = input_data["Weight"] / ((input_data["Height"] / 100) ** 2)
        st.number_input(text["bmi"], value=round(bmi, 2), disabled=True)
        if "BMI" in X.columns:
            input_data["BMI"] = bmi
    
    # 检查缺失特征
    missing_feats = [feat for feat in X.columns if feat not in input_data]
    if missing_feats:
        st.warning(f"{text['missing_feat']} {', '.join(missing_feats)}")
        st.info(text["feat_tip"])
    else:
        # 预测按钮
        if st.button(text["predict_btn"]):
            input_df = pd.DataFrame([input_data])
            input_encoded = encode_data(input_df, cat_cols, num_cols, le_dict)
            input_encoded = input_encoded[X.columns]
            input_encoded = input_encoded.astype(float)
            
            pred = model.predict(input_encoded)[0]
            pred_prob = model.predict_proba(input_encoded)[0][1]
            
            st.subheader(text["pred_result"])
            if pred == 1:
                st.error(text["risk_pos"])
            else:
                st.success(text["risk_neg"])
            st.metric(text["risk_prob"], f"{pred_prob:.2%}")
            
            st.info(text["medical_tip"])
            st.subheader(text["model_desc"])
            st.write(text["model_type"])
            st.write(f"{text['test_acc']}: {acc:.2%}")

# 4. 模型性能评估
elif option == text["modules"][3]:
    st.header(text["model_eval"])
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.subheader(text["core_metrics"])
        metrics_df = pd.DataFrame({
            "Metric": [text["test_acc"], "Precision", "Recall", "F1-Score"],
            "Value": [
                acc,
                class_report["1"]["precision"],
                class_report["1"]["recall"],
                class_report["1"]["f1-score"]
            ]
        }).round(4)
        st.dataframe(metrics_df, use_container_width=True)
        
        st.subheader(text["conf_matrix"])
        fig, ax = plt.subplots(figsize=(6, 4))
        
        if lang_code == "zh":
            plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
        
        sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel(f"{text['pred_label']}")
        ax.set_ylabel(f"{text['true_label']}")
        ax.set_title(text["conf_matrix"])
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.subheader(text["class_metrics"])
        class_df = pd.DataFrame(class_report).T.round(4)
        st.dataframe(class_df, use_container_width=True)
        
        st.subheader(text["model_note"])
        st.write(text["train_data"])
        st.write(text["process_strategy"])
        st.write(text["scenario"])

# 使用提示
st.sidebar.markdown("---")
st.sidebar.subheader(text["usage_tip"])
st.sidebar.write(text["path_tip"])
st.sidebar.write(text["target_tip"])
st.sidebar.write(text["input_tip"])
st.sidebar.write(text["tool_tip"])