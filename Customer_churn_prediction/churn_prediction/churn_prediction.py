import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score

# 中文乱码处理
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
file_path = '../data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
data = pd.read_csv(file_path, engine='python', encoding='utf-8')

# data.info()

# 数据预处理
# 删除缺失数据
data = data.dropna()

# data.info()

# print(data)

# 数据类型转换, 将数据类型转换为数值类型
data['gender'] = data['gender'].map({'Male': 0, 'Female': 1})
data['Partner'] = data['Partner'].map({'No': 0, 'Yes': 1})
data['Dependents'] = data['Dependents'].map({'No': 0, 'Yes': 1})
data['PhoneService'] = data['PhoneService'].map({'No': 0, 'Yes': 1})
data['MultipleLines'] = data['MultipleLines'].map({'No': 0, 'Yes': 1, 'No phone service': 2})
data['InternetService'] = data['InternetService'].map({'DSL': 0, 'Fiber optic': 1, 'No': 2})
data['OnlineSecurity'] = data['OnlineSecurity'].map({'No': 0, 'Yes': 1, 'No internet service': 2})
data['OnlineBackup'] = data['OnlineBackup'].map({'No': 0, 'Yes': 1, 'No internet service': 2})
data['DeviceProtection'] = data['DeviceProtection'].map({'No': 0, 'Yes': 1, 'No internet service': 2})
data['TechSupport'] = data['TechSupport'].map({'No': 0, 'Yes': 1, 'No internet service': 2})
data['StreamingTV'] = data['StreamingTV'].map({'No': 0, 'Yes': 1, 'No internet service': 2})
data['StreamingMovies'] = data['StreamingMovies'].map({'No': 0, 'Yes': 1, 'No internet service': 2})
data['Contract'] = data['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
data['PaperlessBilling'] = data['PaperlessBilling'].map({'No': 0, 'Yes': 1})
data['PaymentMethod'] = data['PaymentMethod'].map(
    {'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3})
data['Churn'] = data['Churn'].map({'No': 0, 'Yes': 1})

# print(data)

# 皮尔森系数进行相关性分析
corr = data.corr(method='pearson')
corr_with_churn = corr['Churn'].sort_values(ascending=False)
# print(corr_with_churn)

# # 绘制热度图
# axis_corr = sns.heatmap(corr, vmin=-1, vmax=1, center=0,
#                         cmap=sns.diverging_palette(50, 500, n=500), square=True)
# plt.show()

# 选择相关系数大于0.3的特征进行后续操作
selected_features = corr_with_churn[corr_with_churn.abs() > 0.3].index.tolist()
# print(selected_features)

# # 查看选择特征的取值分布
# for feature in selected_features:
#     plt.figure()
#     data[feature].value_counts().plot(kind='bar')
#     plt.title(f"{feature} Distribution")
#     plt.xlabel(feature)
#     plt.ylabel("Count")
#     plt.show()

# 划分数据集
X = data[selected_features].drop('Churn', axis=1)
y = data['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# print("训练集大小: " + str(len(X_train)))
# print("测试集大小: " + str(len(X_test)))

# 模型训练
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# 模型评估
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Recall:", recall)
print("F1-score:", f1)
