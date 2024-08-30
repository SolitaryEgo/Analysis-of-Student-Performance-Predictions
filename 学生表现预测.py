import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings('ignore')
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

df = pd.read_csv('./student_performance.csv')
print(df.head())

df.drop('StudentID', axis=1, inplace=True)

f, ax = plt.subplots(figsize=(7, 5))
sns.countplot(x='Gender', edgecolor='black', linewidth=1, data=df)
ax.set_xlabel('性别', size=13)
ax.set_ylabel('计数', size=13)
ax.spines[['top', 'right']].set_visible(False)
ax.set_title('性别分布', size=15, weight='bold')
plt.show()

f, ax = plt.subplots(figsize=(7, 5))
sns.countplot(x='ParentalSupport', edgecolor='black', linewidth=1, data=df)
ax.set_xlabel('父母支持', size=13)
ax.set_ylabel('计数', size=13)
ax.spines[['top', 'right']].set_visible(False)
ax.set_title('父母支持分布', size=15, weight='bold')
plt.show()

f, ax = plt.subplots(1, 2, figsize=(14, 5))
sns.kdeplot(data=df, x='AttendanceRate', hue='Gender', ax=ax[0])
sns.kdeplot(data=df, x='AttendanceRate', hue='ParentalSupport', ax=ax[1])
f.suptitle('出勤率分布', weight='bold', size=15)
plt.show()

f, ax = plt.subplots(1, 2, figsize=(14, 5))
sns.kdeplot(data=df, x='StudyHoursPerWeek', hue='Gender', ax=ax[0])
sns.kdeplot(data=df, x='StudyHoursPerWeek', hue='ParentalSupport', ax=ax[1])
f.suptitle('每周学习时间分布', weight='bold', size=15)
plt.show()

f, ax = plt.subplots(1, 2, figsize=(14, 5))
sns.kdeplot(data=df, x='PreviousGrade', hue='Gender', ax=ax[0])
sns.kdeplot(data=df, x='PreviousGrade', hue='ParentalSupport', ax=ax[1])
f.suptitle('上一次成绩分布', weight='bold', size=15)
plt.show()

f, ax = plt.subplots(1, 2, figsize=(14, 5))
sns.kdeplot(data=df, x='ExtracurricularActivities', hue='Gender', ax=ax[0])
sns.kdeplot(data=df, x='ExtracurricularActivities', hue='ParentalSupport', ax=ax[1])
f.suptitle('课外活动分布', weight='bold', size=15)
plt.show()

f, ax = plt.subplots(1, 2, figsize=(14, 5))
sns.kdeplot(data=df, x='FinalGrade', hue='Gender', ax=ax[0])
sns.kdeplot(data=df, x='FinalGrade', hue='ParentalSupport', ax=ax[1])
f.suptitle("期末成绩分布", weight='bold', size=15)
plt.show()

plt.figure(figsize=(10, 7))
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.show()

y = df['FinalGrade']
y_pred = df['PreviousGrade'] + 2

f, ax = plt.subplots(figsize=(7, 5))
sns.scatterplot(data=df, x=y, y=y_pred, color='blue')
sns.scatterplot(data=df, x=y, y=y, color='red')
ax.set_xlabel('y_true', size=13)
ax.set_ylabel('y_pred', size=13)
ax.spines[['top', 'right']].set_visible(False)
ax.set_title('预期值与实际值之比', size=15, weight='bold')
plt.show()

print('MSE = ', mean_absolute_error(y, y_pred))
print('RMSE = ', np.sqrt(mean_squared_error(y, y_pred)))
print('R2 = ', r2_score(y, y_pred))

label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['ParentalSupport'] = label_encoder.fit_transform(df['ParentalSupport'])
X = df.drop(['Name', 'FinalGrade'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    'Linear Regression': LinearRegression(),
    'KNN': KNeighborsRegressor(),
    'DecisionTree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'{name}: MSE = {mse:.2f}, R^2 = {r2:.2f}')


