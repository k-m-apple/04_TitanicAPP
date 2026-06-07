import pandas as pd

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# 必要な列を選ぶ
features = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
df = df[features]

# 欠損値を平均値で埋める
df['Age'] = df['Age'].fillna(df['Age'].median())

# カテゴリ変数を数値に変換
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

print("前処理後のデータの概要:")
print(df.corr()['Survived'].sort_values(ascending=False))