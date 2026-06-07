import pandas as pd

# インターネットの公式データから読み込み
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print("データの概要:")
print(df.head())

print("\n欠測値の確認:")
print(df.isnull().sum())

print("\n基本統計量:")
print(df.describe())
