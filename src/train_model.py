import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# データの読み込み
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
# 欠損値を平均値で埋める
df['Age'] = df['Age'].fillna(df['Age'].median())

# カテゴリ変数を数値に変換
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# 必要な列を選ぶ
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
X = df[features]
y = df['Survived']

# データの分割（学習用80％、テスト用20％）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ランダムフォレストモデルの作成
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# テストデータで予測
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"モデルの精度: {accuracy:.2f}")

# モデルの保存
with open('models/titanic_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("モデルが 'models/titanic_model.pkl' に保存されました。")