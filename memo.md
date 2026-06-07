## はじめに
本プロジェクトでは機械学習の有名な題材である「タイタニック号の乗客データ」を活用し、ユーザーが自分の情報を入力して生存確率を診断できるミニアプリを制作します。  
機械学習のモデル作成の復習と、実際にモデルの活用をする流れを学んでいきます。  

**開発環境**
* Docker
* Streamlit：PythonだけでWebアプリの画面が作れるフレームワークです。
* Scikit-learn：機械学習モデル（生存予測のアルゴリズム）の作成に使用します。
* Pandas：タイタニックのデータを整理・加工するために使用します。

## 1.モデルの作成
### 1-1.使用データについて
今回はデータ分析のプラットフォームであるKaggleが提供している「Titanic: Machine Learning from Disaster」のデータセットを使用しました。  
実装にあたっては、学習用として広く利用されているData Science Dojoの公開リポジトリからデータを取得します。  

**【データ項目一覧（データ辞書）】**  
カラム名|意味|補足（値の内容など）  
:---|:---|:--- 
PassengerId|乗客ID|各乗客に割り振られた一連番号  
Survived|生存結果|0 = 死亡、1 = 生存（今回の予測対象）  
Pclass|客室ランク|1 = 1等（富裕層）、2 = 2等、3 = 3等（一般層）  
Name|氏名|乗客の名前  
Sex|性別|male（男性）、female（女性）  
Age|年齢|0歳〜80歳（一部欠損あり）  
SibSp|同乗の兄弟・配偶者数|タイタニックに乗っていた兄弟、姉妹、夫、妻の数  
Parch|同乗の親子数|タイタニックに乗っていた親、子供の数  
Ticket|チケット番号|チケットの識別番号  
Fare|運賃|乗客が支払った運賃  
Cabin|客室番号|宿泊していた客室の番号（欠損が非常に多い）  
Embarked|出港地|C = Cherbourg、Q = Queenstown、S = Southampton  

### 1-2.前処理と特徴量の選定
機械学習モデルが数式として処理できるように以下の前処理を行います。
* 欠測値処理：Age（年齢）の欠測値を中央値で補完
* カテゴリ変数の数値化：Sex（性別）を0（男性）と1（女性）に変換

**相関分析の結果**  
相関関係を確認したところ、Survived と最も強い正の相関があったのは Sex（女性であること）であり、最も強い負の相関があったのは Pclass（客室ランクが下がる＝数字が大きくなること）でした。  
この結果から、アプリへの入力項目としてこれらを採用する妥当性が確認できました。

```Python
import pandas as pd

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# 必要な列を選ぶ
features = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
df = df[features]

# 欠損値を中央値で埋める
df['Age'] = df['Age'].fillna(df['Age'].median())

# カテゴリ変数を数値に変換
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

print("前処理後のデータの概要:")
print(df.corr()['Survived'].sort_values(ascending=False))
```

```bash
前処理後のデータの概要:
Survived    1.000000
Sex         0.543351
Fare        0.257307
Parch       0.081629
SibSp      -0.035322
Age        -0.064910
Pclass     -0.338481
Name: Survived, dtype: float64
```

### 1-3.データの分割とモデルの学習
1. **データを分ける（Train/Test Split）**  
全部のデータで学習させてしまうとそのデータに詳しすぎて「未知のデータ」に弱くなってしまいます。（過学習）  
```Python
# 必要な列を選ぶ
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
X = df[features]
y = df['Survived']

# データの分割（学習用80％、テスト用20％）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
2. **学習させる（fit）**  
「ランダムフォレスト」などのアルゴリズムにデータを読み込ませてルールを覚えさせます。  
```Python
# ランダムフォレストモデルの作成
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```
3. **精度を確かめる**  
テスト用データを使って「何％正解できるか」を出します。
```Python
# テストデータで予測
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"モデルの精度: {accuracy:.2f}")
```
```bash
モデルの精度: 0.80
```
4. **モデルの保存**
```Python
# モデルの保存
with open('models/titanic_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("モデルが 'models/titanic_model.pkl' に保存されました。")
```
```bash
モデルが 'models/titanic_model.pkl' に保存されました。
```

## 2.アプリの作成（streamlitを仕様）
### 2-1. 画面遷移の制御（セッション状態の管理）
Streamlitはボタン押下などのイベント毎にスクリプトが上から再実行される仕様のため、通常の変数では入力データや画面状態がリセットされます。  

これを診断アプリらしい1問1答形式にするため、st.session_state（セッション状態）に状態を保持させるという方法を使いました。
（コード部分は実際のコードを説明用に変えて記載しています）

**【管理する状態（変数）】**
* st.session_state.question_step : 現在の質問ステップ（0：スタート画面 〜 5：結果画面）
* 各入力値（user_sex, user_age 等） : ユーザーが入力した回答データ
* st.session_state.animated : ロード画面の実行済みフラグ（再実行時の重複防止用）

```python
# セッション状態の初期化
if "question_step" not in st.session_state:
    st.session_state.question_step = 0

# ステップ数に応じた画面の出し分け（例：スタート画面）
if st.session_state.question_step == 0:
    st.header("タイタニック号であなたは生き残れる？")
    if st.button("乗船手続き（データの入力）", type="primary"):
        st.session_state.question_step = 1  # 次のステップへ
        st.rerun()  # 画面を即時再描画
```

### 2-2. 各質問画面（UI）の実装
バックエンドの機械学習モデルが処理できる形式（数値）に変換して保存するだけでなく、  
わかりにくいチケット情報はラジオボタン選択時に詳細が出るようにしてみました。
```python
# 第3問（乗船プラン）での動的表示の例
ticket_type = st.radio("プランを選んでください", ["1等客室", "2等客室", "3等客室"])

if "1等" in ticket_type:
    st.info("**👑 1等客室詳細**\n* 現代価値: 約100万〜1,000万円以上")
elif "2等" in ticket_type:
    st.success("**🎩 2等客室詳細**\n* 現代価値: 約25万〜30万円相当")
```
### 2-3. 結果画面のロード演出と残像対策
最終画面（ステップ5）では、予測処理の前にローカル画像（titanic.jpg）を width="container" で画面横幅いっぱいに表示し、進捗バーによるロード演出を行うことで診断のワクワクが出るようにしました。
* 残像現象：単に time.sleep(2) で処理を停止させると、Streamlitの描画仕様により直前の画面（第4問）が薄く残ってしまいました。対策として、time.sleep(0.02) の極小停止を100回繰り返すループで st.progress を更新し、画面の強制リフレッシュを走らせることで残像を解消。

* リロード時の演出暴走防止：結果表示後に「もう一度診断する」ボタンで初期画面へ戻る際、再実行の影響でロード演出が再起動するのを防ぐため、animated フラグで2回目以降のロード処理をスキップさせるようにしました。

```python
# 残像対策を施したロード演出部分
if not st.session_state.get("animated", False):
    progress_bar = st.progress(0)
    for percent in range(100):
        time.sleep(0.02)  # 0.02秒ごとに画面を更新して残像を消去
        progress_bar.progress(percent + 1)
    progress_bar.empty()
    st.session_state.animated = True  # 実行済みフラグを立てる
```




