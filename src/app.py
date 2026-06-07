import streamlit as st
import pandas as pd
import pickle

# 1. 保存したモデルの読み込み
@st.cache_resource
def load_model():
    with open('models/titanic_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# 2.画面切り替えのセッションステート
# ①初期化
if "question_step" not in st.session_state:
    st.session_state.question_step = 0

if st.session_state.question_step == 0:
    st.header("タイタニック号であなたは生き残れる？")
    
    st.image("src/Title.jpg", width="stretch")  # タイトル画像を中央に配置
    st.write("時は1912年。あなたはイギリスの港から、夢の豪華客船タイタニック号に乗ろうとしています。")
    st.write("手元に届いた【乗船チケット】の内容を選んで、あなたの運命を占ってみましょう。")

    if st.button("乗船手続き（データの入力）", type="primary"):
        st.session_state.question_step = 1
        st.rerun()  # 画面をリフレッシュして次のステップへ

elif st.session_state.question_step == 1:
    st.subheader("Q1:あなたの性別は？")
    sex_label = st.selectbox("選択してください", ["女性", "男性"])
    
    if st.button("次へ", type="primary"):
        st.session_state.user_sex = 1 if "女性" in sex_label else 0

        st.session_state.question_step = 2
        st.rerun()

elif st.session_state.question_step == 2:
    st.subheader("Q2:おいくつですか？")
    age = st.number_input(
        "年齢を入力してください",
        min_value=0,
        max_value=80,
        value=25,
        step=1
    )

    if st.button("次へ", type="primary"):
        st.session_state.user_age = age

        st.session_state.question_step = 3
        st.rerun()

elif st.session_state.question_step == 3:
    st.subheader("Q3:どの乗船プランのチケットですか？")
    # 客室ランク（難しい名称をツアー名に変更）
    ticket_type = st.radio(
        "どの乗船プランのチケットですか？",
        ["特等席（ファーストクラス・富裕層向けの1等客室）", 
         "一般席（スタンダード・中産階級向けの2等客室）", 
         "格安席（エコノミー・労働者向けの3等客室）"]
    )
    
    if "特等" in ticket_type:
        st.info("""
                **👑 特等席（1等客室)のチケット詳細**
                * **現代の価値**： **約100万～1000万円以上**
                * **部屋の特徴**： 船の最上層にある超高級ホテル。専用の遊歩道、温水プールなど富豪向けの豪華な部屋
                """)
    elif "一般" in ticket_type:
        st.success("""
                **🎫 一般席（2等客室)のチケット詳細**
                * **現代の価値**： **約25万～30万円**
                * **部屋の特徴**： 医者や教師などの中産階級向け。今の高級クルーズに匹敵する綺麗さ。立派な食堂や図書室、喫煙室が使用可能。
                """)
    else:
        st.warning("""
                **💺 格安席（3等客室)のチケット詳細**
                * **現代の価値**： **約10万～15万円**
                * **部屋の特徴**： 新天地アメリカを夢見る多くの労働者や移民向け。船底近くの2段ベッドの相部屋。
                """)
    st.write("")

    if st.button("次へ", type="primary"):
        
    # 選択に応じてモデル用の数字（Pclass）と平均運賃（Fare）を裏側で自動セット！
        if "特等" in ticket_type:
            st.session_state.user_pclass = 1
            st.session_state.user_fare = 84.0  # 当時の1等客室の平均的な運賃（ポンド）
        elif "一般" in ticket_type:
            st.session_state.user_pclass = 2
            st.session_state.user_fare = 20.0  # 2等客室の平均
        else:
            st.session_state.user_pclass = 3
            st.session_state.user_fare = 13.0  # 3等客室の平均

        st.session_state.question_step = 4
        st.rerun()

elif st.session_state.question_step == 4:
    st.subheader("Q4:同行者について教えてください")
    st.write("👪 一緒に旅をするご家族はいますか？")
    # SibSp と Parch を「誰と行くか」に分かりやすく分解
    has_partner = st.checkbox("配偶者（夫・妻）や、兄弟・姉妹と一緒にいく")
    has_family = st.checkbox("親、または自分の子供と一緒にいく")
    
    # 家族の合計人数を計算（演出用）
    if st.button("診断結果を見る", type="primary"):
        st.session_state.user_sibsp = 1 if has_partner else 0
        st.session_state.user_parch = 1 if has_family else 0

        st.session_state.question_step = 5
        st.rerun()

elif st.session_state.question_step == 5:
    st.empty()  # 余白を入れて演出
    st.title("タイタニック号・運命の診断結果")

    col1, col2, col3 = st.columns([1, 2, 1])  # 画像とテキストの比率を調整
    with col2:
        st.image("src/titanic.jpg")
    import time

    if "animated" not in st.session_state:
            st.session_state.animated = False
    if not st.session_state.animated:
        progress_bar = st.progress(0)  # 進捗バーの初期化
        status_text = st.empty()  # 状態テキストの初期化

        for percent in range(100):
            time.sleep(0.02)  # 演出のために少し待つ

            progress_bar.progress(percent + 1)
            status_text.write(f"運命を占っています... {percent}%")

        status_text.empty()  # 状態テキストを消す
        progress_bar.empty()  # 進捗バーを消す

        st.session_state.animated = True  # アニメーションが完了したことを記録

    input_data = pd.DataFrame([{
        'Pclass': st.session_state.user_pclass,
        'Sex': st.session_state.user_sex,
        'Age': st.session_state.user_age,
        'SibSp': st.session_state.user_sibsp,
        'Parch': st.session_state.user_parch,
        'Fare': st.session_state.user_fare
    }])

    prob = model.predict_proba(input_data)[0][1]
    result = model.predict(input_data)[0]

    if result == 1:
        st.success(f"🎉 【生存確率 {prob * 100:.1f}%】 あなたは無事に救命ボートに誘導され、生還を果たしました！")
        st.balloons() # 画面に風船を飛ばすStreamlitの楽しい演出
    else:
        st.error(f"🌊 【生存確率 {prob * 100:.1f}%】 残念ながら、あなたは船と運命を共にすることになりました…")

    st.progress(prob)  
    # --- 【工夫4】組み合わせ（掛け算）による歴史の裏話解説 ---
    st.markdown("### 📜 歴史のプロファイリング（AIの着眼点）")

    sex = st.session_state.user_sex
    age = st.session_state.user_age
    pclass = st.session_state.user_pclass
    
    if sex == 1 and pclass == 1:
        st.info("✨ **歴史の事実**: あなたは最も生存率の高かった『1等客室の女性』です。当日の夜、最優先で救命ボートへ案内されました。映画のヒロイン（ローズ）と同じ最高の条件です。")
    elif sex == 1 and pclass == 3:
        st.warning("⚠️ **歴史の事実**: 女性優先ルールのおかげでボートへの道は開かれましたが、3等客室は船の底深くにあったため、避難デッキへたどり着く迷路のような通路で明暗が分かれました。生存率は五分五分です。")
    elif sex == 0 and age <= 12:
        st.info("👦 **歴史の事実**: 当時は『女性と子供を最優先』という絶対ルールがありました。男の子であっても、幼ければ大人たちに守られ、ボートへ乗せてもらえる確率が非常に高かったです。")
    elif sex == 0 and pclass == 1:
        st.warning("🎩 **歴史の事実**: 富豪の男性です。お金はありましたが、当時の紳士たちは『女性に席を譲り、自分は船に残る』というプライドを貫いた人が多く、多くの著名な富豪が命を落としました。")
    elif sex == 0 and pclass == 3:
        st.error("⚓ **歴史の事実**: 最も過酷な運命を背負った『3等客室の一般男性』です。避難の案内が届くのが最も遅く、デッキへのドアも閉ざされていたため、生存率は1割前後という絶望的な戦いでした。")
    else:
        st.write("💡 入力された条件は、当時のタイタニック号において標準的な避難・誘導が行われた層に位置しています。")
    
    st.divider()

    if st.button("最初からやり直す", use_container_width=True):
        st.session_state.question_step = 0
        st.rerun()