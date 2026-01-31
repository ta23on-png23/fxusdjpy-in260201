import streamlit as st
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import numpy as np
import lightgbm as lgb
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz

# --- ページ設定 ---
st.set_page_config(page_title="USDJPY Range Reversal AI", layout="wide", initial_sidebar_state="collapsed")

# --- CSS ---
st.markdown("""
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .title-text { font-size: 1.8rem; font-weight: bold; color: #2c3e50; margin-bottom: 0.5rem; }
    .stButton { position: fixed; top: 20px; right: 20px; z-index: 999; }
    
    .status-card { background-color: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 5px solid #ccc; margin-bottom: 10px; }
    .status-safe { border-left-color: #00cc66; background-color: #e8f5e9; }
    .status-danger { border-left-color: #ff4b4b; background-color: #ffebee; }
    .status-neutral { border-left-color: #ff9800; background-color: #fff3e0; }
    
    .big-rate { font-size: 2.5rem; font-weight: bold; text-align: center; color: #333; }
    .sub-info { font-size: 0.9rem; color: #666; text-align: center; }
    
    .decision-box { font-size: 2rem; font-weight: 900; text-align: center; padding: 15px; border-radius: 8px; color: white; margin: 15px 0; }
    .d-wait { background-color: #95a5a6; }
    .d-buy { background-color: #9b59b6; box-shadow: 0 4px 6px rgba(0,0,0,0.1); } /* 紫 */
    .d-sell { background-color: #e67e22; box-shadow: 0 4px 6px rgba(0,0,0,0.1); } /* オレンジ */
    
    .dataframe { font-size: 0.8rem !important; }
    </style>
""", unsafe_allow_html=True)

# --- 関数: データ取得 ---
def get_data():
    ticker = "USDJPY=X"
    df = yf.download(ticker, period="7d", interval="5m", progress=False)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.copy()

# --- 関数: 特徴量作成 (レンジ・逆張り指標) ---
def create_features(df):
    df = df.copy()
    
    # ボリンジャーバンド (2σ)
    bb = df.ta.bbands(length=20, std=2)
    # %B (0以下なら下限突破、1以上なら上限突破)
    df['BB_Pb'] = (df['Close'] - bb.iloc[:, 2]) / (bb.iloc[:, 0] - bb.iloc[:, 2])
    df['BB_Width'] = (bb.iloc[:, 0] - bb.iloc[:, 2]) / bb.iloc[:, 1]
    
    # RSI (売られすぎ・買われすぎ)
    df['RSI'] = df.ta.rsi(length=14)
    
    # ADX (トレンドの強さ) -> 逆張りにはこれが低いことが必須
    adx = df.ta.adx(length=14)
    df['ADX'] = adx.iloc[:, 0]
    
    # CCI (Commodity Channel Index) - 逆張りに強いオシレーター
    df['CCI'] = df.ta.cci(length=20)
    
    return df

# --- 関数: 正解ラベル作成 (±15pips) ---
def create_target(df, pips=0.15):
    targets = []
    scan_start = max(0, len(df) - 2000)
    
    for i in range(len(df)):
        if i < scan_start:
            targets.append(np.nan)
            continue
            
        current_close = df['Close'].iloc[i]
        target_up = current_close + pips
        target_down = current_close - pips
        
        future_result = np.nan
        for j in range(i + 1, min(len(df), i + 48)):
            future_high = df['High'].iloc[j]
            future_low = df['Low'].iloc[j]
            
            # 逆張りAIを作るため、ターゲット定義は同じでも、
            # ロジック側で「下がった時に買う」「上がった時に売る」を判定する
            if future_high >= target_up and future_low > target_down:
                future_result = 1 # 上昇勝利
                break
            elif future_low <= target_down and future_high < target_up:
                future_result = 0 # 下降勝利
                break
        
        targets.append(future_result)
        
    df['Target_Buy'] = targets
    return df

# --- メイン処理 ---
jst = pytz.timezone('Asia/Tokyo')

st.markdown("<div class='title-text'>🔄 USD/JPY レンジ逆張りAI</div>", unsafe_allow_html=True)
update = st.button("市場分析・判定 🔄", type="primary")

if update or True:
    with st.spinner('レンジ・過熱感を分析中...'):
        raw_df = get_data()
        
        if raw_df is not None:
            df = create_features(raw_df)
            df = create_target(df, pips=0.15)
            
            # 学習に使う特徴量 (オシレーター重視)
            features = ['RSI', 'BB_Pb', 'BB_Width', 'ADX', 'CCI']
            
            data_ready = df.dropna(subset=features + ['Target_Buy'])
            
            # --- 厳格な学習・テスト分離 ---
            test_size = 120
            
            if len(data_ready) > test_size + 100:
                X_train = data_ready[features].iloc[:-test_size]
                y_train = data_ready['Target_Buy'].iloc[:-test_size]
                
                # LightGBMモデル
                model = lgb.LGBMClassifier(n_estimators=100, max_depth=4, random_state=42, verbose=-1)
                model.fit(X_train, y_train)
                
                # --- 現在の状況取得 ---
                target_idx = -2
                current_row = df.iloc[[target_idx]]
                current_close = current_row['Close'].item()
                current_time = current_row.index[0].replace(tzinfo=pytz.utc).astimezone(jst)
                
                # 指標値
                bb_pb = current_row['BB_Pb'].item()
                adx = current_row['ADX'].item()
                rsi = current_row['RSI'].item()
                cci = current_row['CCI'].item()
                
                # AI予測
                prob_buy = model.predict_proba(current_row[features])[0][1] * 100
                prob_sell = 100 - prob_buy
                
                # --- 🧠 判定ロジック (レンジ逆張り) ---
                
                decision = "WAIT"
                d_class = "d-wait"
                reason = "条件不一致"
                
                threshold = 70
                
                # ★フィルター: トレンドが強すぎる時(ADX>30)は逆張り禁止
                is_range_market = adx < 30
                
                if is_range_market:
                    # 買い条件: AI強気 + バンド下限割れ or 売られすぎ
                    is_oversold = (bb_pb < 0.05) or (rsi < 30) or (cci < -100)
                    
                    if prob_buy >= threshold and is_oversold:
                        decision = "BUY 狙い (逆張り)"
                        d_class = "d-buy"
                        reason = "レンジ下限到達 + 売られすぎ反発狙い"
                    elif prob_buy >= threshold and not is_oversold:
                        reason = "AIは買い予測だが、まだ下がりきっていない"
                    
                    # 売り条件: AI弱気 + バンド上限突破 or 買われすぎ
                    is_overbought = (bb_pb > 0.95) or (rsi > 70) or (cci > 100)
                    
                    if prob_sell >= threshold and is_overbought:
                        decision = "SELL 狙い (逆張り)"
                        d_class = "d-sell"
                        reason = "レンジ上限到達 + 買われすぎ反落狙い"
                    elif prob_sell >= threshold and not is_overbought:
                        reason = "AIは売り予測だが、まだ上がりきっていない"
                        
                else:
                    reason = f"トレンドが強すぎるため逆張り危険 (ADX:{adx:.1f})"

                # --- UI表示 ---
                st.markdown(f"<div class='big-rate'>{current_close:.3f} <span style='font-size:1rem; color:#888'>円</span></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='sub-info'>{current_time.strftime('%m/%d %H:%M')} 確定足 | ロジック: ボリンジャーバンド逆張り</div>", unsafe_allow_html=True)
                
                st.markdown(f"<div class='decision-box {d_class}'>{decision}</div>", unsafe_allow_html=True)
                
                # ステータス表示
                col1, col2, col3 = st.columns(3)
                
                # 相場環境
                env_text = "レンジ相場 (逆張りOK)" if is_range_market else "トレンド相場 (逆張り危険)"
                e_color = "status-safe" if is_range_market else "status-danger"
                col1.info(f"相場環境 (ADX)\n\n**{env_text}**")
                
                # バンド位置
                pos_text = "上限突破 (売り場)" if bb_pb > 1.0 else "下限突破 (買い場)" if bb_pb < 0.0 else "バンド内"
                col2.info(f"ボリンジャーバンド位置\n\n**{pos_text}** (%B: {bb_pb:.2f})")
                
                # AI
                ai_text = f"買い {prob_buy:.1f}%" if prob_buy > prob_sell else f"売り {prob_sell:.1f}%"
                col3.info(f"AI予測\n\n**{ai_text}**")

                st.success(f"💡 **判断根拠:** {reason}")
                st.markdown("---")
                
                # --- 厳格バックテスト ---
                st.subheader("📊 未知データでの実力テスト (直近10時間)")
                
                test_df = data_ready.tail(test_size).copy()
                test_probs = model.predict_proba(test_df[features])
                test_df['Prob_Buy'] = test_probs[:, 1]
                
                pips_history = [0]
                trades = []
                total_pips = 0
                
                for i in range(len(test_df)):
                    row = test_df.iloc[i]
                    p_buy = row['Prob_Buy'] * 100
                    p_sell = 100 - p_buy
                    
                    price = row['Close']
                    r = row['RSI']
                    a = row['ADX']
                    c = row['CCI']
                    pb = row['BB_Pb']
                    
                    actual = row['Target_Buy']
                    
                    trade_res = 0
                    t_type = "-"
                    
                    # 過去データシミュレーション (ADX<30のレンジ環境限定)
                    if a < 30:
                        # 買い逆張り: 売られすぎ (BB下限 or RSI低 or CCI低)
                        if p_buy >= threshold and (pb < 0.05 or r < 30 or c < -100):
                            trade_res = 15 if actual == 1 else -15
                            t_type = "BUY"
                        
                        # 売り逆張り: 買われすぎ (BB上限 or RSI高 or CCI高)
                        elif p_sell >= threshold and (pb > 0.95 or r > 70 or c > 100):
                            trade_res = 15 if actual == 0 else -15
                            t_type = "SELL"
                        
                    total_pips += trade_res
                    pips_history.append(total_pips)
                    
                    if t_type != "-":
                        trades.append({
                            "時間": row.name.strftime('%H:%M'),
                            "売買": t_type,
                            "レート": f"{price:.3f}",
                            "結果": "WIN" if trade_res > 0 else "LOSS",
                        })

                # グラフ
                color_pips = "#00cc66" if total_pips >= 0 else "#ff4b4b"
                st.markdown(f"<div style='text-align:center; font-size:1.5rem; font-weight:bold; color:{color_pips}'>期間損益: {total_pips:+} pips</div>", unsafe_allow_html=True)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=pips_history, mode='lines', line=dict(color='#2c3e50', width=3)))
                fig.update_layout(height=200, margin=dict(l=20, r=20, t=10, b=20), xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#eee'))
                st.plotly_chart(fig, use_container_width=True)
                
                if trades:
                    st.write("▼ エントリー履歴")
                    st.dataframe(pd.DataFrame(trades).iloc[::-1], hide_index=True, use_container_width=True)
                else:
                    st.caption("※ 直近10時間では、レンジ逆張り条件（過熱感あり＋トレンド弱）を満たすポイントがありませんでした。")

            else:
                st.warning("データ不足")
