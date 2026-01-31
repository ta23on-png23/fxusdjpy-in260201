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
st.set_page_config(page_title="USDJPY Hybrid AI", layout="wide", initial_sidebar_state="collapsed")

# --- CSS ---
st.markdown("""
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .title-text { font-size: 1.8rem; font-weight: bold; color: #2c3e50; margin-bottom: 0.5rem; }
    .stButton { position: fixed; top: 20px; right: 20px; z-index: 999; }
    
    .status-card { background-color: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 5px solid #ccc; margin-bottom: 10px; }
    .status-trend { border-left-color: #3498db; background-color: #eaf2f8; }
    .status-range { border-left-color: #f39c12; background-color: #fef5e7; }
    .status-safe { border-left-color: #00cc66; background-color: #e8f5e9; }
    .status-danger { border-left-color: #ff4b4b; background-color: #ffebee; }
    
    .big-rate { font-size: 2.5rem; font-weight: bold; text-align: center; color: #333; }
    .sub-info { font-size: 0.9rem; color: #666; text-align: center; }
    
    .decision-box { font-size: 2rem; font-weight: 900; text-align: center; padding: 15px; border-radius: 8px; color: white; margin: 15px 0; }
    .d-wait { background-color: #95a5a6; }
    .d-buy { background-color: #27ae60; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .d-sell { background-color: #c0392b; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    
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

# --- 関数: 特徴量作成 (ハイブリッド用) ---
def create_features(df):
    df = df.copy()
    
    # トレンド判定用
    df['SMA20'] = df.ta.sma(length=20)
    df['SMA50'] = df.ta.sma(length=50)
    df['SMA200'] = df.ta.sma(length=200)
    
    # レンジ判定用 (ボリンジャーバンド)
    bb = df.ta.bbands(length=20, std=2)
    df['BB_Pb'] = (df['Close'] - bb.iloc[:, 2]) / (bb.iloc[:, 0] - bb.iloc[:, 2])
    df['BB_Width'] = (bb.iloc[:, 0] - bb.iloc[:, 2]) / bb.iloc[:, 1]
    
    # 環境認識用
    adx = df.ta.adx(length=14)
    df['ADX'] = adx.iloc[:, 0]
    
    # ★重要: ATR (ボラティリティ)
    # これが小さい時は「15pipsも動かない」のでエントリーしない
    df['ATR'] = df.ta.atr(length=14)
    
    # オシレーター
    df['RSI'] = df.ta.rsi(length=14)
    macd = df.ta.macd(fast=12, slow=26, signal=9)
    df['MACD_Hist'] = macd.iloc[:, 2]
    
    return df

# --- 関数: 正解ラベル作成 ---
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
        for j in range(i + 1, min(len(df), i + 48)): # 4時間
            future_high = df['High'].iloc[j]
            future_low = df['Low'].iloc[j]
            
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

st.markdown("<div class='title-text'>🤖 USD/JPY ハイブリッドAI (環境認識型)</div>", unsafe_allow_html=True)
update = st.button("市場分析・判定 🔄", type="primary")

if update or True:
    with st.spinner('相場環境(トレンド/レンジ)を判定中...'):
        raw_df = get_data()
        
        if raw_df is not None:
            df = create_features(raw_df)
            df = create_target(df, pips=0.15)
            
            # 特徴量
            features = ['RSI', 'BB_Pb', 'BB_Width', 'ADX', 'ATR', 'MACD_Hist']
            data_ready = df.dropna(subset=features + ['Target_Buy', 'SMA200'])
            
            # 厳格テスト用分割
            test_size = 120
            
            if len(data_ready) > test_size + 100:
                X_train = data_ready[features].iloc[:-test_size]
                y_train = data_ready['Target_Buy'].iloc[:-test_size]
                
                # モデル学習
                model = lgb.LGBMClassifier(n_estimators=100, max_depth=4, random_state=42, verbose=-1)
                model.fit(X_train, y_train)
                
                # --- 現在の状況取得 ---
                target_idx = -2
                current_row = df.iloc[[target_idx]]
                current_close = current_row['Close'].item()
                current_time = current_row.index[0].replace(tzinfo=pytz.utc).astimezone(jst)
                
                # 指標値
                adx = current_row['ADX'].item()
                atr = current_row['ATR'].item()
                sma200 = current_row['SMA200'].item()
                sma20 = current_row['SMA20'].item()
                rsi = current_row['RSI'].item()
                bb_pb = current_row['BB_Pb'].item()
                
                # AI予測
                prob_buy = model.predict_proba(current_row[features])[0][1] * 100
                prob_sell = 100 - prob_buy
                
                # --- 🧠 ハイブリッド判定ロジック ---
                
                decision = "WAIT"
                d_class = "d-wait"
                reason = "分析中..."
                regime = "不明"
                
                threshold = 73
                
                # 1. ボラティリティチェック (ATRフィルター)
                # 5分足の平均値幅(ATR)が極端に小さい(例: 0.03円以下)と、15pips動くのに何時間もかかり不利
                is_volatile_enough = atr > 0.03
                
                if not is_volatile_enough:
                    reason = f"値動きが小さすぎるため見送り (ATR: {atr:.3f})"
                    regime = "閑散相場"
                else:
                    # 2. レジーム判定 (トレンド vs レンジ)
                    # ADX > 25 ならトレンド、それ以下ならレンジ
                    if adx > 25:
                        regime = "トレンド相場"
                        # --- トレンドロジック (押し目・戻り) ---
                        is_uptrend = current_close > sma200
                        
                        if is_uptrend:
                            # 上昇中の押し目 (SMA20付近 or RSI低下)
                            is_dip = (current_close < sma20 * 1.01) and (rsi < 60)
                            if prob_buy >= threshold and is_dip:
                                decision = "BUY 狙い (押し目)"
                                d_class = "d-buy"
                                reason = "上昇トレンド + 押し目 + AI確度高"
                            else:
                                reason = "上昇トレンドだが、押し目待ち or AI確度不足"
                        else:
                            # 下降中の戻り
                            is_rally = (current_close > sma20 * 0.99) and (rsi > 40)
                            if prob_sell >= threshold and is_rally:
                                decision = "SELL 狙い (戻り)"
                                d_class = "d-sell"
                                reason = "下降トレンド + 戻り目 + AI確度高"
                            else:
                                reason = "下降トレンドだが、戻り待ち or AI確度不足"
                                
                    else:
                        regime = "レンジ相場"
                        # --- レンジロジック (逆張り) ---
                        # バンドブレイク or オシレーター過熱
                        
                        if prob_buy >= threshold:
                            # 売られすぎ確認
                            if bb_pb < 0.1 or rsi < 35:
                                decision = "BUY 狙い (逆張り)"
                                d_class = "d-buy"
                                reason = "レンジ下限 + 売られすぎ反発"
                            else:
                                reason = "レンジ内だが、十分安くない"
                                
                        elif prob_sell >= threshold:
                            # 買われすぎ確認
                            if bb_pb > 0.9 or rsi > 65:
                                decision = "SELL 狙い (逆張り)"
                                d_class = "d-sell"
                                reason = "レンジ上限 + 買われすぎ反落"
                            else:
                                reason = "レンジ内だが、十分高くない"
                        else:
                            reason = "レンジ内浮遊中 (方向感なし)"

                # --- UI表示 ---
                st.markdown(f"<div class='big-rate'>{current_close:.3f} <span style='font-size:1rem; color:#888'>円</span></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='sub-info'>{current_time.strftime('%m/%d %H:%M')} 確定足 | 戦略: 自動切替 ({regime})</div>", unsafe_allow_html=True)
                
                st.markdown(f"<div class='decision-box {d_class}'>{decision}</div>", unsafe_allow_html=True)
                
                # 環境認識カード
                col1, col2, col3 = st.columns(3)
                
                # レジーム
                r_color = "status-trend" if regime == "トレンド相場" else "status-range" if regime == "レンジ相場" else "status-danger"
                col1.info(f"現在の相場環境 (ADX)\n\n**{regime}**")
                
                # ボラティリティ
                v_text = "十分あり" if is_volatile_enough else "過小 (危険)"
                v_color = "status-safe" if is_volatile_enough else "status-danger"
                col2.info(f"値幅エネルギー (ATR)\n\n**{v_text}** ({atr:.3f})")
                
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
                    s200 = row['SMA200']
                    s20 = row['SMA20']
                    r = row['RSI']
                    a = row['ADX']
                    tr = row['ATR']
                    pb = row['BB_Pb']
                    
                    actual = row['Target_Buy']
                    
                    trade_res = 0
                    t_type = "-"
                    
                    # 過去シミュレーション (ロジック分岐を再現)
                    if tr > 0.03: # ATRフィルター
                        if a > 25: # トレンド
                            # Buy: 上昇トレンド + 押し目
                            if p_buy >= threshold and price > s200 and (price < s20 * 1.01 and r < 60):
                                trade_res = 15 if actual == 1 else -15
                                t_type = "BUY"
                            # Sell: 下降トレンド + 戻り
                            elif p_sell >= threshold and price < s200 and (price > s20 * 0.99 and r > 40):
                                trade_res = 15 if actual == 0 else -15
                                t_type = "SELL"
                        else: # レンジ
                            # Buy: 逆張り
                            if p_buy >= threshold and (pb < 0.1 or r < 35):
                                trade_res = 15 if actual == 1 else -15
                                t_type = "BUY"
                            # Sell: 逆張り
                            elif p_sell >= threshold and (pb > 0.9 or r > 65):
                                trade_res = 15 if actual == 0 else -15
                                t_type = "SELL"
                                
                    total_pips += trade_res
                    pips_history.append(total_pips)
                    
                    if t_type != "-":
                        trades.append({
                            "時間": row.name.strftime('%H:%M'),
                            "環境": "Trend" if a > 25 else "Range",
                            "売買": t_type,
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
                    st.caption("※ 直近10時間では、条件を満たすエントリーポイントがありませんでした。（ATRフィルター等により回避）")

            else:
                st.warning("データ不足")
