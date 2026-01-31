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
st.set_page_config(page_title="USDJPY Pullback AI", layout="wide", initial_sidebar_state="collapsed")

# --- CSS (デザイン調整) ---
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
    .d-buy { background-color: #27ae60; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .d-sell { background-color: #c0392b; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    
    .dataframe { font-size: 0.8rem !important; }
    </style>
""", unsafe_allow_html=True)

# --- 関数: データ取得 ---
def get_data():
    ticker = "USDJPY=X"
    # データ期間を確保
    df = yf.download(ticker, period="7d", interval="5m", progress=False)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.copy()

# --- 関数: 特徴量作成 (乖離・押し目重視) ---
def create_features(df):
    df = df.copy()
    
    # 基本指標
    df['SMA20'] = df.ta.sma(length=20)
    df['SMA200'] = df.ta.sma(length=200) # 長期トレンド
    df['RSI'] = df.ta.rsi(length=14)
    
    # ★重要: 「乖離率」を追加 (移動平均からどれくらい離れているか)
    # これがプラスに大きいと「上がりすぎ」、マイナスだと「下がりすぎ(押し目)」
    df['Disp_SMA20'] = (df['Close'] - df['SMA20']) / df['SMA20'] * 100
    df['Disp_SMA200'] = (df['Close'] - df['SMA200']) / df['SMA200'] * 100
    
    # ボラティリティ
    adx = df.ta.adx(length=14)
    df['ADX'] = adx.iloc[:, 0]
    
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
        # 15pips動くか、最大4時間経過するまで
        for j in range(i + 1, min(len(df), i + 48)):
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

st.markdown("<div class='title-text'>📉 USD/JPY 押し目買い/戻り売りAI</div>", unsafe_allow_html=True)
update = st.button("市場分析・判定 🔄", type="primary")

if update or True:
    with st.spinner('最適なエントリーポイントを探索中...'):
        raw_df = get_data()
        
        if raw_df is not None:
            df = create_features(raw_df)
            df = create_target(df, pips=0.15)
            
            # 学習に使う特徴量 (乖離率を重視)
            features = ['RSI', 'Disp_SMA20', 'Disp_SMA200', 'ADX']
            
            data_ready = df.dropna(subset=features + ['Target_Buy', 'SMA200'])
            
            # --- 厳格な学習・テスト分離 ---
            test_size = 120
            
            if len(data_ready) > test_size + 100:
                X_train = data_ready[features].iloc[:-test_size]
                y_train = data_ready['Target_Buy'].iloc[:-test_size]
                
                # LightGBMモデル
                model = lgb.LGBMClassifier(n_estimators=100, max_depth=5, random_state=42, verbose=-1)
                model.fit(X_train, y_train)
                
                # --- 現在の状況取得 ---
                target_idx = -2
                current_row = df.iloc[[target_idx]]
                current_close = current_row['Close'].item()
                current_time = current_row.index[0].replace(tzinfo=pytz.utc).astimezone(jst)
                
                # 指標値
                sma200 = current_row['SMA200'].item()
                sma20 = current_row['SMA20'].item()
                adx = current_row['ADX'].item()
                rsi = current_row['RSI'].item()
                disp_sma20 = current_row['Disp_SMA20'].item()
                
                # AI予測
                prob_buy = model.predict_proba(current_row[features])[0][1] * 100
                prob_sell = 100 - prob_buy
                
                # --- 🧠 判定ロジック (押し目・戻り売り戦略) ---
                
                decision = "WAIT"
                d_class = "d-wait"
                reason = "条件不一致"
                
                threshold = 70 # 少し緩和してチャンスを増やす
                
                # 1. 長期トレンド判定 (SMA200)
                is_uptrend = current_close > sma200
                
                # 2. 「引きつけ」判定 (短期的に逆行しているか？)
                # 上昇トレンド中なら、価格がSMA20付近かそれ以下、またはRSIが低めなら「押し目」
                is_dip = (current_close < sma20) or (rsi < 55)
                # 下降トレンド中なら、価格がSMA20付近かそれ以上、またはRSIが高めなら「戻り」
                is_rally = (current_close > sma20) or (rsi > 45)
                
                if adx > 20: # ある程度動いている時
                    if is_uptrend:
                        # 買い条件: AI強気 + 押し目(Dip)発生中
                        if prob_buy >= threshold and is_dip:
                            decision = "BUY 狙い (押し目)"
                            d_class = "d-buy"
                            reason = "上昇トレンド中の調整局面を狙う"
                        elif prob_buy >= threshold and not is_dip:
                            reason = "AIは強気だが、価格が高すぎる(押し目待ち)"
                            
                    else: # 下降トレンド
                        # 売り条件: AI弱気(Buy低) + 戻り(Rally)発生中
                        if prob_sell >= threshold and is_rally:
                            decision = "SELL 狙い (戻り)"
                            d_class = "d-sell"
                            reason = "下降トレンド中の反発局面を狙う"
                        elif prob_sell >= threshold and not is_rally:
                            reason = "AIは弱気だが、価格が安すぎる(戻り待ち)"
                else:
                    reason = "相場エネルギー不足 (ADX低迷)"

                # --- UI表示 ---
                st.markdown(f"<div class='big-rate'>{current_close:.3f} <span style='font-size:1rem; color:#888'>円</span></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='sub-info'>{current_time.strftime('%m/%d %H:%M')} 確定足 | ロジック: 押し目買い/戻り売り</div>", unsafe_allow_html=True)
                
                st.markdown(f"<div class='decision-box {d_class}'>{decision}</div>", unsafe_allow_html=True)
                
                # ステータス表示
                col1, col2, col3 = st.columns(3)
                
                # トレンド
                t_icon = "↗️ 上昇中" if is_uptrend else "↘️ 下降中"
                col1.info(f"長期トレンド(SMA200)\n\n**{t_icon}**")
                
                # 現在位置（重要）
                pos_text = "安い (買い場)" if disp_sma20 < 0 else "高い (売り場)" if disp_sma20 > 0 else "中立"
                col2.info(f"短期的な価格位置\n\n**{pos_text}** (乖離 {disp_sma20:.3f}%)")
                
                # AI
                ai_text = f"買い {prob_buy:.1f}%" if is_uptrend else f"売り {prob_sell:.1f}%"
                col3.info(f"AI予測\n\n**{ai_text}**")

                st.success(f"💡 **判断根拠:** {reason}")
                st.markdown("---")
                
                # --- 厳格バックテスト結果 ---
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
                    sma200_val = row['SMA200']
                    sma20_val = row['SMA20']
                    r = row['RSI']
                    a = row['ADX']
                    
                    actual = row['Target_Buy']
                    
                    trade_res = 0
                    t_type = "-"
                    
                    # 過去データでも同じロジックで検証
                    # 買い: AIOK + トレンド上 + (価格<SMA20 or RSI<55)
                    if p_buy >= threshold and price > sma200_val and (price < sma20_val or r < 55) and a > 20:
                        trade_res = 15 if actual == 1 else -15
                        t_type = "BUY"
                    
                    # 売り: AIOK + トレンド下 + (価格>SMA20 or RSI>45)
                    elif p_sell >= threshold and price < sma200_val and (price > sma20_val or r > 45) and a > 20:
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
                    st.caption("※ 直近10時間では、エントリー条件（押し目・戻り）を満たすポイントがありませんでした。")

            else:
                st.warning("データ不足")
