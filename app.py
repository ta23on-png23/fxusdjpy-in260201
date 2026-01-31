import streamlit as st
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import numpy as np
import lightgbm as lgb
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz

# --- ページ設定 (スマホ対応) ---
st.set_page_config(page_title="USDJPY 15pips AI", layout="centered", initial_sidebar_state="collapsed")

# CSSで文字サイズやデザインをスマホ向けに調整
st.markdown("""
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .big-rate { font-size: 3rem !important; font-weight: bold; text-align: center; color: #333; margin-bottom: 0px; }
    .time-label { font-size: 1.2rem; text-align: center; color: #666; margin-bottom: 20px; }
    .decision-text { font-size: 2.5rem; font-weight: 900; text-align: center; padding: 10px; border-radius: 10px; color: white; margin: 10px 0; }
    .decision-wait { background-color: #888; }
    .decision-up { background-color: #00cc66; }
    .decision-down { background-color: #ff3333; }
    .bar-container { width: 100%; height: 40px; background-color: #eee; border-radius: 20px; overflow: hidden; display: flex; margin-bottom: 5px;}
    .bar-up { height: 100%; background-color: #00cc66; text-align: left; padding-left: 10px; display: flex; align-items: center; color: white; font-weight: bold; font-size: 1.2rem;}
    .bar-down { height: 100%; background-color: #ff3333; text-align: right; padding-right: 10px; display: flex; align-items: center; justify-content: flex-end; color: white; font-weight: bold; font-size: 1.2rem;}
    .total-pips { font-size: 1.5rem; font-weight: bold; text-align: center; margin-top: 10px; }
    .plus-pips { color: #00cc66; }
    .minus-pips { color: #ff3333; }
    </style>
""", unsafe_allow_html=True)

# --- 関数: データの取得と特徴量作成 ---
def get_data_and_features():
    ticker = "USDJPY=X"
    # 5日分のデータを取得
    df = yf.download(ticker, period="5d", interval="5m", progress=False)
    
    if df.empty:
        return None

    # ★ここが修正ポイント: MultiIndex対策
    # 列名が ('Close', 'USDJPY=X') のようになっている場合、 'Close' だけにする
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.copy()
    
    # 特徴量エンジニアリング
    # 1. RSIとその変化
    df['RSI'] = df.ta.rsi(length=14)
    df['RSI_Diff'] = df['RSI'].diff()
    
    # 2. ボリンジャーバンド%Bと幅
    bb = df.ta.bbands(length=20, std=2)
    df['BB_Pb'] = (df['Close'] - bb.iloc[:, 2]) / (bb.iloc[:, 0] - bb.iloc[:, 2])
    df['BB_Width'] = (bb.iloc[:, 0] - bb.iloc[:, 2]) / bb.iloc[:, 1]
    
    # 3. MACD
    macd = df.ta.macd(fast=12, slow=26, signal=9)
    df['MACD_Hist'] = macd.iloc[:, 2]

    # 4. SMA乖離率
    df['SMA20'] = df.ta.sma(length=20)
    df['SMA20_Disp'] = (df['Close'] - df['SMA20']) / df['SMA20'] * 100

    return df

# --- 関数: 正解ラベルの作成 (±15pips判定) ---
def create_target(df, pips=0.15):
    targets = []
    scan_start = max(0, len(df) - 1000)
    
    for i in range(len(df)):
        if i < scan_start:
            targets.append(np.nan)
            continue
            
        current_close = df['Close'].iloc[i]
        target_up = current_close + pips
        target_down = current_close - pips
        
        future_result = np.nan
        # 未来4時間(48本)先まで探索
        for j in range(i + 1, min(len(df), i + 48)):
            future_high = df['High'].iloc[j]
            future_low = df['Low'].iloc[j]
            
            hit_up = future_high >= target_up
            hit_down = future_low <= target_down
            
            if hit_up and not hit_down:
                future_result = 1 # 上昇勝利
                break
            elif hit_down and not hit_up:
                future_result = 0 # 下降勝利
                break
            elif hit_up and hit_down:
                future_result = np.nan 
                break
        
        targets.append(future_result)
        
    df['Target'] = targets
    return df

# --- メイン処理 ---
jst = pytz.timezone('Asia/Tokyo')

col_head, col_btn = st.columns([2, 1])
with col_head:
    st.markdown("### 🇯🇵 USD/JPY 5分足AI")
with col_btn:
    update = st.button("更新・判定 🔄", type="primary")

if update or True:
    with st.spinner('AIがチャート分析中...'):
        df = get_data_and_features()
        
        if df is not None:
            df = create_target(df, pips=0.15)
            
            features = ['RSI', 'RSI_Diff', 'BB_Pb', 'BB_Width', 'MACD_Hist', 'SMA20_Disp']
            train_df = df.dropna(subset=features + ['Target'])
            
            if len(train_df) > 50:
                X = train_df[features]
                y = train_df['Target']
                
                model = lgb.LGBMClassifier(n_estimators=100, max_depth=3, random_state=42, verbose=-1)
                model.fit(X, y)
                
                # 判定用データ（Liveデータの直前の確定足を使用）
                last_timestamp = df.index[-1].replace(tzinfo=pytz.utc).astimezone(jst)
                target_row_idx = -2
                target_data = df.iloc[[target_row_idx]] 
                current_rate = target_data['Close'].item()
                target_time = target_data.index[0].replace(tzinfo=pytz.utc).astimezone(jst)
                
                prob = model.predict_proba(target_data[features])[0]
                prob_up = int(prob[1] * 100)
                prob_down = 100 - prob_up
                
                # --- 画面表示 ---
                time_str = target_time.strftime('%H:%M')
                st.markdown(f"<div class='time-label'>{time_str} 確定足 (日本時間)</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='big-rate'>{current_rate:.3f}</div>", unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class='bar-container'>
                    <div class='bar-up' style='width: {prob_up}%;'>{prob_up}%</div>
                    <div class='bar-down' style='width: {prob_down}%;'>{prob_down}%</div>
                </div>
                """, unsafe_allow_html=True)
                
                threshold = 73
                decision = "WAIT"
                css_class = "decision-wait"
                
                if prob_up >= threshold:
                    decision = "UP 狙い"
                    css_class = "decision-up"
                elif prob_down >= threshold:
                    decision = "DOWN 狙い"
                    css_class = "decision-down"
                    
                st.markdown(f"<div class='decision-text {css_class}'>{decision}</div>", unsafe_allow_html=True)
                st.markdown("<div style='text-align:center; color:#888; font-size:0.8rem;'>目標: ±15pips / 閾値: 73%</div>", unsafe_allow_html=True)

                st.markdown("---")
                st.subheader("📊 過去3時間の戦績 (60本)")
                
                sim_df = df.iloc[-62:-2].copy()
                if not sim_df.empty:
                    sim_probs = model.predict_proba(sim_df[features])
                    sim_df['Prob_Up'] = sim_probs[:, 1]
                    
                    pips_history = [0]
                    total_pips = 0
                    
                    for i in range(len(sim_df)):
                        p_up = sim_df['Prob_Up'].iloc[i] * 100
                        actual_target = sim_df['Target'].iloc[i]
                        
                        result_pips = 0
                        if p_up >= threshold:
                            if actual_target == 1: result_pips = 15
                            elif actual_target == 0: result_pips = -15
                        elif p_up <= (100 - threshold):
                            if actual_target == 0: result_pips = 15
                            elif actual_target == 1: result_pips = -15
                        
                        total_pips += result_pips
                        pips_history.append(total_pips)
                    
                    pips_color = "plus-pips" if total_pips >= 0 else "minus-pips"
                    st.markdown(f"<div class='total-pips {pips_color}'>合計: {total_pips:+} pips</div>", unsafe_allow_html=True)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=pips_history, mode='lines',
                        line=dict(color='#333', width=3), name='Pips'
                    ))
                    fig.update_layout(
                        margin=dict(l=20, r=20, t=10, b=20),
                        height=200, showlegend=False,
                        xaxis=dict(showgrid=False, visible=False),
                        yaxis=dict(showgrid=True, gridcolor='#eee')
                    )
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            else:
                st.warning("学習データ不足のため計算できません")
