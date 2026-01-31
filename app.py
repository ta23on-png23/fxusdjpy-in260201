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
st.set_page_config(page_title="USDJPY 15pips AI", layout="wide", initial_sidebar_state="collapsed")

# CSS設定
st.markdown("""
    <style>
    .block-container { padding: 1rem; }
    .title-text { font-size: 1.5rem; font-weight: bold; color: #333; margin-bottom: 0px; }
    .stButton { position: fixed; top: 15px; right: 15px; z-index: 999; }
    .big-rate { font-size: 3rem !important; font-weight: bold; text-align: center; color: #333; margin-top: 10px; margin-bottom: 0px; }
    .time-label { font-size: 1rem; text-align: center; color: #666; margin-bottom: 5px; }
    .decision-text { font-size: 2.5rem; font-weight: 900; text-align: center; padding: 15px; border-radius: 10px; color: white; margin: 10px 0; }
    .decision-wait { background-color: #888; }
    .decision-up { background-color: #00cc66; }
    .decision-down { background-color: #ff3333; }
    .bar-container { width: 100%; height: 30px; background-color: #eee; border-radius: 15px; overflow: hidden; display: flex; margin-bottom: 5px; margin-top: 15px;}
    .bar-up { height: 100%; background-color: #00cc66; text-align: left; padding-left: 10px; display: flex; align-items: center; color: white; font-weight: bold; font-size: 1rem;}
    .bar-down { height: 100%; background-color: #ff3333; text-align: right; padding-right: 10px; display: flex; align-items: center; justify-content: flex-end; color: white; font-weight: bold; font-size: 1rem;}
    .total-pips { font-size: 1.2rem; font-weight: bold; text-align: center; margin-top: 5px; }
    .plus-pips { color: #00cc66; }
    .minus-pips { color: #ff3333; }
    
    /* 根拠表示エリアのスタイル */
    .reason-box { background-color: #f8f9fa; border: 1px solid #e9ecef; border-radius: 10px; padding: 15px; margin-top: 20px; }
    .reason-title { font-weight: bold; font-size: 1.1rem; margin-bottom: 10px; color: #444; border-bottom: 2px solid #ddd; padding-bottom: 5px; }
    .reason-item { margin-bottom: 8px; font-size: 0.95rem; line-height: 1.5; }
    .tag-up { color: #00cc66; font-weight: bold; background: #e6fffa; padding: 2px 6px; border-radius: 4px; }
    .tag-down { color: #ff3333; font-weight: bold; background: #ffe6e6; padding: 2px 6px; border-radius: 4px; }
    .tag-mid { color: #666; font-weight: bold; background: #eee; padding: 2px 6px; border-radius: 4px; }
    </style>
""", unsafe_allow_html=True)

# --- 関数: データ取得 ---
def get_data_and_features():
    ticker = "USDJPY=X"
    df = yf.download(ticker, period="7d", interval="5m", progress=False)
    
    if df.empty: return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.copy()
    
    # 特徴量作成
    df['RSI'] = df.ta.rsi(length=14)
    df['RSI_Diff'] = df['RSI'].diff()
    bb = df.ta.bbands(length=20, std=2)
    df['BB_Pb'] = (df['Close'] - bb.iloc[:, 2]) / (bb.iloc[:, 0] - bb.iloc[:, 2])
    df['BB_Width'] = (bb.iloc[:, 0] - bb.iloc[:, 2]) / bb.iloc[:, 1]
    macd = df.ta.macd(fast=12, slow=26, signal=9)
    df['MACD_Hist'] = macd.iloc[:, 2]
    df['SMA20'] = df.ta.sma(length=20)
    df['SMA20_Disp'] = (df['Close'] - df['SMA20']) / df['SMA20'] * 100

    return df

# --- 関数: 正解ラベル作成 ---
def create_target(df, pips=0.15):
    targets = []
    scan_start = max(0, len(df) - 1500)
    
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
            
            if future_high >= target_up and future_low > target_down:
                future_result = 1; break
            elif future_low <= target_down and future_high < target_up:
                future_result = 0; break
        
        targets.append(future_result)
        
    df['Target'] = targets
    return df

# --- メイン処理 ---
jst = pytz.timezone('Asia/Tokyo')

st.markdown("<div class='title-text'>🇯🇵 USD/JPY 5分足AI</div>", unsafe_allow_html=True)
update = st.button("更新・判定 🔄", type="primary")

if update or True:
    with st.spinner('AI解析中...'):
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
                
                # 判定用データ
                target_row_idx = -2
                target_data = df.iloc[[target_row_idx]] 
                current_rate = target_data['Close'].item()
                target_time = target_data.index[0].replace(tzinfo=pytz.utc).astimezone(jst)
                
                prob = model.predict_proba(target_data[features])[0]
                prob_up = int(prob[1] * 100)
                prob_down = 100 - prob_up
                
                # 表示
                time_str = target_time.strftime('%H:%M')
                date_str = target_time.strftime('%m/%d')
                
                st.markdown(f"<div class='time-label'>{date_str} {time_str} 確定足</div>", unsafe_allow_html=True)
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
                st.markdown("<div style='text-align:center; color:#888; font-size:0.8rem;'>目標: ±15pips / 基準値: 73%</div>", unsafe_allow_html=True)

                st.markdown("---")
                
                # グラフ
                st.subheader("📊 直近の戦績 (確定分30本)")
                valid_history_df = df.dropna(subset=['Target'])
                
                if not valid_history_df.empty:
                    sim_df = valid_history_df.tail(30).copy()
                    sim_probs = model.predict_proba(sim_df[features])
                    sim_df['Prob_Up'] = sim_probs[:, 1]
                    
                    pips_history = [0]
                    total_pips = 0
                    
                    for i in range(len(sim_df)):
                        p_up = sim_df['Prob_Up'].iloc[i] * 100
                        actual = sim_df['Target'].iloc[i]
                        res = 0
                        if p_up >= threshold: res = 15 if actual==1 else -15
                        elif p_up <= (100-threshold): res = 15 if actual==0 else -15
                        total_pips += res
                        pips_history.append(total_pips)
                    
                    p_col = "plus-pips" if total_pips >= 0 else "minus-pips"
                    st.markdown(f"<div class='total-pips {p_col}'>合計: {total_pips:+} pips</div>", unsafe_allow_html=True)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=pips_history, mode='lines', line=dict(color='#333', width=3)))
                    fig.update_layout(margin=dict(l=10, r=10, t=10, b=20), height=150, showlegend=False, xaxis=dict(visible=False), yaxis=dict(showgrid=True, gridcolor='#eee'))
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

                # --- ★改善機能: 使用インジケーター全リスト表示 ---
                st.markdown("<div class='reason-box'>", unsafe_allow_html=True)
                st.markdown("<div class='reason-title'>📝 AI判断材料 (インジケーター一覧)</div>", unsafe_allow_html=True)
                
                # --- 1. RSI ---
                rsi_val = target_data['RSI'].item()
                rsi_status = "<span class='tag-mid'>中立</span>"
                if rsi_val > 60: rsi_status = "<span class='tag-up'>上昇圏</span>"
                elif rsi_val < 40: rsi_status = "<span class='tag-down'>下降圏</span>"
                st.markdown(f"<div class='reason-item'><b>RSI (14)</b>: {rsi_val:.1f} → {rsi_status}</div>", unsafe_allow_html=True)
                
                # --- 2. SMA乖離 ---
                sma_val = target_data['SMA20_Disp'].item()
                sma_status = "<span class='tag-mid'>レンジ気味</span>"
                if sma_val > 0.05: sma_status = "<span class='tag-up'>上昇トレンド (SMAより上)</span>"
                elif sma_val < -0.05: sma_status = "<span class='tag-down'>下降トレンド (SMAより下)</span>"
                st.markdown(f"<div class='reason-item'><b>移動平均線 (20)</b>: 乖離{sma_val:.2f}% → {sma_status}</div>", unsafe_allow_html=True)

                # --- 3. ボリンジャーバンド ---
                bb_pb = target_data['BB_Pb'].item()
                bb_status = "<span class='tag-mid'>バンド内推移</span>"
                if bb_pb > 1.0: bb_status = "<span class='tag-up'>+2σブレイク (強気)</span>"
                elif bb_pb < 0.0: bb_status = "<span class='tag-down'>-2σブレイク (弱気)</span>"
                elif bb_pb > 0.8: bb_status = "<span class='tag-up'>高値圏</span>"
                elif bb_pb < 0.2: bb_status = "<span class='tag-down'>安値圏</span>"
                st.markdown(f"<div class='reason-item'><b>ボリンジャーバンド</b>: 位置{bb_pb:.2f} → {bb_status}</div>", unsafe_allow_html=True)

                # --- 4. MACD ---
                macd_val = target_data['MACD_Hist'].item()
                macd_status = "<span class='tag-mid'>中立</span>"
                if macd_val > 0.005: macd_status = "<span class='tag-up'>買い優勢</span>"
                elif macd_val < -0.005: macd_status = "<span class='tag-down'>売り優勢</span>"
                st.markdown(f"<div class='reason-item'><b>MACD</b>: ヒストグラム{macd_val:.3f} → {macd_status}</div>", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                # ----------------------------------------

            else:
                st.warning("データ不足")
