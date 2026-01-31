import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import lightgbm as lgb
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz

# --- ページ設定 ---
st.set_page_config(page_title="USDJPY Pure AI", layout="wide", initial_sidebar_state="collapsed")

# CSS設定
st.markdown("""
    <style>
    .block-container { padding-top: 3rem; padding-bottom: 2rem; padding-left: 1rem; padding-right: 1rem; }
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
    .condition-note { font-size: 0.9rem; color: #666; margin-bottom: 10px; }
    .reason-box { background-color: #f8f9fa; border: 1px solid #e9ecef; border-radius: 10px; padding: 15px; margin-top: 20px; }
    .reason-title { font-weight: bold; font-size: 1.1rem; margin-bottom: 10px; color: #444; border-bottom: 2px solid #ddd; padding-bottom: 5px; }
    .reason-item { margin-bottom: 8px; font-size: 0.95rem; line-height: 1.5; }
    </style>
""", unsafe_allow_html=True)

# --- 関数: データ取得 ---
def get_data():
    ticker = "USDJPY=X"
    # データ量確保
    df = yf.download(ticker, period="7d", interval="5m", progress=False)
    
    if df.empty: return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.copy()
    return df

# --- 関数: 特徴量作成 (インジケーターなし・形状認識) ---
def create_features(df):
    df = df.copy()
    
    # AIに「チャートの形」を教えるための処理
    # 過去12本（1時間分）の「値動きの比率」を計算
    # Log Return (対数収益率) を使うことで、130円でも150円でも同じ「形」として認識させる
    
    lags = 12 # 過去12本分を見る
    cols = []
    
    for i in range(1, lags + 1):
        col_name = f'Lag_{i}'
        # (今の終値 - i本前の終値) / i本前の終値 * 10000 (pips換算に近い値)
        df[col_name] = np.log(df['Close'] / df['Close'].shift(i)) * 10000
        cols.append(col_name)
        
    # 現在の足の実体の大きさ (勢い)
    df['Body_Size'] = np.log(df['Close'] / df['Open']) * 10000
    
    # ヒゲの長さ（反発の強さ）
    df['Upper_Shadow'] = np.log(df['High'] / df[['Close', 'Open']].max(axis=1)) * 10000
    df['Lower_Shadow'] = np.log(df[['Close', 'Open']].min(axis=1) / df['Low']) * 10000
    
    cols.extend(['Body_Size', 'Upper_Shadow', 'Lower_Shadow'])
    
    return df, cols

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
        # 15pips動くのに十分な時間（最大4時間）を見る
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

st.markdown("<div class='title-text'>🇯🇵 USD/JPY 純粋形状認識AI</div>", unsafe_allow_html=True)
update = st.button("更新・判定 🔄", type="primary")

if update or True:
    with st.spinner('AIがチャートの形状をパターン認識中...'):
        df = get_data()
        
        if df is not None:
            df, features = create_features(df)
            df = create_target(df, pips=0.15)
            
            full_data = df.dropna(subset=features + ['Target'])
            
            simulation_count = 120 # 未知データテスト数
            
            if len(full_data) > simulation_count + 100:
                # 学習用とテスト用を分離
                X_train = full_data[features].iloc[:-simulation_count]
                y_train = full_data['Target'].iloc[:-simulation_count]
                sim_df = full_data.tail(simulation_count).copy()
                
                # インジケーターが無い分、木の深さを深くして複雑なパターンを読めるようにする
                model = lgb.LGBMClassifier(n_estimators=200, max_depth=7, num_leaves=31, random_state=42, verbose=-1)
                model.fit(X_train, y_train)
                
                # --- 現在の判定 ---
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
                
                # 基準値設定 (形状認識はノイズに強いので少し下げても機能するが、安全のため73%)
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
                st.markdown(f"<div style='text-align:center; color:#888; font-size:0.8rem;'>目標: ±15pips / 基準値: {threshold}% (形状認識)</div>", unsafe_allow_html=True)

                st.markdown("---")
                
                # --- グラフ表示 ---
                st.subheader("📊 直近の戦績 (確定分120本)")
                st.markdown(f"<div class='condition-note'>※ インジケーター不使用：ローソク足の形状パターンのみで判断</div>", unsafe_allow_html=True)

                if not sim_df.empty:
                    sim_probs = model.predict_proba(sim_df[features])
                    sim_df['Prob_Up'] = sim_probs[:, 1]
                    
                    pips_history = [0]
                    total_pips = 0
                    
                    for i in range(len(sim_df)):
                        p_up = sim_df['Prob_Up'].iloc[i] * 100
                        p_down = 100 - p_up
                        actual = sim_df['Target'].iloc[i]
                        
                        res = 0
                        if p_up >= threshold: res = 15 if actual==1 else -15
                        elif p_down >= threshold: res = 15 if actual==0 else -15
                        
                        total_pips += res
                        pips_history.append(total_pips)
                    
                    p_col = "plus-pips" if total_pips >= 0 else "minus-pips"
                    st.markdown(f"<div class='total-pips {p_col}'>合計: {total_pips:+} pips</div>", unsafe_allow_html=True)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=pips_history, mode='lines', line=dict(color='#333', width=3)))
                    fig.update_layout(
                        margin=dict(l=10, r=10, t=10, b=30), height=180, showlegend=False,
                        xaxis=dict(visible=True, showgrid=False, tickmode='linear', tick0=0, dtick=20, fixedrange=True),
                        yaxis=dict(showgrid=True, gridcolor='#eee')
                    )
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

                # --- 形状分析レポート ---
                st.markdown("<div class='reason-box'>", unsafe_allow_html=True)
                st.markdown("<div class='reason-title'>📝 AI形状分析レポート (直近1時間の動き)</div>", unsafe_allow_html=True)
                
                # 直近の動きを言語化
                last_move = target_data['Lag_1'].item()
                body_size = target_data['Body_Size'].item()
                u_shadow = target_data['Upper_Shadow'].item()
                l_shadow = target_data['Lower_Shadow'].item()
                
                # 1. 直近の勢い
                if last_move > 0.05: st.markdown("<div class='reason-item'>🚀 <b>直近の足</b>: 強い上昇</div>", unsafe_allow_html=True)
                elif last_move < -0.05: st.markdown("<div class='reason-item'>🔻 <b>直近の足</b>: 強い下落</div>", unsafe_allow_html=True)
                else: st.markdown("<div class='reason-item'>➡ <b>直近の足</b>: 停滞/小動き</div>", unsafe_allow_html=True)

                # 2. ヒゲの分析
                if u_shadow > 0.05: st.markdown("<div class='reason-item'>✋ <b>上ヒゲ検知</b>: 上値が重い (売り圧力あり)</div>", unsafe_allow_html=True)
                if l_shadow > 0.05: st.markdown("<div class='reason-item'>💪 <b>下ヒゲ検知</b>: 底堅い (買い支えあり)</div>", unsafe_allow_html=True)
                
                # 3. 過去1時間の累積
                cumulative_move = target_data[[f'Lag_{i}' for i in range(1, 13)]].sum(axis=1).item()
                if cumulative_move > 0.1: st.markdown("<div class='reason-item'>📈 <b>1時間の流れ</b>: 全体的に上昇基調</div>", unsafe_allow_html=True)
                elif cumulative_move < -0.1: st.markdown("<div class='reason-item'>📉 <b>1時間の流れ</b>: 全体的に下落基調</div>", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)

            else:
                st.warning("データ不足")
