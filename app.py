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
st.set_page_config(page_title="USDJPY 5pips Scalping", layout="wide", initial_sidebar_state="collapsed")

# --- CSS ---
st.markdown("""
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .title-text { font-size: 1.8rem; font-weight: bold; color: #2c3e50; margin-bottom: 0.5rem; }
    .stButton { position: fixed; top: 20px; right: 20px; z-index: 999; }
    
    .status-card { background-color: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 5px solid #ccc; margin-bottom: 10px; }
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

# --- 関数: 特徴量作成 (短期決戦用) ---
def create_features(df):
    df = df.copy()
    
    # 短期トレンド (SMA20) と 中期トレンド (SMA75 = 15分足相当)
    df['SMA20'] = df.ta.sma(length=20)
    df['SMA75'] = df.ta.sma(length=75)
    
    # RSI (買われすぎ判定)
    df['RSI'] = df.ta.rsi(length=14)
    
    # 乖離率 (SMA20からの距離) -> これが小さい時(押し目)を狙う
    df['Disp_SMA20'] = (df['Close'] - df['SMA20']) / df['SMA20'] * 100
    
    # ADX (勢い) -> 5pips抜くにはある程度の勢いが必要
    df['ADX'] = df.ta.adx(length=14).iloc[:, 0]
    
    # 瞬発力 (1本前の値動きの大きさ)
    df['Body_Size'] = abs(df['Close'] - df['Open'])
    
    return df

# --- 関数: 正解ラベル作成 (目標: 5pips / 損切: 7pips) ---
def create_target(df, target_pips=0.05, stop_pips=0.07):
    targets = []
    scan_start = max(0, len(df) - 2000)
    
    for i in range(len(df)):
        if i < scan_start:
            targets.append(np.nan)
            continue
            
        current_close = df['Close'].iloc[i]
        target_up = current_close + target_pips
        stop_up = current_close - stop_pips  # 買いの損切ライン
        
        target_down = current_close - target_pips
        stop_down = current_close + stop_pips # 売りの損切ライン
        
        future_result = np.nan
        
        # 5pipsなら、早ければ数本で決着がつく (最大2時間=24本見る)
        for j in range(i + 1, min(len(df), i + 24)):
            future_high = df['High'].iloc[j]
            future_low = df['Low'].iloc[j]
            
            # 買い判定
            if future_high >= target_up: # 利確
                future_result = 1
                break
            if future_low <= stop_up: # 損切
                future_result = 0
                break
                
        targets.append(future_result)
        
    df['Target_Buy'] = targets
    return df

# --- メイン処理 ---
jst = pytz.timezone('Asia/Tokyo')

st.markdown("<div class='title-text'>⚡ USD/JPY 5pips 高速スキャル</div>", unsafe_allow_html=True)
update = st.button("スキャルピング分析 🔄", type="primary")

if update or True:
    with st.spinner('短期の押し目・戻り目を探索中...'):
        raw_df = get_data()
        
        if raw_df is not None:
            df = create_features(raw_df)
            # ★変更: 目標5pips (0.05), 損切7pips (0.07)
            df = create_target(df, target_pips=0.05, stop_pips=0.07)
            
            # 学習機能
            features = ['RSI', 'Disp_SMA20', 'ADX', 'Body_Size']
            data_ready = df.dropna(subset=features + ['Target_Buy', 'SMA75'])
            
            test_size = 120
            
            if len(data_ready) > test_size + 100:
                X_train = data_ready[features].iloc[:-test_size]
                y_train = data_ready['Target_Buy'].iloc[:-test_size]
                
                # スキャルピング用モデル設定 (浅い木で過学習を防ぐ)
                model = lgb.LGBMClassifier(n_estimators=100, max_depth=3, random_state=42, verbose=-1)
                model.fit(X_train, y_train)
                
                # --- 現在の状況 ---
                target_idx = -2
                current_row = df.iloc[[target_idx]]
                current_close = current_row['Close'].item()
                current_time = current_row.index[0].replace(tzinfo=pytz.utc).astimezone(jst)
                
                # 指標
                sma20 = current_row['SMA20'].item()
                sma75 = current_row['SMA75'].item()
                adx = current_row['ADX'].item()
                rsi = current_row['RSI'].item()
                disp20 = current_row['Disp_SMA20'].item()
                
                # AI予測
                prob_buy = model.predict_proba(current_row[features])[0][1] * 100
                prob_sell = 100 - prob_buy
                
                # --- 判定ロジック (5pips特化) ---
                decision = "WAIT"
                d_class = "d-wait"
                reason = "チャンス待ち"
                
                threshold = 65 # 目標が小さいので、確度65%以上でGO
                
                # 1. フィルター: 全く動かない相場は避ける
                if adx > 15:
                    # 2. トレンド判定 (SMA75 = 15分足相当のトレンド)
                    is_uptrend = current_close > sma75
                    is_downtrend = current_close < sma75
                    
                    # 3. エントリー判定 (トレンド方向 + 押し目 + AI)
                    if is_uptrend:
                        # 買い: 短期的に下がりすぎていないかチェック (RSI < 70)
                        # かつ、AIがGOサインを出している
                        if prob_buy >= threshold and rsi < 70:
                            decision = "BUY 狙い (5pips)"
                            d_class = "d-buy"
                            reason = "上昇基調 + AI確度良 (短期決戦)"
                        elif prob_buy >= threshold:
                            reason = "AIは買いだが、RSIが高すぎる(高値掴み警戒)"
                            
                    elif is_downtrend:
                        # 売り: 短期的に上がりすぎていないかチェック (RSI > 30)
                        if prob_sell >= threshold and rsi > 30:
                            decision = "SELL 狙い (5pips)"
                            d_class = "d-sell"
                            reason = "下降基調 + AI確度良 (短期決戦)"
                        elif prob_sell >= threshold:
                            reason = "AIは売りだが、RSIが低すぎる(突っ込み売り警戒)"
                else:
                    reason = "ボラティリティ不足 (ADX低迷)"

                # --- UI ---
                st.markdown(f"<div class='big-rate'>{current_close:.3f} <span style='font-size:1rem; color:#888'>円</span></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='sub-info'>{current_time.strftime('%m/%d %H:%M')} 確定足 | 戦略: 利確+5pips / 損切-7pips</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='decision-box {d_class}'>{decision}</div>", unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                # トレンド
                t_state = "上昇 (買いのみ)" if current_close > sma75 else "下降 (売りのみ)"
                col1.info(f"環境認識 (SMA75)\n\n**{t_state}**")
                
                # RSI位置
                r_text = "過熱 (高値)" if rsi > 70 else "過熱 (安値)" if rsi < 30 else "適正"
                col2.info(f"現在位置 (RSI)\n\n**{r_text}** ({rsi:.1f})")
                
                # AI
                ai_text = f"買い {prob_buy:.1f}%" if prob_buy > prob_sell else f"売り {prob_sell:.1f}%"
                col3.info(f"AI確度 (閾値65%)\n\n**{ai_text}**")
                
                st.success(f"💡 **判断根拠:** {reason}")
                st.markdown("---")
                
                # --- バックテスト (5pipsルール) ---
                st.subheader("📊 未知データでの実力テスト (直近10時間)")
                
                test_df = data_ready.tail(test_size).copy()
                test_probs = model.predict_proba(test_df[features])
                test_df['Prob_Buy'] = test_probs[:, 1]
                
                pips_history = [0]
                trades = []
                total_pips = 0
                win_count = 0
                total_count = 0
                
                for i in range(len(test_df)):
                    row = test_df.iloc[i]
                    p_buy = row['Prob_Buy'] * 100
                    p_sell = 100 - p_buy
                    
                    price = row['Close']
                    s75 = row['SMA75']
                    r = row['RSI']
                    a = row['ADX']
                    
                    actual = row['Target_Buy'] # 1=Buy成功(5pips), 0=Buy失敗(-7pips)
                    
                    trade_res = 0
                    t_type = "-"
                    
                    if a > 15: # ADXフィルター
                        # Buy
                        if price > s75 and p_buy >= threshold and r < 70:
                            trade_res = 5 if actual == 1 else -7
                            t_type = "BUY"
                        # Sell
                        elif price < s75 and p_sell >= threshold and r > 30:
                            trade_res = 5 if actual == 0 else -7
                            t_type = "SELL"
                    
                    if t_type != "-":
                        total_pips += trade_res
                        pips_history.append(total_pips)
                        total_count += 1
                        if trade_res > 0: win_count += 1
                        
                        trades.append({
                            "時間": row.name.strftime('%H:%M'),
                            "売買": t_type,
                            "結果": "WIN" if trade_res > 0 else "LOSS",
                            "Pips": trade_res
                        })
                
                # 統計情報
                if total_count > 0:
                    win_rate = (win_count / total_count) * 100
                else:
                    win_rate = 0
                    pips_history.append(0) # 描画用ダミー
                    
                # グラフ
                st.markdown(f"""
                <div style='display:flex; justify-content:space-around; align-items:center; margin-bottom:10px;'>
                    <div style='font-size:1.5rem; font-weight:bold; color:{"#00cc66" if total_pips >= 0 else "#ff4b4b"}'>合計: {total_pips:+} pips</div>
                    <div style='font-size:1.2rem; font-weight:bold;'>勝率: {win_rate:.1f}% ({win_count}/{total_count})</div>
                </div>
                """, unsafe_allow_html=True)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=pips_history, mode='lines', line=dict(color='#2c3e50', width=3)))
                fig.update_layout(height=200, margin=dict(l=20, r=20, t=10, b=20), xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#eee'))
                st.plotly_chart(fig, use_container_width=True)
                
                if trades:
                    st.write("▼ エントリー履歴")
                    st.dataframe(pd.DataFrame(trades).iloc[::-1], hide_index=True, use_container_width=True)
                else:
                    st.caption("※ 直近10時間はエントリーなし")

            else:
                st.warning("データ不足")
