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
st.set_page_config(page_title="USDJPY Safety Trend AI", layout="wide", initial_sidebar_state="collapsed")

# --- CSS (デザイン調整) ---
st.markdown("""
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .title-text { font-size: 1.8rem; font-weight: bold; color: #2c3e50; margin-bottom: 0.5rem; }
    .stButton { position: fixed; top: 20px; right: 20px; z-index: 999; }
    
    /* ステータスカード */
    .status-card { background-color: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 5px solid #ccc; margin-bottom: 10px; }
    .status-safe { border-left-color: #00cc66; background-color: #e8f5e9; }
    .status-danger { border-left-color: #ff4b4b; background-color: #ffebee; }
    .status-neutral { border-left-color: #ff9800; background-color: #fff3e0; }
    
    .big-rate { font-size: 2.5rem; font-weight: bold; text-align: center; color: #333; }
    .sub-info { font-size: 0.9rem; color: #666; text-align: center; }
    
    /* 判定文字 */
    .decision-box { font-size: 2rem; font-weight: 900; text-align: center; padding: 15px; border-radius: 8px; color: white; margin: 15px 0; }
    .d-wait { background-color: #95a5a6; }
    .d-buy { background-color: #27ae60; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .d-sell { background-color: #c0392b; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    
    /* トレード履歴テーブル */
    .dataframe { font-size: 0.8rem !important; }
    </style>
""", unsafe_allow_html=True)

# --- 関数: データ取得 ---
def get_data():
    ticker = "USDJPY=X"
    # 長期MA(200)とADX計算のために十分な期間を取得 (7日分)
    df = yf.download(ticker, period="7d", interval="5m", progress=False)
    
    if df.empty: return None

    # MultiIndex対策
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.copy()
    return df

# --- 関数: 特徴量作成 (インジケーター活用) ---
def create_features(df):
    df = df.copy()
    
    # 1. トレンド系 (SMA)
    df['SMA20'] = df.ta.sma(length=20)
    df['SMA50'] = df.ta.sma(length=50)
    df['SMA200'] = df.ta.sma(length=200) # 長期トレンドフィルター用
    
    # 2. オシレーター系 (RSI, MACD)
    df['RSI'] = df.ta.rsi(length=14)
    macd = df.ta.macd(fast=12, slow=26, signal=9)
    df['MACD'] = macd.iloc[:, 0]
    df['MACD_Signal'] = macd.iloc[:, 1]
    df['MACD_Hist'] = macd.iloc[:, 2]
    
    # 3. ボラティリティ系 (BB, ADX)
    bb = df.ta.bbands(length=20, std=2)
    df['BB_Width'] = (bb.iloc[:, 0] - bb.iloc[:, 2]) / bb.iloc[:, 1]
    df['BB_Pb'] = (df['Close'] - bb.iloc[:, 2]) / (bb.iloc[:, 0] - bb.iloc[:, 2])
    
    # ADX (トレンドの強さ) - 重要フィルター
    adx = df.ta.adx(length=14)
    df['ADX'] = adx.iloc[:, 0] # ADXメイン線

    # 4. 乖離率
    df['Dist_SMA200'] = (df['Close'] - df['SMA200']) / df['SMA200'] * 100
    
    return df

# --- 関数: 正解ラベル作成 (±15pips) ---
def create_target(df, pips=0.15):
    targets = []
    # 処理高速化のため直近2000本のみ計算
    scan_start = max(0, len(df) - 2000)
    
    for i in range(len(df)):
        if i < scan_start:
            targets.append(np.nan)
            continue
            
        current_close = df['Close'].iloc[i]
        target_up = current_close + pips
        target_down = current_close - pips
        
        future_result = np.nan # 0:Wait/Loss, 1:Win
        
        # 最大48本(4時間)先まで見る
        for j in range(i + 1, min(len(df), i + 48)):
            future_high = df['High'].iloc[j]
            future_low = df['Low'].iloc[j]
            
            # 順張りAIを作るため、「買い成功」か「売り成功」かをトレンドに合わせて判定させたいが、
            # ここではシンプルに「次に+15pipsにタッチするか？」を予測させる (買い目線モデル)
            # ※売りは逆ロジックで判定
            
            if future_high >= target_up and future_low > target_down:
                future_result = 1 # 上昇勝利
                break
            elif future_low <= target_down and future_high < target_up:
                future_result = 0 # 下降勝利 (買いなら負け)
                break
        
        targets.append(future_result)
        
    df['Target_Buy'] = targets
    return df

# --- メイン処理 ---
jst = pytz.timezone('Asia/Tokyo')

st.markdown("<div class='title-text'>🛡️ USD/JPY 安全重視トレンドAI</div>", unsafe_allow_html=True)
update = st.button("市場分析・判定 🔄", type="primary")

if update or True:
    with st.spinner('市場環境を精査中...'):
        raw_df = get_data()
        
        if raw_df is not None:
            # 特徴量エンジニアリング
            df = create_features(raw_df)
            df = create_target(df, pips=0.15)
            
            # --- フィルタリング条件の定義 ---
            # 1. 必要なカラムが揃っているか
            features = ['RSI', 'MACD_Hist', 'BB_Width', 'BB_Pb', 'ADX', 'Dist_SMA200']
            data_ready = df.dropna(subset=features + ['Target_Buy', 'SMA200'])
            
            # --- モデル学習 (厳格な時系列スプリット) ---
            # 直近120本(約10時間)はテスト用に取り分ける
            test_size = 120
            
            if len(data_ready) > test_size + 100:
                X_train = data_ready[features].iloc[:-test_size]
                y_train = data_ready['Target_Buy'].iloc[:-test_size]
                
                # LightGBMモデル
                model = lgb.LGBMClassifier(n_estimators=100, max_depth=4, num_leaves=15, random_state=42, verbose=-1)
                model.fit(X_train, y_train)
                
                # --- 現在の状況取得 ---
                # 判定に使うのは「確定した最新の足」
                target_idx = -2
                current_row = df.iloc[[target_idx]]
                current_close = current_row['Close'].item()
                current_time = current_row.index[0].replace(tzinfo=pytz.utc).astimezone(jst)
                
                # インジケーター値の取得
                sma200 = current_row['SMA200'].item()
                adx = current_row['ADX'].item()
                rsi = current_row['RSI'].item()
                
                # AI予測 (上昇確率)
                prob_buy = model.predict_proba(current_row[features])[0][1] * 100
                prob_sell = 100 - prob_buy # 2値分類なので逆が売り確率
                
                # --- 🛡️ 安全装置 (フィルターロジック) ---
                
                # 1. トレンドフィルター (SMA200)
                trend_direction = "NEUTRAL"
                if current_close > sma200: trend_direction = "UP"
                elif current_close < sma200: trend_direction = "DOWN"
                
                # 2. ボラティリティフィルター (ADX)
                # ADXが20未満はトレンドなし（レンジ・停滞）とみなす
                is_active_market = adx > 20
                
                # --- 最終判定 ---
                decision = "WAIT"
                d_class = "d-wait"
                reason = "様子見推奨"
                
                # 閾値設定 (75%以上の確度が必要)
                threshold = 75
                
                if not is_active_market:
                    reason = "市場エネルギー不足 (ADX低迷)"
                else:
                    # 買い判定: AIが強気 + 長期トレンドが上 + RSIが買われすぎでない
                    if prob_buy >= threshold and trend_direction == "UP":
                        if rsi < 70: # 70以上は高値掴み警戒
                            decision = "BUY 狙い"
                            d_class = "d-buy"
                            reason = "上昇トレンド順張り + AI確度高"
                        else:
                            reason = "トレンドは上だが過熱気味 (RSI高)"
                            
                    # 売り判定: AIが弱気 + 長期トレンドが下 + RSIが売られすぎでない
                    elif prob_sell >= threshold and trend_direction == "DOWN":
                        if rsi > 30: # 30以下は突っ込み売り警戒
                            decision = "SELL 狙い"
                            d_class = "d-sell"
                            reason = "下降トレンド順張り + AI確度高"
                        else:
                            reason = "トレンドは下だが売られすぎ (RSI低)"
                    else:
                        reason = "トレンドとAI予測が不一致、または確度不足"

                # --- UI表示 ---
                
                # ヘッダー情報
                st.markdown(f"<div class='big-rate'>{current_close:.3f} <span style='font-size:1rem; color:#888'>円</span></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='sub-info'>{current_time.strftime('%m/%d %H:%M')} 確定足 | 目標: ±15pips</div>", unsafe_allow_html=True)
                
                # 判定表示
                st.markdown(f"<div class='decision-box {d_class}'>{decision}</div>", unsafe_allow_html=True)
                
                # 環境認識カード
                col1, col2, col3 = st.columns(3)
                
                # トレンド状態
                t_color = "status-safe" if trend_direction != "NEUTRAL" else "status-neutral"
                t_icon = "↗️ 上昇 (強)" if trend_direction == "UP" else "↘️ 下降 (強)" if trend_direction == "DOWN" else "➡️ レンジ"
                col1.markdown(f"""
                <div class='status-card {t_color}'>
                    <div style='font-size:0.8rem; color:#555;'>長期トレンド (SMA200)</div>
                    <div style='font-weight:bold; font-size:1.1rem;'>{t_icon}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # ボラティリティ状態
                v_color = "status-safe" if is_active_market else "status-danger"
                v_text = f"活発 (ADX:{adx:.1f})" if is_active_market else f"停滞 (ADX:{adx:.1f})"
                col2.markdown(f"""
                <div class='status-card {v_color}'>
                    <div style='font-size:0.8rem; color:#555;'>相場の勢い</div>
                    <div style='font-weight:bold; font-size:1.1rem;'>{v_text}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # AI確度
                ai_prob = prob_buy if trend_direction == "UP" else prob_sell
                a_color = "status-safe" if ai_prob >= threshold else "status-neutral"
                col3.markdown(f"""
                <div class='status-card {a_color}'>
                    <div style='font-size:0.8rem; color:#555;'>AI順張り確度</div>
                    <div style='font-weight:bold; font-size:1.1rem;'>{ai_prob:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

                st.info(f"💡 **判断根拠:** {reason}")
                
                st.markdown("---")
                
                # --- 厳格なバックテスト結果 (直近120本) ---
                st.subheader("📊 未知データでの実力テスト (直近10時間)")
                
                # テストデータを作成
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
                    sma = row['SMA200']
                    r = row['RSI']
                    a = row['ADX']
                    
                    actual = row['Target_Buy'] # 1=BuyWin, 0=BuyLoss(SellWin)
                    
                    # フィルター適用済みのシミュレーション
                    trade_res = 0
                    trade_type = "-"
                    
                    # 買い条件: AI75%以上 + 価格がSMA200より上 + ADX>20 + RSI<70
                    if p_buy >= threshold and price > sma and a > 20 and r < 70:
                        trade_res = 15 if actual == 1 else -15
                        trade_type = "BUY"
                    
                    # 売り条件: AI75%以上(Sell) + 価格がSMA200より下 + ADX>20 + RSI>30
                    elif p_sell >= threshold and price < sma and a > 20 and r > 30:
                        trade_res = 15 if actual == 0 else -15 # actual=0なら売り勝ち
                        trade_type = "SELL"
                        
                    total_pips += trade_res
                    pips_history.append(total_pips)
                    
                    if trade_type != "-":
                        trades.append({
                            "時間": row.name.strftime('%H:%M'),
                            "売買": trade_type,
                            "結果": "WIN" if trade_res > 0 else "LOSS",
                            "Pips": trade_res
                        })
                
                # グラフ描画
                color_pips = "#00cc66" if total_pips >= 0 else "#ff4b4b"
                st.markdown(f"<div style='text-align:center; font-size:1.5rem; font-weight:bold; color:{color_pips}'>期間損益: {total_pips:+} pips</div>", unsafe_allow_html=True)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=pips_history, mode='lines', name='Pips', line=dict(color='#2c3e50', width=3)))
                fig.update_layout(height=200, margin=dict(l=20, r=20, t=10, b=20), xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#eee'))
                st.plotly_chart(fig, use_container_width=True)
                
                # 取引履歴
                if trades:
                    st.write("▼ 直近のエントリー履歴")
                    st.dataframe(pd.DataFrame(trades).iloc[::-1], hide_index=True, use_container_width=True)
                else:
                    st.caption("※ 直近10時間では、安全基準（フィルター）を満たすエントリーはありませんでした。")

            else:
                st.warning("データ不足のため分析できません。")
