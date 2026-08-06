import datetime
import json
import os
import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ==================== 1. 機器人系統設定 ====================
INITIAL_CAPITAL = 500000.0  # 初始模擬資金 50 萬
PORTFOLIO_FILE = 'ai_portfolio.json'  # 持倉與資金記錄檔
# 監控標的：以台股市值型/債券型 ETF 為例（如 0050, 0056, 00679B, 2330）
SYMBOLS = ['0050.TW', '0056.TW', '00679B.TW', '2330.TW']


# ==================== 2. 帳戶與紀錄管理 ====================
def load_portfolio():
  """載入帳戶紀錄，若無則建立新帳戶"""
  if os.path.exists(PORTFOLIO_FILE):
    with open(PORTFOLIO_FILE, 'r', encoding='utf-8') as f:
      return json.load(f)
  else:
    portfolio = {
        'cash': INITIAL_CAPITAL,
        'holdings': {s: 0 for s in SYMBOLS},  # 持有股數
        'history': [],  # 資產變化歷史記錄
    }
    save_portfolio(portfolio)
    return portfolio


def save_portfolio(portfolio):
  with open(PORTFOLIO_FILE, 'w', encoding='utf-8') as f:
    json.dump(portfolio, f, ensure_ascii=False, indent=4)


# ==================== 3. 最佳化策略模型 (AI/量化) ====================
def calculate_optimal_weights(symbols):
  """使用歷史報酬與共變異數矩陣計算「最大夏普比率」的最佳化資產權重"""
  # 抓取過去一年的日收盤價
  data = yf.download(symbols, period='1y')['Close']
  returns = data.pct_change().dropna()

  mean_returns = returns.mean() * 252  # 年化報酬率
  cov_matrix = returns.cov() * 252  # 年化共變異數

  num_assets = len(symbols)
  num_portfolios = 5000  # 蒙地卡羅模擬最佳化
  results = np.zeros((3, num_portfolios))
  weights_record = []

  for i in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)  # 權重相加為 1
    weights_record.append(weights)

    # 組合預期回報與波動度
    portfolio_return = np.sum(mean_returns * weights)
    portfolio_stddev = np.sqrt(
        np.dot(weights.T, np.dot(cov_matrix, weights))
    )

    # 紀錄回報、波動度與夏普比率 (假設無風險利率 1.5%)
    results[0, i] = portfolio_return
    results[1, i] = portfolio_stddev
    results[2, i] = (portfolio_return - 0.015) / portfolio_stddev

  # 尋找夏普比率最高的組合
  max_sharpe_idx = np.argmax(results[2])
  optimal_weights = weights_record[max_sharpe_idx]

  weight_dict = {
      symbols[i]: round(optimal_weights[i], 4) for i in range(num_assets)
  }
  return weight_dict, data.iloc[-1].to_dict()


# ==================== 4. 模擬交易執行與再平衡 ====================
def run_simulation():
  portfolio = load_portfolio()
  print('--- 正在執行 AI 策略最佳化計算 ---')

  optimal_weights, current_prices = calculate_optimal_weights(SYMBOLS)
  print(f'AI 最佳化建議權重: {optimal_weights}')

  # 計算當前總資產 (現金 + 持有股票總值)
  total_stock_value = sum(
      portfolio['holdings'][s] * current_prices[s] for s in SYMBOLS
  )
  total_asset = portfolio['cash'] + total_stock_value

  print(f'當前帳戶總資產: ${total_asset:,.2f} TWD')

  # 執行模擬再平衡 (Rebalancing)
  new_holdings = {}
  used_cash = 0

  for s in SYMBOLS:
    target_value = total_asset * optimal_weights[s]
    target_shares = int(target_value // current_prices[s])  # 模擬股數整數買入
    new_holdings[s] = target_shares
    used_cash += target_shares * current_prices[s]

  remaining_cash = total_asset - used_cash

  # 更新帳戶狀態
  portfolio['cash'] = remaining_cash
  portfolio['holdings'] = new_holdings

  # 記錄本日觀察數據
  today_record = {
      'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
      'total_asset': round(total_asset, 2),
      'return_pct': round(
          ((total_asset - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100, 2
      ),
      'cash': round(remaining_cash, 2),
      'holdings': new_holdings,
      'prices': {s: round(current_prices[s], 2) for s in SYMBOLS},
  }
  portfolio['history'].append(today_record)

  save_portfolio(portfolio)

  print('\n=== 自動模擬交易完成 ===')
  print(f'最新總資產: ${total_asset:,.2f}')
  print(f'總累計報酬率: {today_record["return_pct"]}%')
  print(f'持倉股數狀況: {new_holdings}')
  print(f'剩餘現金: ${remaining_cash:,.2f}')


if __name__ == '__main__':
  run_simulation()
