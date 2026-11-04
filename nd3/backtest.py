# Financial Intelligence homework 3: trading strategy, simulation and optimisation
# Daniel Sosna
# Trend-following strategy (EMA crossover) with stop loss, take profit and trading costs,
# tested on 1 year of AMD hourly bars from MetaTrader 5. Parameters are optimised by Sharpe ratio.

import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SHARES = 100  # shares bought on every buy
FEE = 5  # $ paid for every buy and every sell (set to 1000 to see the drops on the profit chart)
BARS_PER_YEAR = 252 * 8  # 252 trading days, 8 hourly bars per day


def load_data(filename):
    df = pd.read_csv(filename, sep="\t")
    # MT5 column names look like <CLOSE>, so remove the brackets
    df.columns = [c.strip("<>").lower() for c in df.columns]
    df["time"] = pd.to_datetime(df["date"] + " " + df["time"])
    return df.set_index("time")


def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


# Returns the profit curve and the list of trades (time, price, reason).
# A signal is taken at the bar's close, so the new position earns from the next bar.
def backtest(close, fast, slow, sl_pct, tp_pct):
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    buy = (fast_ema > slow_ema) & (fast_ema.shift() <= slow_ema.shift())
    sell = (fast_ema < slow_ema) & (fast_ema.shift() >= slow_ema.shift())

    prices = close.tolist()
    position = 0  # 1 = holding SHARES shares, 0 = no shares
    entry = 0  # price of the last buy
    profit = [0]
    trades = []

    for i in range(1, len(prices)):
        bar_profit = position * SHARES * (prices[i] - prices[i - 1])

        # buy signals are ignored while holding and sell signals while not holding,
        # so trades always go buy, sell, buy, sell
        reason = None
        if position == 1:
            if prices[i] <= entry * (1 - sl_pct):
                reason = "stop loss"
            elif prices[i] >= entry * (1 + tp_pct):
                reason = "take profit"
            elif sell.iloc[i]:
                reason = "EMA sell"
            if reason:
                position = 0
        elif buy.iloc[i]:
            reason = "EMA buy"
            position = 1
            entry = prices[i]

        if reason:
            bar_profit -= FEE
            trades.append((close.index[i], prices[i], reason))
        profit.append(bar_profit)

    profit = pd.Series(profit, index=close.index).cumsum()
    return profit, trades


# Sharpe ratio = average profit per bar / standard deviation of profit per bar,
# multiplied by sqrt(bars per year) to make it yearly.
# It shows how much profit the strategy makes per unit of risk (volatility). Higher is better.
def sharpe(profit):
    returns = profit.diff().dropna()
    if returns.std() == 0:
        return 0
    return returns.mean() / returns.std() * np.sqrt(BARS_PER_YEAR)


# Brute force: tries every combination from param_grid and keeps the one with the best objective
def optimize(objective, param_grid, close):
    best_params = None
    best_score = -np.inf
    for values in itertools.product(*param_grid.values()):
        params = dict(zip(param_grid.keys(), values))
        profit, _ = backtest(close, **params)
        score = objective(profit)
        if score > best_score:
            best_score = score
            best_params = params
    return best_params, best_score


def print_results(profit, trades, params, name):
    reasons = [t[2] for t in trades]
    print("\n" + name)
    print("Parameters:", params)
    print("Buys:", reasons.count("EMA buy"),
          " sells: EMA cross", reasons.count("EMA sell"),
          ", stop loss", reasons.count("stop loss"),
          ", take profit", reasons.count("take profit"))
    print(f"Trading costs: ${FEE * len(trades)} ({len(trades)} trades x ${FEE})")
    print(f"Net profit (after costs), {SHARES} shares per trade: ${profit.iloc[-1]:.2f}")
    print(f"Sharpe ratio (yearly): {sharpe(profit):.2f}")


MARKERS = {
    "EMA buy": ("^", "green", "Buy: fast EMA crossed above slow EMA"),
    "EMA sell": ("v", "red", "Sell: fast EMA crossed below slow EMA"),
    "stop loss": ("x", "black", "Sell: stop loss hit"),
    "take profit": ("*", "orange", "Sell: take profit hit"),
}


def plot_trades(close, trades, params, name):
    plt.figure(figsize=(12, 6))
    plt.plot(close, color="gray", label="Close")
    plt.plot(ema(close, params["fast"]), label=f"EMA({params['fast']})")
    plt.plot(ema(close, params["slow"]), label=f"EMA({params['slow']})")
    for reason, (marker, color, label) in MARKERS.items():
        points = [t for t in trades if t[2] == reason]
        plt.scatter([t[0] for t in points], [t[1] for t in points],
                    marker=marker, color=color, s=60, zorder=3, label=label)
    plt.title(name + ": trades")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.tight_layout()
    plt.show()


# profits is a dict {label: profit curve}, so one or several curves can be drawn
def plot_profit(profits, close, title):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    for label, profit in profits.items():
        ax1.plot(profit, label=label)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Profit ($)")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(close, color="gray", alpha=0.4, label="AMD close price")
    ax2.set_ylabel("Price ($)")
    ax2.legend(loc="lower right")

    plt.title(title)
    plt.tight_layout()
    plt.show()


def run(close, params, name):
    profit, trades = backtest(close, **params)
    print_results(profit, trades, params, name)
    plot_trades(close, trades, params, name)
    plot_profit({name: profit}, close, name + ": profit and AMD price")
    return profit


close = load_data("AMD_H1_202510271500_202610232300.csv")["close"]
print("AMD hourly bars:", len(close), "from", close.index[0], "to", close.index[-1])

# 1. strategy with parameters chosen by hand
default_params = {"fast": 12, "slow": 26, "sl_pct": 0.03, "tp_pct": 0.06}
default_profit = run(close, default_params, "Default parameters")

# 2. optimisation (the default parameters are in the grid, so the result can't be worse)
param_grid = {
    "fast": [5, 8, 12, 16, 20],
    "slow": [26, 40, 60, 80, 100],
    "sl_pct": [0.02, 0.03, 0.05],
    "tp_pct": [0.04, 0.06, 0.10],
}
best_params, best_sharpe = optimize(sharpe, param_grid, close)
print(f"\nBest of {np.prod([len(v) for v in param_grid.values()])} combinations by Sharpe:",
      best_params, f"(Sharpe {best_sharpe:.2f})")
best_profit = run(close, best_params, "Optimised parameters")

plot_profit({"Default parameters": default_profit, "Optimised parameters": best_profit},
            close, "Profit: default vs optimised parameters")
