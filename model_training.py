# import libraries
from __future__ import annotations
import datetime
import os
import sys
from pprint import pprint

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import config_tickers
from agents.stablebaselines3.models import DRLAgent
from main import check_and_make_directories
from meta.data_processor import DataProcessor
from meta.env_stock_trading.env_stocktrading import StockTradingEnv
from meta.preprocessor.preprocessors import data_split
from meta.preprocessor.preprocessors import FeatureEngineer
from meta.preprocessor.yahoodownloader import YahooDownloader
from plot import backtest_plot
from plot import backtest_stats
from plot import get_baseline
from plot import get_daily_return

import itertools

from config import (
    DATA_SAVE_DIR,
    TRAINED_MODEL_DIR,
    TENSORBOARD_LOG_DIR,
    RESULTS_DIR,

)

from settrade.openapi import Investor

investor = Investor(
    app_id="pZ2aU14TBcrHe6QY",
    app_secret="QbB50GwPfd5yAi2wY9zy8Na4aOKwwZSs3Rqrq/hlFH0=",
    broker_id="025",
    app_code="ALGO_EQ",
    is_auto_queue=False)

equity = investor.Equity(account_no="8301176")
account_info = equity.get_account_info()


def get_thai_stocks(stock, freq='1d'):
    result = investor.MarketData().get_candlestick(stock, freq)
    df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
    df['date'] = result['data']['time']
    df['open'] = result['data']['open']
    df['high'] = result['data']['high']
    df['close'] = result['data']['close']
    df['low'] = result['data']['low']
    df['volume'] = result['data']['volume']
    df['date'] = pd.to_datetime(df['date'], unit='s')
    df['date'] = df.date.dt.date
    # df.index = df.date
    #    df['date'] = df.date
    # df = df.drop(columns='date')
    df['tic'] = "AAPL"

    return df


def train_model(ticker=0, df=0):
    check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])

    print("HEREEEEEEEEEEE ")

    if len(df) == 0:
        df = get_thai_stocks(ticker)
        # df = YahooDownloader(start_date="2009-01-01", end_date=str(datetime.datetime.now().date()),
        #                      ticker_list=[ticker], ).fetch_data()

    # df['tic'] = 'AAPL'
    print(df.head())
    df.sort_values(["date", "tic"], ignore_index=True).head()

    print("HEREEEEEEEEEEE ")
    fe = FeatureEngineer(use_technical_indicator=True, tech_indicator_list=config.INDICATORS, use_vix=True,
                         use_turbulence=False, user_defined_feature=False)

    processed = fe.preprocess_data(df)

    list_ticker = processed["tic"].unique().tolist()
    list_date = list(pd.date_range(processed["date"].min(), processed["date"].max()).astype(str))
    combination = list(itertools.product(list_date, list_ticker))

    train = processed
    trade = processed
    print(f"len(train): {len(train)}")
    print(f"len(trade): {len(trade)}")
    print(f"train.tail(): {train.tail()}")
    print(f"trade.head(): {trade.head()}")
    print(f"config.INDICATORS: {config.INDICATORS}")

    stock_dimension = len(train.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(config.INDICATORS) * stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension

    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": config.INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4,
    }

    e_train_gym = StockTradingEnv(df=train, **env_kwargs)
    df_res = pd.DataFrame(columns=['total_profit', 'win_ratio', 'p'])

    env_train, _ = e_train_gym.get_sb_env()
    print(f"type(env_train): {type(env_train)}")

    agent = DRLAgent(env=env_train)
    model = agent.get_model("ddpg")
    trained_model = agent.train_model(model=model, tb_log_name='ddpg', total_timesteps=500)
    trained_model.save("trained_models/{}_ticker".format(ticker))
    # print("Loading Model")
    # trained_a2c = model_a2c.load("a2c_first_iteration.zip")
    # trade = data_split(processed_full, '2020-07-01','2021-10-31')
    e_trade_gym = StockTradingEnv(df=trade, turbulence_threshold=70, risk_indicator_col="vix", **env_kwargs)

    print(f"trade.head(): {trade.head()}")
    df_account_value, df_actions = DRLAgent.DRL_prediction(model=trained_model, environment=e_trade_gym)

    print(f"df_account_value.shape: {df_account_value.shape}")
    print(f"df_account_value.tail(): {df_account_value.tail()}")
    print(f"df_actions.head(): {df_actions.head()}")

    # df_actions.to_csv("results/{}_act.csv".format(ticker))
    from meta.preprocessor.preprocessors import get_ratios
    win_ratio, total_profit, p = get_ratios(df)
    df_res.loc[0] = [total_profit, win_ratio, p]
    df_res.to_csv("results/{}_account.csv".format(ticker))

    return df_account_value, df_actions
