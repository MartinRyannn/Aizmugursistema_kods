import tpqoa
from datetime import datetime, timedelta
import pandas as pd
import os
import sys


class ConTrader(tpqoa.tpqoa):
    def __init__(self, config_file, instrument, bar_length="3S"):
        super().__init__(config_file)
        self.instrument = instrument
        self.bar_length = pd.to_timedelta(bar_length)
        self.tick_data = pd.DataFrame()
        self.last_bar = pd.Timestamp(datetime.utcnow(), tz="UTC")
        self.last_print_time = datetime.utcnow()
        self.csv_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{self.instrument}_candles.csv")
        self.initialize_csv()

    def initialize_csv(self):
        try:
            pd.read_csv(self.csv_file)
        except FileNotFoundError:
            columns = ['timestamp', 'open', 'high', 'low', 'close']
            empty_df = pd.DataFrame(columns=columns)
            empty_df.to_csv(self.csv_file, index=False)

    def on_success(self, time, bid, ask):
        recent_tick = pd.to_datetime(time).tz_convert("UTC")
        df = pd.DataFrame({self.instrument: (ask + bid) / 2}, index=[recent_tick])
        self.tick_data = pd.concat([self.tick_data, df])

        if datetime.utcnow() - self.last_print_time >= timedelta(seconds=3):
            self.resample_and_save()

    def resample_and_save(self):
        resampled_data = self.tick_data.resample('3S', label="right").ohlc().ffill().iloc[:-1]
        if not resampled_data.empty:
            latest_candle = resampled_data.iloc[[-1]]
            latest_candle.columns = latest_candle.columns.droplevel(0)
            latest_candle['timestamp'] = latest_candle.index
            latest_candle = latest_candle[['timestamp', 'open', 'high', 'low', 'close']]
            latest_candle.to_csv(self.csv_file, mode='a', header=False, index=False)
            self.last_print_time = datetime.utcnow()
            self.tick_data = self.tick_data[self.tick_data.index > latest_candle['timestamp'].iloc[-1]]


trader = ConTrader(os.path.join(os.path.dirname(os.path.abspath(__file__)), "oanda.cfg"), "XAU_USD", "3S")
trader.stream_data("XAU_USD")
