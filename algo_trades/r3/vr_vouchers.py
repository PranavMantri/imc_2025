import json
from typing import Any, List, Dict
import statistics
import math
import numpy as np
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        lo, hi = 0, min(len(value), max_length)
        out = ""

        while lo <= hi:
            mid = (lo + hi) // 2

            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."

            encoded_candidate = json.dumps(candidate)

            if len(encoded_candidate) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1

        return out
logger = Logger()


class OptionsTrader:
    def __init__(self, name, strike, expiration, option_type):
        self.name = name
        self.strike = strike
        self.expiration = expiration  # in years
        self.option_type = option_type
        self.pos_lim = 200
        self.max_trade_size = 10
        self.r = 0.01

    def implied_volatility(self, market_price, S, K, T):
        if market_price < 1e-5:
            return float('nan')
        low, high = 1e-6, 3.0
        for _ in range(100):
            mid = (low + high) / 2
            price = self.black_scholes_call_price(S, K, T, self.r, mid)
            if abs(price - market_price) < 1e-5:
                return mid
            if price > market_price:
                high = mid
            else:
                low = mid
        return (low + high) / 2

    def black_scholes_call_price(self, S, K, T, r, sigma):
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            return 0.0
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2)*T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        Nd1 = 0.5 * (1 + math.erf(d1 / math.sqrt(2)))
        Nd2 = 0.5 * (1 + math.erf(d2 / math.sqrt(2)))
        return S * Nd1 - K * math.exp(-r*T) * Nd2

    def compute_m_t(self, K, S, T):
        return np.log(K / S) / np.sqrt(T)


class VR_VoucherTrader:
    def __init__(self, name, strike, state: TradingState, trader_memory):
        self.name = name
        self.strike = strike
        #self.expiration = (1000000-state.timestamp + 1000000*(5))/365000000
        self.expiration = 1 / 365
        self.state = state

        self.pos_lim = 200
        self.max_trade_size = 10
        self.option_type = "call"
        self.r = 0.01

        self.od = state.order_depths.get(name, OrderDepth())
        self.curr_pos = state.position.get(name, 0)
        self.best_bid = max(self.od.buy_orders.keys()) if self.od.buy_orders else 0
        self.best_ask = min(self.od.sell_orders.keys()) if self.od.sell_orders else 0

        self.timestamp = state.timestamp
        self.memory = trader_memory.setdefault(name, {"last_price": 0})

    def compute_m_t(self, K, S, T):
        return math.log(K / S) / math.sqrt(T)

    def norm_cdf(self, x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def black_scholes_call_price(self, S, K, T, r, sigma):
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            return 0.0
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return S * self.norm_cdf(d1) - K * math.exp(-r * T) * self.norm_cdf(d2)

    def implied_volatility(self, market_price, S, K, T):
        low, high = 1e-6, 4.0
        for _ in range(20):
            mid = (low + high) / 2
            price = self.black_scholes_call_price(S, K, T, self.r, mid)
            if abs(price - market_price) < 1e-4:
                return mid
            if price > market_price:
                high = mid
            else:
                low = mid
        return (low + high) / 2

    def trade(self, result: dict[str, list[Order]]) -> dict[str, list[Order]]:
        rock_od = self.state.order_depths.get("VOLCANIC_ROCK", OrderDepth())
        if not (rock_od.buy_orders and rock_od.sell_orders):
            return result

        S = (max(rock_od.buy_orders.keys()) + min(rock_od.sell_orders.keys())) / 2

        all_mtv = []
        for symbol in self.state.order_depths:
            if "VOLCANIC_ROCK_VOUCHER" not in symbol:
                continue
            od = self.state.order_depths[symbol]
            if od.buy_orders and od.sell_orders:
                strike = int(symbol.split("_")[-1])
                mid = (max(od.buy_orders.keys()) + min(od.sell_orders.keys())) / 2
                iv = self.implied_volatility(mid, S, strike, self.expiration)
                m_val = self.compute_m_t(strike, S, self.expiration)
                if not math.isnan(iv):
                    all_mtv.append((m_val, iv))

        if len(all_mtv) < 3:
            return result

        x_vals, y_vals = zip(*all_mtv)
        coeffs = self.polyfit_2(x_vals, y_vals)
        a, b, c = coeffs
        fitted_iv_fn = lambda m: a * m * m + b * m + c

        base_iv = fitted_iv_fn(0)
        if base_iv > 0.06:
            return result

        if not (self.best_bid and self.best_ask):
            return result

        mid_price = (self.best_bid + self.best_ask) / 2
        actual_iv = self.implied_volatility(mid_price, S, self.strike, self.expiration)
        m_t = self.compute_m_t(self.strike, S, self.expiration)
        fitted_iv = fitted_iv_fn(m_t)
        error = actual_iv - fitted_iv

        threshold = 5
        trade_qty = min(int(abs(error) / threshold), self.max_trade_size)

        # 📈 Main trade logic
        if error > threshold:
            qty = min(trade_qty, self.pos_lim + self.curr_pos)
            if qty > 0:
                result.setdefault(self.name, []).append(Order(self.name, self.best_bid, -qty))
        elif error < -threshold:
            qty = min(trade_qty, self.pos_lim - self.curr_pos)
            if qty > 0:
                result.setdefault(self.name, []).append(Order(self.name, self.best_ask, qty))

        # 🔁 Unwind logic: if position is held but signal reversed
        if self.curr_pos > 0 and error > -threshold:
            unwind_qty = min(self.max_trade_size, self.curr_pos)
            result.setdefault(self.name, []).append(Order(self.name, self.best_bid, -unwind_qty))
        elif self.curr_pos < 0 and error < threshold:
            unwind_qty = min(self.max_trade_size, -self.curr_pos)
            result.setdefault(self.name, []).append(Order(self.name, self.best_ask, unwind_qty))

        # 🧠 Track PnL context
        self.memory["last_price"] = mid_price
        return result

    def polyfit_2(self, x, y):
        n = len(x)
        x1 = sum(x)
        x2 = sum(xi**2 for xi in x)
        x3 = sum(xi**3 for xi in x)
        x4 = sum(xi**4 for xi in x)
        y1 = sum(y)
        xy = sum(xi*yi for xi, yi in zip(x, y))
        x2y = sum(xi**2*yi for xi, yi in zip(x, y))

        det = n*x2*x4 + 2*x1*x2*x3 - x2**3 - n*x3**2 - x1**2*x4
        a_num = y1*x2*x4 + x1*xy*x3 + x2*x2y*n - x2**2*y1 - x3**2*n - x1*x4*xy
        b_num = n*xy*x4 + y1*x2*x3 + x2*x2y*x1 - x2*xy*x2 - x3*x2y*n - x1*x4*y1
        c_num = n*x2*x2y + x1*xy*x2 + x1*x3*y1 - x2**2*y1 - x3*x2*x1 - n*xy*x3
        return (a_num / det, b_num / det, c_num / det)


class Trader:
    def run(self, state: TradingState):
        traderData = json.loads(state.traderData) if state.traderData else {}
        result: Dict[str, List[Order]] = {}

        option_configs = {
            "VOLCANIC_ROCK_VOUCHER_9500": 9500,
            "VOLCANIC_ROCK_VOUCHER_9750": 9750,
            "VOLCANIC_ROCK_VOUCHER_10000": 10000,
            "VOLCANIC_ROCK_VOUCHER_10250": 10250,
            "VOLCANIC_ROCK_VOUCHER_10500": 10500,
        }

        for name, strike in option_configs.items():
            VR_VoucherTrader(name, strike, state, traderData).trade(result)

        logger.flush(state, result, 0, json.dumps(traderData))
        return result, 0, json.dumps(traderData)





#eof