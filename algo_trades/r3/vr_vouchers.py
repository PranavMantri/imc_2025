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

LIMITS = {
    'RAINFOREST_RESIN' : 50,
    'KELP' : 50,
    'SQUID_INK' : 50,
    'CROISSANTS' : 250,
    'JAMS' : 350,
    'DJEMBES' : 60,
    'PICNIC_BASKET1' : 60,
    'PICNIC_BASKET2' :100,
    'VOLCANIC_ROCK' : 400,
}

PROD_TR_AR = {
    'RAINFOREST_RESIN' : 10000,
    'KELP' : 2000,
    'SQUID_INK' : -1,
    'CROISSANTS' : -1,
    'JAMS' : -1,
    'DJEMBES' : -1,
    'PICNIC_BASKET1' : -1,
    'PICNIC_BASKET2' : -1,
    'VOLCANIC_ROCK' : 10280
}

class ProductTrader:

    def __init__(self, state: TradingState, traderDict: Dict, name: str, ppw: int):

        # declare ALL variables here first
        # yer
        self.name = name
        self.od = state.order_depths.get(self.name, {})
        self.trader_data: Dict = traderDict.get(self.name, {})

        # price vars
        self.prev_prices = []
        self.prev_price_window = ppw if ppw != 0 else 101
        self.best_buy = 0
        self.best_sell = 0
        self.midprice = 0
        self.trade_around = PROD_TR_AR[self.name]
        self.gap_trigger = 3

        # volume vars
        self.curr_pos = state.position.get(self.name, 0)
        self.pos_lim = LIMITS[self.name]
        self.buying_power = self.pos_lim - self.curr_pos
        self.selling_power = self.pos_lim + self.curr_pos
        self.market_take_buy_vol = 10
        self.market_take_sell_vol = 10
        self.market_make_buy_vol = 10
        self.market_make_sell_vol = 10

        ## probably don't need these
        # self.buyVol_this_timestamp = 0
        # self.sellVol_this_timestamp = 0
        ################################

        # call calculations functions
        (self.best_buy, self.midprice, self.best_sell) = self.calc_vwaps()
        self.gap = self.best_sell - self.best_buy if self.best_buy and self.best_sell else -1

        ######################################
        # Pull out whatever from trader_data #
        ######################################
        self.get_prev_prices()
        self.smooth_price = self.savgol_filter_manual(self.prev_prices)

        ########################
        # SMOOOOOTTTH OPERATOR #
        ########################

    def savgol_filter_manual(self, a: list, window_size: int = 101, poly_order: int = 2) -> float:
        """
        Applies Savitzky-Golay filtering to the input list of midprices.

        Parameters:
            y (list): List of midprices, length must equal window_size.
            window_size (int): Must be odd. The number of points to consider in the window.
            poly_order (int): The order of the polynomial to fit (e.g., 2 = quadratic).

        Returns:
            float: The smoothed value at the center of the window.
        """
        "Length of input must match window_size"
        if len(a) < self.prev_price_window:
            return self.midprice

        y = a[-self.prev_price_window:]
        assert window_size % 2 == 1, "Window size must be odd"

        half = window_size // 2
        x = np.arange(-half, half + 1)  # e.g., [-50, ..., 0, ..., +50]
        X = np.vander(x, poly_order + 1, increasing=True)  # Vandermonde matrix
        X_pinv = np.linalg.pinv(X)  # Pseudo-inverse for least squares

        coeffs = X_pinv @ np.array(y)
        smoothed_val = coeffs[0]  # Value at center of polynomial (x=0)

        return smoothed_val

    def calc_vwaps(self) -> tuple[int, int, int]:
        """
        Returns the (best_buy, midprice, best_sell) using volume weighted average products

        does vwap on buy orders --> best buy
        does vwap on sell orders --> best sell
        avg buy/sell -> mid
        """
        buy_sum = 0
        buy_vol = 0
        sell_sum = 0
        sell_vol = 0

        if (self.od != {} and len(self.od.buy_orders)):
            for key, value in self.od.buy_orders.items():
                buy_sum += key * value
                buy_vol += value
            best_buy = int(math.ceil(buy_sum / buy_vol))
        else:
            best_buy = 0

        if (self.od != {} and len(self.od.sell_orders)):
            for key, value in self.od.sell_orders.items():
                sell_sum += key * value
                sell_vol += value
            best_sell = int(math.floor(sell_sum / sell_vol))
        else:
            best_sell = 0

        # TODO: Revaluate this
        midprice = (best_buy + best_sell) // 2 if best_buy and best_sell else -1

        return (best_buy, midprice, best_sell)

    # this returns the last prev_price_window prices including this timestamp
    def get_prev_prices(self) -> None:
        prev_prices = self.trader_data.get('prev_prices', [])

        if (len(prev_prices) > self.prev_price_window):
            prev_prices.pop(0)

        prev_prices.append(self.midprice)
        self.prev_prices = prev_prices
        self.trader_data['prev_prices'] = prev_prices

    def balance(self, result: Dict[str, List[Order]]):
        orders: List[Order] = []
        if (self.curr_pos != 0):

            ##TODO: buy/sell at the farthest price from trade-around for balancing

            if (self.curr_pos < 0):
                orders.append(Order(self.name, self.best_buy + 1, -self.curr_pos))
                self.buying_power += self.curr_pos
            else:
                orders.append(Order(self.name, self.best_sell - 1, -self.curr_pos))
                self.selling_power += self.curr_pos

        if self.name in result:
            result[self.name].extend(orders)
        else:
            result[self.name] = orders

    # ABSOLUTE VALUES OF VOLUME
    def short(self, price: int, abs_quantity: int, result: Dict[str, List[Order]]):
        if self.name in result:
            result[self.name].extend([Order(self.name, price, -abs_quantity)])
        else:
            result[self.name] = [Order(self.name, price, -abs_quantity)]

    def long(self, price: int, abs_quantity: int, result: Dict[str, List[Order]]):
        if self.name in result:
            result[self.name].extend([Order(self.name, price, abs_quantity)])
        else:
            result[self.name] = [Order(self.name, price, abs_quantity)]

    def market_take(self, result: Dict[str, List[Order]]):
        orders: List[Order] = []

        # TODO: IMPLEMENT SMARTER ORDER VOLUME CHOICE
        i = 0
        for key, val in self.od.buy_orders.items():
            i += 1
            if (i > 3):
                break
            if (key > self.trade_around):
                orders.append(Order(self.name, key, -self.market_take_sell_vol))
                self.selling_power -= self.market_take_sell_vol

        j = 0
        for key, val in self.od.sell_orders.items():
            j += 1
            if (j > 3):
                break
            if (key < self.trade_around):
                orders.append(Order(self.name, key, self.market_take_buy_vol))
                self.buying_power -= self.market_take_buy_vol

        if self.name in result:
            result[self.name].extend(orders)
        else:
            result[self.name] = orders

    def market_make(self, result: Dict[str, List[Order]]):
        orders: List[Order] = []

        # TODO: IMPLEMENT SMARTER ORDER VOLUME CHOICE

        for backoff in range(1, 20):

            bv_ = int(self.market_make_buy_vol / (backoff))
            sv_ = int(self.market_make_sell_vol / (backoff))

            logger.print(f'bv_ {bv_} > self.buying_power {self.buying_power}')
            logger.print(f'sv_ {sv_} > self.buying_power {self.buying_power}')

            if bv_ > self.buying_power or sv_ > self.selling_power:
                logger.print(f"backoff {backoff}")
                continue

            logger.print(f'sv_ {self.gap} > self.buying_power {self.gap_trigger}')

            if self.gap >= self.gap_trigger:
                logger.print(f"backoff {backoff}")

                orders.append(Order(self.name, self.best_buy + 1, bv_))
                orders.append(Order(self.name, self.best_sell - 1, -sv_))
                break

        if self.name in result:
            result[self.name].extend(orders)
        else:
            result[self.name] = orders

    def balance(self, result: Dict[str, List[Order]]):
        orders: List[Order] = []
        if (self.curr_pos != 0):

            ##TODO: buy/sell at the farthest price from trade-around for balancing

            if (self.curr_pos < 0):
                orders.append(Order(self.name, self.best_buy + 1, -self.curr_pos))
                # confusing, yes, but position < 0, and we are BUYING so we want to reduce our buying power
                self.buying_power += self.curr_pos
            else:
                orders.append(Order(self.name, self.best_sell - 1, -self.curr_pos))
                # confusing, yes, but position > 0, and we are SELLING so we want to reduce our selling power
                self.selling_power -= self.curr_pos

        if self.name in result:
            result[self.name].extend(orders)
        else:
            result[self.name] = orders


class VRTrader(ProductTrader):
    def __init__(self, state: TradingState, traderData: Dict, result:Dict[Symbol, List[Order]], ppw:int):
        super().__init__(state, traderData, 'VOLCANIC_ROCK', ppw)

        # GRID SEARCHED
        self.mm_vol_r = 0.5
        self.mt_bv = 15
        self.mt_sv = 15
        self.gap_trigger = 4
        self.best_delta = 1

        # NOT GRID SEARCHED (but necessary for product trader)
        self.market_make_buy_vol = 120
        self.market_make_sell_vol = 120


class OptionsTrader:
    def __init__(self, name, strike, expiration, option_type):
        self.name = name
        self.strike = strike
        self.expiration = expiration  # in years
        self.option_type = option_type
        self.pos_lim = 200
        self.max_trade_size = 100
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
        if base_iv > 0.6:
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

        vr = VRTrader(state, traderData, result, 101)
        #vr.balance(result)
        #vr.market_make(result)
        vr.market_take(result)

        logger.flush(state, result, 0, json.dumps(traderData))
        return result, 0, json.dumps(traderData)





#eof