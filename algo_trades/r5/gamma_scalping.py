import json
from typing import Any
import statistics
import math
import numpy as np
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, Dict, \
    List
import time


from statistics import NormalDist
from math import log, sqrt
'''
EVERYTHING BELOW HERE NEEDED FOR BT VISUALIZER
'''
import json
from typing import Any

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

'''
EVERYTHING ABOVE HERE NEEDED FOR BT VISUALIZER
'''
LIMITS = {
    'RAINFOREST_RESIN': 50,
    'KELP': 50,
    'SQUID_INK': 50,
    'CROISSANTS': 250,
    'JAMS': 350,
    'DJEMBES': 60,
    'PICNIC_BASKET1': 60,
    'PICNIC_BASKET2': 100,
    'VOLCANIC_ROCK': 400,
    'MAGNIFICENT_MACARONS': 75,
    'VOLCANIC_ROCK_VOUCHER_9500': 200,
    'VOLCANIC_ROCK_VOUCHER_9750': 200,
    'VOLCANIC_ROCK_VOUCHER_10000': 200,
    'VOLCANIC_ROCK_VOUCHER_10250': 200,
    'VOLCANIC_ROCK_VOUCHER_10500': 200,

}

PROD_TR_AR = {
    'RAINFOREST_RESIN': 10000,
    'KELP': 2000,
    'SQUID_INK': -1,
    'CROISSANTS': -1,
    'JAMS': -1,
    'DJEMBES': -1,
    'PICNIC_BASKET1': -1,
    'PICNIC_BASKET2': -1,
    'VOLCANIC_ROCK': -1,
    'MAGNIFICENT_MACARONS': 100,
    'VOLCANIC_ROCK_VOUCHER_9500': 200,
    'VOLCANIC_ROCK_VOUCHER_9750': 200,
    'VOLCANIC_ROCK_VOUCHER_10000': 200,
    'VOLCANIC_ROCK_VOUCHER_10250': 200,
    'VOLCANIC_ROCK_VOUCHER_10500': 200,
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
        self.prev_price_window = ppw
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
        if ppw != -1:
             self.get_prev_prices()
             self.smooth_price = self.savgol_filter_manual(self.prev_prices)
    def get_prev_prices(self) -> None:
        if self.prev_price_window < 0:
            return
        prev_prices = self.trader_data.get('prev_prices', [])

        if (len(prev_prices) > self.prev_price_window):
            prev_prices.pop(0)

        prev_prices.append(self.midprice)
        self.prev_prices = prev_prices
        self.trader_data['prev_prices'] = prev_prices
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
        best_buy = 0
        best_sell = 0

        if (self.od != {} and len(self.od.buy_orders)):
            for key, value in self.od.buy_orders.items():
                buy_sum += key * value
                buy_vol += value
            
            if buy_vol != 0:
                best_buy = int(math.ceil(buy_sum / buy_vol))
        else:
            best_buy = 0

        if (self.od != {} and len(self.od.sell_orders)):
            for key, value in self.od.sell_orders.items():
                sell_sum += key * value
                sell_vol += value
            if sell_vol != 0:
                best_sell = int(math.floor(sell_sum / sell_vol))
        else:
            best_sell = 0

        # TODO: Revaluate this
        midprice = (best_buy + best_sell) // 2 if best_buy and best_sell else -1

        return (best_buy, midprice, best_sell)

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


class Volcanic_Rock(ProductTrader):
    def __init__(self, state: TradingState, traderData: Dict, result: Dict[Symbol, List[Order]], ppw: int):
        super().__init__(state, traderData, 'VOLCANIC_ROCK', ppw)

        self.realized_vol = self.realized_vol(self.prev_prices)

    def realized_vol(self, prices, periods_per_year=252):
        if(len(prices) < 3):
            return 0
        prices = np.array(
            prices
        )
        log_returns = np.diff(np.log(prices))
        return np.std(log_returns, ddof=1) * np.sqrt(periods_per_year)

class Voucher(ProductTrader):
    def __init__(self, name:Symbol, state: TradingState, traderData: Dict, result: Dict[Symbol, List[Order]], ppw: int):
        super().__init__(state, traderData, name, ppw)

        strike = 0
        self.pos_lim = 200

        if (name == 'VOLCANIC_ROCK_VOUCHER_9500'):
            strike = 9500
        elif (name == 'VOLCANIC_ROCK_VOUCHER_9750'):
            strike = 9750
        elif (name == 'VOLCANIC_ROCK_VOUCHER_10000'):
            strike = 10_000
        elif (name == 'VOLCANIC_ROCK_VOUCHER_10250'):
            strike = 10_250
        elif (name == 'VOLCANIC_ROCK_VOUCHER_10500'):
            strike = 10_500
                
        self.strike = strike



@staticmethod
def black_scholes_call(spot, strike, time_to_expiry, volatility):
    if spot <= 0 or strike <= 0 or time_to_expiry <= 0 or volatility <= 0:
        return 0  # or float('nan') or raise ValueError("Invalid inputs to Black-Scholes")

    d1 = (
        log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
    ) / (volatility * sqrt(time_to_expiry))
    d2 = d1 - volatility * sqrt(time_to_expiry)
    call_price = spot * NormalDist().cdf(d1) - strike * NormalDist().cdf(d2)
    return call_price


@staticmethod
def delta(spot, strike, time_to_expiry, volatility):

    logger.print(f"spot={spot}, strike={strike}, TTE={time_to_expiry}, vol={volatility}")

    if spot <= 0 or strike <= 0 or time_to_expiry <= 0 or volatility <= 0:
        return float('nan')  # or float('nan') or raise ValueError("Invalid inputs to Black-Scholes")
    d1 = (
        log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
    ) / (volatility * sqrt(time_to_expiry))

    logger.print(f'norm dist = {NormalDist().cdf(d1)}')
    return NormalDist().cdf(d1)

@staticmethod
def gamma(spot, strike, time_to_expiry, volatility):
    d1 = (
        log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
    ) / (volatility * sqrt(time_to_expiry))
    return NormalDist().pdf(d1) / (spot * volatility * sqrt(time_to_expiry))


@staticmethod
def theta(spot, strike, time_to_expiry, volatility):
    d1 = (
    log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
    ) / (volatility * sqrt(time_to_expiry))

    return - (spot * NormalDist().pdf(d1) * volatility) / (2 * sqrt(time_to_expiry))

@staticmethod
def implied_volatility(
    call_price, spot, strike, time_to_expiry, max_iterations=200, tolerance=1e-10):
    low_vol = 0.01
    high_vol = 1.0
    volatility = (low_vol + high_vol) / 2.0  # Initial guess as the midpoint
    for _ in range(max_iterations):
        estimated_price = black_scholes_call(
            spot, strike, time_to_expiry, volatility
        )
        diff = estimated_price - call_price
        if abs(diff) < tolerance:
            break
        elif diff > 0:
            high_vol = volatility
        else:
            low_vol = volatility
        volatility = (low_vol + high_vol) / 2.0
    return volatility

@staticmethod
def execute_trade(result:Dict[Symbol, List[Order]], prod:ProductTrader, price:int, quantity:int):
        orders = [Order(prod.name, price, int(quantity))]
        if (quantity == 0):
            return
        if prod.name in result:
            result[prod.name].extend(orders)
        else:
            result[prod.name] = orders

class options_trader():
    def __init__(self, timestamp:int, v_array:List[Voucher], underlying:Volcanic_Rock):

        #TODO:::::: MUST CHANGE THIS 
        self.DTE = 5
        self.tte = ((1_000_000 * self.DTE) - timestamp) / (1_000_000 * 250)

        self.v_array = v_array
        self.underlying = underlying
        self.spot = self.underlying.midprice

    def gamma_scalp(self, timestamp: int, result: Dict[Symbol, List[Order]]):
        vol_gap_tol = 2

        # Skip until you have enough data
        if len(self.underlying.prev_prices) < 100:
            return result

        logger.print(f'underlying mid = {self.underlying.midprice}')
        strike_candidates = [9500, 9750, 10000, 10250, 10500]
        closest_strike = min(strike_candidates, key=lambda s: abs(s - self.underlying.midprice))
        atm_voucher = next(v for v in self.v_array if v.strike == closest_strike)

        # Balance other vouchers
        for v in self.v_array:
            if v != atm_voucher:
                v.balance(result)

        # Recalculate IV and vol ratio
        iv = implied_volatility(atm_voucher.midprice, self.underlying.midprice, atm_voucher.strike, self.tte)
        realized_vol = max(self.underlying.realized_vol, 1e-6)
        vol_ratio = iv / realized_vol


        # If vol is too high, sell vouchers and hedge
        logger.print(f"vol ratio = {vol_ratio:.2f}, iv = {iv:.4f}, realized_vol = {realized_vol:.4f}")
        if vol_ratio > 30:
            if vol_ratio > 40 and atm_voucher.curr_pos > -atm_voucher.pos_lim:
                qty = -atm_voucher.pos_lim - atm_voucher.curr_pos  # move to full short
                execute_trade(result, atm_voucher, atm_voucher.best_buy, qty)

                curr_delta = delta(self.underlying.midprice, atm_voucher.strike, self.tte, iv)
                hedge_qty = curr_delta * qty
                hedge_price = self.underlying.best_sell if hedge_qty > 0 else self.underlying.best_buy
                execute_trade(result, self.underlying, hedge_price, -hedge_qty)
            return result

        # Else: Normal gamma scalping
        curr_delta = delta(self.underlying.midprice, atm_voucher.strike, self.tte, iv)

        # Target a partial long voucher position (say 20%)
        target_voucher_pos = 0.1 * atm_voucher.pos_lim
        voucher_trade_qty = target_voucher_pos - atm_voucher.curr_pos

        if abs(voucher_trade_qty) >= 1:
            voucher_price = atm_voucher.best_sell if voucher_trade_qty > 0 else atm_voucher.best_buy
            execute_trade(result, atm_voucher, voucher_price, voucher_trade_qty)

            # Hedge only the change
            hedge_qty = curr_delta * voucher_trade_qty
            hedge_price = self.underlying.best_sell if hedge_qty > 0 else self.underlying.best_buy
            execute_trade(result, self.underlying, hedge_price, -hedge_qty)

        # Rebalance hedge if delta changed
        target_underlying_pos = curr_delta * atm_voucher.curr_pos
        delta_diff = target_underlying_pos - self.underlying.curr_pos
        logger.print(f"curr_delta = {curr_delta:.4f}, delta_diff = {delta_diff:.2f}, iv = {iv:.4f}")

        if abs(delta_diff) >= 2:
            hedge_price = self.underlying.best_sell if delta_diff > 0 else self.underlying.best_buy
            execute_trade(result, self.underlying, hedge_price, delta_diff)

        return result


class Trader:

    # def __init__ (self, params:Dict = {}):
    #     self.params = params
    def run(self, state: TradingState):

        if state.traderData == '' or state.traderData == None:
            traderData = {
                'RAINFOREST_RESIN': {},
                'KELP': {},
                'SQUID_INK': {},
                'CROISSANTS': {},
                'JAMS': {},
                'DJEMBES': {},
                'PICNIC_BASKET1': {},
                'PICNIC_BASKET2': {},
                'VOLCANIC_ROCK': {}
            }
        else:
            traderData = json.loads(state.traderData)

        result: Dict[str, List[Order]] = {}

        vr = Volcanic_Rock(state, traderData, result, 101)
        voucher_9500 = Voucher('VOLCANIC_ROCK_VOUCHER_9500',state, {}, result, -1)
        voucher_9750 = Voucher('VOLCANIC_ROCK_VOUCHER_9750', state, {}, result, -1)
        voucher_10000 = Voucher('VOLCANIC_ROCK_VOUCHER_10000', state, {}, result, -1)
        voucher_10250 = Voucher('VOLCANIC_ROCK_VOUCHER_10250', state, {}, result, -1)
        voucher_10500 = Voucher('VOLCANIC_ROCK_VOUCHER_10500', state, {}, result, -1)

        v_arr = [voucher_9500, voucher_9750, voucher_10000, voucher_10250, voucher_10500]

        money_printer = options_trader(state.timestamp, v_arr, vr)

        money_printer.gamma_scalp(state.timestamp, result)



        traderData = json.dumps(traderData)
        conversions = 0
        logger.flush(state, result, conversions, traderData)

        return result, conversions, traderData
    




