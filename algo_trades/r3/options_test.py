import json
import math
from typing import Dict, List
from typing import Any, List, Dict
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
import statistics

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
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."
logger = Logger()

#underlying asset base
class ProductTrader:
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

    def __init__(self, name):

        self.name = name
        self.od = Dict[Symbol, OrderDepth]
        self.curr_pos = 0
        self.curr_sell_vol = 0
        self.curr_buy_vol = 0

        # Market Making
        self.pos_lim = 0
        self.best_sell = 0
        self.best_buy = 0
        self.mm_bv = 0
        self.mm_sv = 0

        # Market Taking
        self.trade_around = 0
        self.mt_bv = 0
        self.mt_sv = 0

        # Moving Avg
        self.midprice = 0
        self.big_window_size = 100
        self.ma_bv = 20
        self.ma_sv = -20

        # big mean - small mean
        self.prev_diff = 0

    def market_take(self, result: Dict[str, List[Order]]):
        orders: List[Order] = []

        # TODO: IMPLEMENT SMARTER ORDER VOLUME CHOICE
        i = 0
        for key, val in self.od.buy_orders.items():
            i += 1
            if (i > 3):
                break
            if (key > self.trade_around):
                orders.append(Order(self.name, key, self.mt_sv))
                self.curr_sell_vol += -val

        j = 0
        for key, val in self.od.sell_orders.items():
            j += 1
            if (j > 3):
                break
            if (key < self.trade_around):
                orders.append(Order(self.name, key, self.mt_bv))
                self.curr_buy_vol += -val

        if self.name in result:
            result[self.name].extend(orders)
        else:
            result[self.name] = orders

    def market_make(self, result: Dict[str, List[Order]]):
        orders: List[Order] = []

        # TODO: IMPLEMENT SMARTER ORDER VOLUME CHOICE

        # this conditional assumes we market take the position we are missing out on here
        # if (prod.best_buy < prod.trade_around and prod.best_sell > prod.trade_around):

        for backoff in range(1, 20):
            '''
            bv_ = max(int(0.8 * b_bank), 20)
            sv_ = min(int(0.8 * s_bank), -20)
            '''

            bv_ = int(self.mm_bv / (backoff))
            sv_ = int(self.mm_sv / (backoff))
            if (
                    self.gap >= 3 and
                    self.curr_pos < (self.pos_lim - bv_ - self.curr_buy_vol) and
                    self.curr_pos > -(self.pos_lim + sv_ + self.curr_sell_vol)):
                orders.append(Order(self.name, self.best_buy + 1, bv_))
                orders.append(Order(self.name, self.best_sell - 1, sv_))
                break

        # TODO: Decide to delete pls
        # for i in range(1,10):

        #     '''
        #     bv_ = max(int(0.8 * b_bank), 20)
        #     sv_ = min(int(0.8 * s_bank), -20)
        #     '''
        #     backoff_b = b_bank/self.gap
        #     backoff_s = s_bank/self.gap

        #     bv_ = int(self.mm_bv - i*backoff_b)
        #     sv_ = int(self.mm_sv + i*backoff_s)
        #     if (self.gap >= 3 and self.curr_pos < (self.pos_lim - bv_- self.curr_buy_vol) and self.curr_pos > -(self.pos_lim + sv_ + self.curr_sell_vol)):
        #         orders.append(Order(self.name, self.best_buy + 1, bv_))
        #         orders.append(Order(self.name, self.best_sell - 1 , sv_))
        #         break
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
                self.curr_buy_vol += self.curr_pos
            else:
                orders.append(Order(self.name, self.best_sell - 1, -self.curr_pos))
                self.curr_sell_vol += self.curr_pos

        if self.name in result:
            result[self.name].extend(orders)
        else:
            result[self.name] = orders

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

        if (len(self.od.buy_orders)):
            for key, value in self.od.buy_orders.items():
                buy_sum += key * value
                buy_vol += value
            best_buy = int(math.ceil(buy_sum / buy_vol))
        else:
            best_buy = 0

        if (len(self.od.sell_orders)):
            for key, value in self.od.sell_orders.items():
                sell_sum += key * value
                sell_vol += value
            best_sell = int(math.floor(sell_sum / sell_vol))
        else:
            best_sell = 0

        # TODO: Revaluate this
        midprice = (best_buy + best_sell) // 2 if best_buy and best_sell else -1

        return (best_buy, midprice, best_sell)

    def update_td(self, trader_data):
        """
        Implement this in child and ONLY update trader_data[self.name]
        """
        pass


class VolanicRockTrader(ProductTrader):
    def __init__(self, state: TradingState, traderData: Dict):
        super().__init__('CROISSANTS')
        self.od = state.order_depths[self.name]
        self.pos_lim = 400

        # Using "wvap" to find ideal best buy/sell
        (self.best_buy, self.midprice, self.best_sell) = self.calc_vwaps()

        self.gap = (self.best_sell - self.best_buy) if (self.best_buy and self.best_sell) else -1
        self.curr_pos = state.position.get(self.name, 0)
        self.curr_sell_vol = 0
        self.curr_buy_vol = 0

        # OPTIMIZABLE VARS
        self.mm_bv = 1
        self.mm_sv = -1
        self.mt_bv = 1
        self.mt_sv = -1


# making a new class for options kinda like ProductTrader
class OptionsTrader:
    def __init__(self, name, strike, expiration, option_type):
        self.name = name
        self.strike = strike
        self.expiration = expiration  # in years
        self.option_type = option_type  # 'call' or 'put'

        self.od = {}
        self.curr_pos = 0
        self.best_bid = 0
        self.best_ask = 0
        self.pos_lim = 200
        self.max_trade_size = 10

        self.underlying_price = 0
        self.implied_vol = 0.3
        self.r = 0.01
        self.T = expiration

    def short(self, price: int, abs_quantity: int, result: Dict[str, List[Order]]):
        result.setdefault(self.name, []).append(Order(self.name, price, -abs_quantity))

    def long(self, price: int, abs_quantity: int, result: Dict[str, List[Order]]):
        result.setdefault(self.name, []).append(Order(self.name, price, abs_quantity))

    def black_scholes_call_price(self, S, K, T, r, sigma):
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            return 0.0
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2)*T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        Nd1 = 0.5 * (1 + math.erf(d1 / math.sqrt(2)))
        Nd2 = 0.5 * (1 + math.erf(d2 / math.sqrt(2)))
        return S * Nd1 - K * math.exp(-r*T) * Nd2


class VR_VoucherTrader(OptionsTrader):
    def __init__(self, name, strike, state: TradingState):
        super().__init__(name, strike, expiration=1/365, option_type="call")

        self.od = state.order_depths.get(self.name, OrderDepth())
        self.curr_pos = state.position.get(self.name, 0)
        self.best_bid = max(self.od.buy_orders.keys()) if self.od.buy_orders else 0
        self.best_ask = min(self.od.sell_orders.keys()) if self.od.sell_orders else 0

        # Estimate underlying price using other vouchers (assumed fair value)
        all_mids = []
        for product in state.order_depths:
            if "VOLCANIC_ROCK_VOUCHER" in product:
                od = state.order_depths[product]
                if od.buy_orders and od.sell_orders:
                    bid = max(od.buy_orders.keys())
                    ask = min(od.sell_orders.keys())
                    all_mids.append((bid + ask) / 2)
        self.underlying_price = statistics.mean(all_mids) if all_mids else 10000

    def compute_action(self):
        market_mid = (self.best_bid + self.best_ask) / 2 if self.best_bid and self.best_ask else 0
        bs_price = self.black_scholes_call_price(self.underlying_price, self.strike, self.T, self.r, self.implied_vol)
        if market_mid < bs_price:
            return "BUY"
        elif market_mid > bs_price:
            return "SELL"
        return "HOLD"

    def trade(self, result: Dict[str, List[Order]]):
        action = self.compute_action()
        if action == "BUY" and self.best_ask:
            qty = min(self.max_trade_size, self.pos_lim - self.curr_pos)
            if qty > 0:
                self.long(self.best_ask, qty, result)
        elif action == "SELL" and self.best_bid:
            qty = min(self.max_trade_size, self.pos_lim + self.curr_pos)
            if qty > 0:
                self.short(self.best_bid, qty, result)
        return result


class Trader:
    def run(self, state: TradingState):
        traderData = json.loads(state.traderData) if state.traderData else {p: {} for p in state.order_depths}
        result: Dict[str, List[Order]] = {}

        option_configs = {
            "VOLCANIC_ROCK_VOUCHER_9500": 9500,
            "VOLCANIC_ROCK_VOUCHER_9750": 9750,
            "VOLCANIC_ROCK_VOUCHER_10000": 10000,
            "VOLCANIC_ROCK_VOUCHER_10250": 10250,
            "VOLCANIC_ROCK_VOUCHER_10500": 10500,
        }

        for name, strike in option_configs.items():
            trader = VR_VoucherTrader(name, strike, state)
            trader.trade(result)

        traderData = json.dumps(traderData)
        conversions = 1
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
