import json
from typing import Any
import math
import numpy as np
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, Dict, List
import time
'''
EVERYTHING BELOW HERE NEEDED FOR BT VISUALIZER
'''
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
'''
EVERYTHING ABOVE HERE NEEDED FOR BT VISUALIZER
'''

#GLOBALS
class ProductTrader:

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
        self.big_window_size =100
        self.ma_bv = 20
        self.ma_sv = -20

        #big mean - small mean
        self.prev_diff = 0
    
    def market_take(self, result:Dict[str,List[Order]]):
        orders: List[Order] = []

        #TODO: IMPLEMENT SMARTER ORDER VOLUME CHOICE
        i = 0
        for key, val in self.od.buy_orders.items():
            i += 1
            if(i > 3):
                break
            if (key > self.trade_around):
                orders.append(Order(self.name, key, self.mt_sv))
                self.curr_sell_vol += -val


        j = 0
        for key, val in self.od.sell_orders.items():
            j += 1
            if(j > 3):
                break
            if (key < self.trade_around):
                orders.append(Order(self.name, key, self.mt_bv))
                self.curr_buy_vol += -val

        if self.name in result:
            result[self.name].extend(orders)
        else:
            result[self.name] = orders
            
    def market_make(self, result:Dict[str,List[Order]]):
        orders: List[Order] = []

        #TODO: IMPLEMENT SMARTER ORDER VOLUME CHOICE

        #this conditional assumes we market take the position we are missing out on here
        #if (prod.best_buy < prod.trade_around and prod.best_sell > prod.trade_around):
        
        for backoff in range(1,20):
            '''
            bv_ = max(int(0.8 * b_bank), 20)
            sv_ = min(int(0.8 * s_bank), -20)
            '''

            bv_ = int(self.mm_bv/(backoff ))
            sv_ = int(self.mm_sv/(backoff ))
            if (
                self.gap >= 3 and 
                self.curr_pos < (self.pos_lim - bv_- self.curr_buy_vol) and 
                self.curr_pos > -(self.pos_lim + sv_ + self.curr_sell_vol)):
                orders.append(Order(self.name, self.best_buy + 1, bv_))
                orders.append(Order(self.name, self.best_sell - 1 , sv_))
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
            
    def balance(self, result:Dict[str, List[Order]]):
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
        midprice = (best_buy + best_sell)//2 if best_buy and best_sell else -1

        return (best_buy, midprice, best_sell)
    
    def update_td(self, trader_data):
        """
        Implement this in child and ONLY update trader_data[self.name]
        """
        pass


    
    
class ResinTrader(ProductTrader):
    def __init__(self, state:TradingState):
        # super().__init__('RAINFOREST_RESIN')
        self.name = 'RAINFOREST_RESIN'
        # HARD SET VARS
        self.trade_around = 10000
        self.pos_lim = 50
        
        self.curr_pos = state.position.get(self.name, 0)
        self.curr_sell_vol = 0
        self.curr_buy_vol = 0
        self.od = state.order_depths[self.name]
                
        #OPTIMIZABLE VARS
        # Market Make
        self.best_sell = min(self.od.sell_orders) if (len(self.od.sell_orders)) else self.trade_around
        self.best_buy = max(self.od.buy_orders) if (len(self.od.buy_orders)) else self.trade_around
        self.gap = self.best_sell - self.best_buy if self.best_buy and self.best_sell else -1
        self.mm_bv = 20
        self.mm_sv = -20
        
        self.mt_bv = 15
        self.mt_sv = -15

class Kelp(ProductTrader):
    def __init__(self, state:TradingState):
        super().__init__('KELP')
        self.od = state.order_depths[self.name]
        self.trade_around = 10000
        self.pos_lim = 50
        
        # Using "wvap" to find ideal best buy/sell
        (self.best_buy, _, self.best_sell) = self.calc_vwaps()  
        
        self.gap = (self.best_sell - self.best_buy) if (self.best_buy and self.best_sell) else -1
        self.curr_pos = state.position.get(self.name, 0)
        self.curr_sell_vol = 0
        self.curr_buy_vol = 0
        
        #OPTIMIZABLE VARS
        self.mm_bv = 20
        self.mm_sv = -20
        self.mt_bv = 20
        self.mt_sv = -20

class SquidInk(ProductTrader):

    def __init__(self, state: TradingState, traderData: Dict):
        super().__init__('SQUID_INK')
        self.od = state.order_depths[self.name]
        
        # HARDCODED VARS
        # TODO: Do we need this
        # self.trade_around = 2000
        self.pos_lim = 50

        # Using "wvap" to find ideal best buy/sell
        (self.best_buy, self.midprice, self.best_sell) = self.calc_vwaps()
        self.update_td(traderData, self.midprice)
        self.gap = (self.best_sell - self.best_buy) if (self.best_buy and self.best_sell) else -1
        self.curr_pos = state.position.get(self.name, 0)
        self.curr_sell_vol = 0
        self.curr_buy_vol = 0


        # OPTIMIZABLE VARS
        # Market Making
        self.mm_bv = 10
        self.mm_sv = -10
        
        self.mt_bv = 20
        self.mt_sv = -20
        
        self.ma_bv = 40
        self.ma_sv = -40
        self.big_window_size = 100
        
    def clearing_avg(self, result: Dict[str, List[Order]], traderData):
        orders: List[Order] = []

        bw, sw = self.get_windows()
        if not bw or not sw:
            return

        bw_mean = np.mean(bw)
        sw_mean = np.mean(sw)
        curr_diff = bw_mean - sw_mean

        bw_std = np.std(bw)
        dynamic_threshold = max(20, min(bw_std * 1.5,40))

        if len(self.prev_prices) < self.big_window_size:
            logger.print(f"Insufficient data: {len(self.prev_prices)}/{self.big_window_size}")
            return

        if self.midprice < bw_mean - dynamic_threshold:
            order_volume = min(self.ma_bv, int(self.pos_lim * 0.5))  # Limit to 10% of max position
            orders.append(Order(self.name, self.best_sell, order_volume))
            logger.print(f"Buy signal at {self.midprice} (mean: {bw_mean:.2f}, threshold: {dynamic_threshold:.2f})")
        elif self.midprice > bw_mean + dynamic_threshold:
            order_volume = min(self.ma_sv, int(self.pos_lim * 0.5   ))
            orders.append(Order(self.name, self.best_buy, -order_volume))  # Negative for sell
            logger.print(f"Sell signal at {self.midprice} (mean: {bw_mean:.2f}, threshold: {dynamic_threshold:.2f})")

        self.prev_diff = curr_diff

        if orders:
            if self.name in result:
                result[self.name].extend(orders)
            else:
                result[self.name] = orders

    def update_td(self, traderData, new_mid_price):
        """
        For Squid Ink we maintain a sliding window
        
        """
        self.prev_prices = traderData[self.name].get('prev_prices', [])
        
        if (len(self.prev_prices) == self.big_window_size):
            self.prev_prices.pop(0)
            
        self.prev_prices.append(new_mid_price)
        
        traderData[self.name]['prev_prices'] = self.prev_prices
        
        return
    

    def get_windows(self):
        return self.prev_prices, self.prev_prices[-(self.big_window_size//2):]


class Trader:

    def run(self, state: TradingState):

        if state.traderData == '' or state.traderData == None:
            traderData = {
                "RAINFOREST_RESIN" : {},
                "SQUID_INK" : {},
                "KELP" : {}
            }
        else:
            traderData = json.loads(state.traderData)
        
        rr = ResinTrader(state)
        kl = Kelp(state)
        si = SquidInk(state, traderData)

        result: Dict[str, List[Order]] = {}

        kl.balance(result)
        # # self.market_bully(kl, result)
        kl.market_make(result)

        rr.balance(result)
        rr.market_make(result)
        rr.market_take(result)

        si.balance(result)
        si.clearing_avg(result, traderData)
        si.market_make(result)

        traderData = json.dumps(traderData)
        conversions = 1
        logger.flush(state, result, conversions, traderData)

        return result, conversions, traderData


