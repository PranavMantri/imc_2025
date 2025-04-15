import json
from typing import Any
import math
import numpy as np
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, Dict, List
import time
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
    'RAINFOREST_RESIN' : 50,
    'KELP' : 50,
    'SQUID_INK' : 50,
    'CROISSANTS' : 250,
    'JAMS' : 350,
    'DJEMBES' : 60,
    'PICNIC_BASKET1' : 60,
    'PICNIC_BASKET2' :100,
    'VOLCANIC_ROCK' : 400
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
    'VOLCANIC_ROCK' : -1
}
#GLOBALS
class ProductTrader:

    def __init__(self, state:TradingState, traderDict:Dict, name:str, ppw:int):
        
        #declare ALL variables here first
        #yer
        self.name = name
        self.od = state.order_depths.get(self.name, {})
        self.trader_data:Dict = traderDict.get(self.name, {})

        #price vars
        self.prev_prices = []
        self.prev_price_window = ppw if ppw != 0 else 101
        self.best_buy = 0
        self.best_sell = 0
        self.midprice = 0
        self.trade_around = PROD_TR_AR[self.name]
        self.gap_trigger = 3


        #volume vars
        self.curr_pos = state.position.get(self.name, 0)
        self.pos_lim = LIMITS[self.name]
        self.buying_power = self.pos_lim - self.curr_pos
        self.selling_power = self.pos_lim + self.curr_pos
        self.market_take_buy_vol = 10
        self.market_take_sell_vol = 10
        self.market_make_buy_vol = 10
        self.market_make_sell_vol = 10
        
        ## probably don't need these 
        #self.buyVol_this_timestamp = 0
        #self.sellVol_this_timestamp = 0
        ################################



         #call calculations functions
        (self.best_buy, self.midprice, self.best_sell) = self.calc_vwaps()
        self.gap = self.best_sell - self.best_buy if self.best_buy and self.best_sell else -1
        
        ######################################
        # Pull out whatever from trader_data #
        ######################################
        self.get_prev_prices()

        ########

   
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
        midprice = (best_buy + best_sell)//2 if best_buy and best_sell else -1

        return (best_buy, midprice, best_sell)
    
    #this returns the last prev_price_window prices including this timestamp
    def get_prev_prices(self) -> None:
        prev_prices = self.trader_data.get('prev_prices', [])
        
        if (len(prev_prices) > self.prev_price_window):
            prev_prices.pop(0)
        
        prev_prices.append(self.midprice)
        self.prev_prices = prev_prices
        self.trader_data['prev_prices'] = prev_prices


    def balance(self, result:Dict[str, List[Order]]):
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
            sv_ = int(self.market_make_sell_vol/ (backoff))

            logger.print(f'bv_ {bv_} > self.buying_power {self.buying_power}')
            logger.print(f'sv_ {sv_} > self.buying_power {self.buying_power}')

            if bv_ > self.buying_power or sv_> self.selling_power:
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

class ResinTrader(ProductTrader):
    def __init__(self, state: TradingState, traderData: Dict, result:Dict[Symbol, List[Order]], ppw:int):
        super().__init__(state, traderData, 'RAINFOREST_RESIN', ppw)

        # GRID SEARCHED
        self.mm_vol_r = 0.5
        self.mt_bv = 15
        self.mt_sv = 15
        self.gap_trigger = 4
        self.best_delta = 1

        # NOT GRID SEARCHED (but necessary for product trader)
        self.market_make_buy_vol = 15
        self.market_make_sell_vol = 15


class Kelp(ProductTrader):
    def __init__(self, state: TradingState, traderData: Dict, result:Dict[Symbol, List[Order]], ppw:int):
        super().__init__(state, traderData, 'KELP', ppw)

        # OPTIMIZABLE VARS
        # GRID SEARCHED
        self.mm_vol_r = 0.5
        self.gap_trigger = 2
        self.best_delta = 1

        # NOT GRID SEARCHED (but nesc. for ts to work)
        self.mm_bv = 15
        self.mm_sv = -15


class SquidInk(ProductTrader):

    def __init__(self, state: TradingState, traderData: Dict, result:Dict[Symbol, List[Order]], ppw:int):
        super().__init__(state, traderData, 'SQUID_INK', ppw)

        self.trade_around = self.moving_avg()
        #self.trade_around = self.moving_avg()

        # OPTIMIZABLE VARS
        # Market Making
        self.mm_vol_r = 0.5
        self.gap_trigger = 2
        self.best_delta = 3

        # Moving Avg
        # self.ma_vol_r = params['ma_vol_r']
        # self.fixed_threshold = params['fixed_threshold']
        # self.big_window_size = params['big_window_size']
    
    def moving_avg(self):

        bw, sw = self.get_windows()
        if not bw or not sw:
            return

        return np.mean(bw)

    def get_windows(self):
        return self.prev_prices, self.prev_prices[-(self.prev_price_window // 2):]

    def market_take(self, result: Dict[str, List[Order]]):
        orders: List[Order] = []

        if len(self.prev_prices) < self.prev_price_window:
            logger.print(f"Insufficient data: {len(self.prev_prices)}/{self.prev_price_window}")
            return

        super().market_take(result)





class Croissants(ProductTrader):
   
    def __init__(self, state: TradingState, traderData: Dict, result:Dict[Symbol, List[Order]], ppw:int):
        super().__init__(state, traderData, 'CROISSANTS', ppw)

        self.curr_index = 0
        self.sell_index = 0
        self.buy_index = 0
        self.end_index = 0

    
        (self.curr_index, self.sell_index, self.buy_index, self.end_index) = self.get_indicies()
        self.increment_curr_index() if self.end_index > 0 else None
        logger.print(f"ci {self.curr_index} sell at: {self.sell_index} buy at: {self.buy_index}, end at: {self.end_index}")


    
    def get_indicies(self) -> tuple[int,int,int,int]:
        return (self.trader_data.get('curr_index', -1),
                self.trader_data.get('sell_index', -1),
                self.trader_data.get('buy_index', -1),
                self.trader_data.get('end_index', -1))
    
    def increment_curr_index(self) -> None:
        self.curr_index += 1
        self.trader_data['curr_index'] = self.curr_index
    
    def set_indicies(self, curr:int, sell:int, buy:int, end:int) -> None:
        self.trader_data['curr_index'] = curr
        self.trader_data['sell_index'] = sell
        self.trader_data['buy_index'] = buy
        self.trader_data['end_index'] = end
    
    def execute_if_time(self, result:Dict[Symbol, List[Order]]):

        orders:List[Order] = []
        if self.curr_index < 0 or self.buy_index < 0 or self.sell_index < 0:
            self.balance(result)
            logger.print("at least 1 index less than 0!")
            return
        
        #all indicies are valid

        if self.curr_index == self.buy_index:
            #time to buy!
            logger.print("now we long")
            orders.append(Order(self.name, self.best_sell, self.buying_power))
        
        elif self.curr_index == self.sell_index:
            #time to sell!
            logger.print("now we short")
            orders.append(Order(self.name, self.best_buy, self.selling_power))
        
        elif self.curr_index >= self.end_index:
            logger.print("at the end, now we balance")
            self.balance(result)
            self.set_indicies(-2,-2,-2,-2)
            return

        if self.name in result:
            result[self.name].extend(orders)
        else:
            result[self.name] = orders
        
class Jams (ProductTrader):
    def __init__(self, state: TradingState, traderData: Dict, result:Dict[Symbol, List[Order]], ppw:int):
        super().__init__(state, traderData, 'JAMS', ppw)    

class Croissants_Jams_Trader():

    def __init__(self, crst:Croissants, jams:Jams):
        
        self.crst = crst
        self.jams = jams

    def predict_the_future(self, result:Dict[Symbol, List[Order]]):

        if (self.crst.end_index > 0):
            logger.print(f"we still have {self.crst.end_index - self.crst.curr_index} left!")
            return
        
        #effectively wait 1 timestamp for the position to clear
        if (self.crst.end_index == -2):
            self.crst.set_indicies(-1,-1,-1,-1)
            return
            
        # here means that we need to set the indicies!
        if (len(self.jams.prev_prices) < self.jams.prev_price_window):
            logger.print("not enough prices yet!")
            return

        ## we have enough prices to do some min-maxxing
        orders:List[Order] = []

        min_val = min(self.jams.prev_prices)
        max_val = max(self.jams.prev_prices)

        if (max_val - min_val < 20):
            return


        max_index = self.jams.prev_prices.index(max_val)
        min_index = self.jams.prev_prices.index(min_val)

        logger.print(f'min_index is {min_index} and max_index is {max_index}')

        #TODO: ADD A MIN DIFF BETWEEN CURR, MIN, MAX, END
        if (min_index < max_index):
            #short to the min, long min -> max, short max -> end
            logger.print("we are going to short first!")
            if (min_index != 0):
                orders.append(Order(self.crst.name, self.crst.best_buy, self.crst.selling_power))
                self.crst.set_indicies(0, max_index, min_index, self.jams.prev_price_window)

        else:
            logger.print("we are going to long first!")

            #long to the max, short max -> min, long max ->end
            if (max_index != 0):
                orders.append(Order(self.crst.name, self.crst.best_sell, self.crst.buying_power))
                self.crst.set_indicies(0, max_index, min_index, self.jams.prev_price_window)
            
        if self.crst.name in result:
            result[self.crst.name].extend(orders)
        else:
            result[self.crst.name] = orders


class Djembes (ProductTrader):
    def __init__(self, state: TradingState, traderData: Dict, result:Dict[Symbol, List[Order]], ppw:int):
        super().__init__(state, traderData, 'DJEMBES', ppw)    



class Picnic_Basket1(ProductTrader):
    def __init__(self, state: TradingState, traderData: Dict, result:Dict[Symbol, List[Order]], ppw:int):
        super().__init__(state, traderData, 'PICNIC_BASKET1', ppw)


class pb1_trader(ProductTrader):

    def __init__(self, traderData: Dict, pb1: Picnic_Basket1, crst: Croissants, jams: Jams, djem: Djembes):

        # parametrizers!
        self.premium = 48.75
        self.dev = 85.1194
        self.num_devs = 1.2
        self.break_out = self.dev * self.num_devs - self.premium

        # normal vars
        self.pb1 = pb1
        self.crst = crst
        self.jams = jams
        self.djem = djem
        self.synth_midprice = (6 * crst.midprice) + (3 * jams.midprice) + djem.midprice

        ########################
        # ACTUAL - SYNTHETIC ! #
        ########################
        self.market_diff = self.pb1.midprice - self.synth_midprice - self.premium

    def trade_the_diff(self, result):
        orders: List[Order] = []

        if (abs(self.market_diff) < self.break_out):
            self.crst.balance(result)
            self.jams.balance(result)
            self.djem.balance(result)
            self.pb1.balance(result)
            return

        # we diff'd too hard. lets make some shells

        # case 1.1 : synthetic is overvalued -> short this
        # case 1.2 : actual is undervalued -> long this

        if (self.market_diff < 0):
            if (self.check_lims() < 0):
                return

            self.short_synthetics(result)

            # TODO: CALL ANOTHER FUNCTION THAT MAKES ALL OF THESE TRADES.
            # TODO: THINK ABOUT WHAT HAPPENS IF A SINGLE ONE OF THESE TRADES
            # DOESN'T GO THROUGH.

            # TODO: what price? what amount?
            orders.append(Order(self.pb1.name, self.pb1.best_sell, 10))

        # case 1: synthetic is undervalued -> long this
        # case 2: actual is overvalued -> short this
        elif (self.market_diff > 0):
            if (self.check_lims() > 0):
                return
            self.long_synthetics(result)

            # TODO: what price? what amount?
            orders.append(Order(self.pb1.name, self.pb1.best_buy, -10))

        if self.pb1.name in result:
            result[self.pb1.name].extend(orders)
        else:
            result[self.pb1.name] = orders

    # TODO!!!!!!
    def check_lims(self) -> int:
        synth_pos = self.crst.curr_pos + self.jams.curr_pos + self.djem.curr_pos
        synth_lim = 6 * self.crst.pos_lim + 4 * self.jams.pos_lim + self.djem.pos_lim

        if (self.pb1.curr_pos > self.pb1.pos_lim or synth_pos < -synth_lim):
            return 1

        if (synth_pos > synth_lim or self.pb1.curr_pos < -self.pb1.pos_lim):
            return -1

        return 0

    def short_synthetics(self, result):
        # self.crst.short(self.crst.best_sell, 6, result)

        # TODO: FIX THIS TO MIRROR ACTUAL
        self.jams.short(self.jams.best_sell, 30, result)
        # self.djem.short(self.djem.best_sell, 1, result)

    def long_synthetics(self, result):
        # self.crst.long(self.crst.best_buy, 6, result)

        # TODO: FIX THIS TO MIRROR ACTUAL
        self.jams.long(self.jams.best_buy, 30, result)
        # self.djem.long(self.djem.best_buy, 1, result)


class PB2ResidualTrader(ProductTrader):
    def __init__(self, state: TradingState, traderData: Dict, result:Dict[Symbol, List[Order]], ppw:int):
        super().__init__(state, traderData, 'PICNIC_BASKET2', ppw)
        self.state = state
        self.traderData = traderData

        self.components = {"CROISSANTS": 4, "JAMS": 2}
        self.flat_cost = 0
        self.max_pos = 100

        self.trade_at = 17
        self.max_trade_size = 1
        self.orders = []

    def get_mid_price(self, symbol: str) -> float:
        od = self.state.order_depths.get(symbol, None)
        if od and od.buy_orders and od.sell_orders:
            best_bid = max(od.buy_orders.keys())
            best_ask = min(od.sell_orders.keys())
            return (best_bid + best_ask) / 2
        return None

    def compute_residual(self) -> float:
        basket_mid = self.get_mid_price(self.name)
        if basket_mid is None:
            return 0.0

        synthetic_value = 0.0
        for comp, qty in self.components.items():
            price = self.get_mid_price(comp)
            if price is None:
                return 0.0
            synthetic_value += qty * price

        return basket_mid - (synthetic_value + self.flat_cost)

    def trade_residual(self, result: Dict[str, List[Order]]):
        residual = self.compute_residual()

        if residual > self.trade_at and self.curr_pos > -self.max_pos:
            best_bid = max(self.od.buy_orders.keys()) if self.od.buy_orders else None
            if best_bid:
                size = min(self.max_trade_size, self.max_pos + self.curr_pos)
                result[self.name] = [Order(self.name, best_bid, -size)]

        elif residual < -self.trade_at and self.curr_pos < self.max_pos:
            best_ask = min(self.od.sell_orders.keys()) if self.od.sell_orders else None
            if best_ask:
                size = min(self.max_trade_size, self.max_pos - self.curr_pos)
                result[self.name] = [Order(self.name, best_ask, size)]



class Trader:

    def run(self, state: TradingState):

        if state.traderData == '' or state.traderData == None:
            traderData = {
                'RAINFOREST_RESIN' : {},
                'KELP' : {},
                'SQUID_INK' : {},
                'CROISSANTS' : {},
                'JAMS' : {},
                'DJEMBES' : {},
                'PICNIC_BASKET1' : {},
                'PICNIC_BASKET2' :{},
                'VOLCANIC_ROCK' : {}
            }
        else:
            traderData = json.loads(state.traderData)
        

        result: Dict[str, List[Order]] = {}

        param_window = 200
        
        rr = ResinTrader(state, traderData, result, param_window)
        kl = Kelp(state, traderData, result, param_window)
        si = SquidInk(state, traderData, result, param_window)
        pb1 = Picnic_Basket1(state, traderData, result, param_window)
        pb2 = PB2ResidualTrader(state, traderData, result, param_window)
        crst = Croissants(state, traderData, result, param_window)
        djem = Djembes(state, traderData, result, param_window)
        jams = Jams(state, traderData, result, param_window)
        cj_t = Croissants_Jams_Trader(crst, jams)
        pb1_t = pb1_trader(traderData, pb1, crst, jams, djem)


        rr.balance(result)
        rr.market_make(result)
        rr.market_take(result)

        kl.balance(result)
        kl.market_make(result)

        si.balance(result)
        si.market_make(result)
       
        jams.market_make(result)
       
        pb2.trade_residual(result)  # your custom PB2 residual logic
        pb1_t.trade_the_diff(result)

        traderData = json.dumps(traderData)
        conversions = 1
        logger.flush(state, result, conversions, traderData)

        return result, conversions, traderData


                

