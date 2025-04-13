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


LIMITS = {
    'RAINFOREST_RESIN' : 50,
    'KELP' : 50,
    'CROISSANTS' : 250,
    'JAMS' : 350,
    'DJEMBES' : 60,
    'PICNIC_BASKET1' : 60,
    'PICNIC_BASKET2' :100,
    'VOLCANIC_ROCK' : 400
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
        self.prev_price_window = ppw
        self.best_buy = 0
        self.best_sell = 0
        self.midprice = 0

        #volume vars
        self.curr_pos = state.position.get(self.name, 0)
        self.pos_lim = LIMITS[self.name]
        self.buying_power = self.pos_lim - self.curr_pos
        self.selling_power = -self.pos_lim - self.curr_pos
        self.buy_this_timestamp = 0
        self.sell_this_timestamp = 0



         #call calculations functions
        (self.best_buy, self.midprice, self.best_sell) = self.calc_vwaps()
        
        ######################################
        # Pull out whatever from trader_data #
        ######################################
        self.get_prev_prices()


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


        

class Croissant(ProductTrader):
   
    def __init__(self, state: TradingState, traderData: Dict, result:Dict[Symbol, List[Order]], ppw:int):
        super().__init__(state, traderData, 'CROISSANTS', ppw)

        self.curr_index = 0
        self.sell_index = 0
        self.buy_index = 0
        self.end_index = 0

    
        (self.curr_index, self.sell_index, self.buy_index, self.end_index) = self.get_indicies()
        self.increment_curr_index() if self.end_index > 0 else None
        logger.print(f"ci {self.curr_index} sell at: {self.sell_index} buy at: {self.buy_index}, end at: {self.end_index}")


    
    def get_indicies(self) -> tuple[int,int,int]:
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

    def __init__(self, crst:Croissant, jams:Jams):
        
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


class Trader:

    def run(self, state: TradingState):

        if state.traderData == '' or state.traderData == None:
            traderData = {
                'RAINFOREST_RESIN' : {},
                'KELP' : {},
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

        param_window = 150
        jams = Jams(state, traderData, result, param_window)
        crst = Croissant(state, traderData, result, param_window)

        crst.execute_if_time(result)
        
        cj_t = Croissants_Jams_Trader(crst, jams)
        cj_t.predict_the_future(result)

        traderData = json.dumps(traderData)
        conversions = 1
        logger.flush(state, result, conversions, traderData)

        return result, conversions, traderData


                

