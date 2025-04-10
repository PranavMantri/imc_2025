import json
from typing import Any
import math
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, Dict, List

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
class Sideways_Product:
    name = ''
    od = Dict[Symbol, OrderDepth]
    trade_around = 0
    pos_lim = 0
    best_sell = 0
    best_buy = 0
    gap = 0
    curr_pos = 0
    curr_sell_pos = 0
    curr_buy_pos = 0
    mm_bv = 0
    mm_sv = 0
    mt_bv = 0
    mt_sv = 0
    
    def wvap_orders(self) -> None:
        
        buy_avg = 0
        buy_vol = 0
        sell_avg = 0
        sell_vol = 0
        
        if (len(self.od.buy_orders)):
        
            for key, value in self.od.buy_orders.items():
                buy_avg += key * value
                buy_vol += value
            
            self.best_buy = int(math.ceil(buy_avg / buy_vol))            
        
        else:
            
            self.best_buy = 0

        if (len(self.od.sell_orders)):
            for key, value in self.od.sell_orders.items():
                sell_avg += key * value
                sell_vol += value
            
            self.best_sell = int(math.floor(sell_avg / sell_vol))
        
        else:
            
            self.best_sell = 0
        
        logger.print(f"best-buy: {self.best_buy}, best-sell: {self.best_sell}")
        return
    
    
class Resin(Sideways_Product):
    def __init__(self, state:TradingState):
        self.name = 'RAINFOREST_RESIN'
        self.od = state.order_depths['RAINFOREST_RESIN']
        self.trade_around = 10000
        self.pos_lim = 50
        self.best_sell = min(self.od.sell_orders) if (len(self.od.sell_orders)) else 10000
        self.best_buy = max(self.od.buy_orders) if (len(self.od.buy_orders)) else 10000
        self.gap = self.best_sell - self.best_buy if self.best_buy and self.best_sell else -1
        self.curr_pos = state.position.get('RAINFOREST_RESIN', 0)
        self.curr_sell_pos = 0
        self.curr_buy_pos = 0
        
        #OPTIMIZABLE VARS
        self.mm_bv = 20
        self.mm_sv = -20
        self.mt_bv = 20
        self.mt_sv = -20

class Kelp(Sideways_Product):
    def __init__(self, state:TradingState):
        self.name = 'KELP'
        self.od = state.order_depths['KELP']
        self.trade_around = 10000
        self.pos_lim = 50
        
        # Using "wvap" to find ideal best buy/sell
        self.wvap_orders()
        
        
        # min max for best buy/sell
        # self.best_sell = min(self.od.sell_orders) if (len(self.od.sell_orders)) else 0
        # self.best_buy = max(self.od.buy_orders) if (len(self.od.buy_orders)) else 0    
        
        self.gap = (self.best_sell - self.best_buy) if (self.best_buy and self.best_sell) else -1
        self.curr_pos = state.position.get('KELP', 0)
        self.curr_sell_pos = 0
        self.curr_buy_pos = 0
        
        #OPTIMIZABLE VARS
        self.mm_bv = 20
        self.mm_sv = -20
        self.mt_bv = 20
        self.mt_sv = -20


class Trader:

    def balance(self, prod:Sideways_Product, result:Dict[str, List[Order]]):
        orders: List[Order] = []
        if (prod.curr_pos != 0):

            ##TODO: buy/sell at the farthest price from trade-around for balancing 
       
            if (prod.curr_pos < 0):
                orders.append(Order(prod.name, prod.best_buy + 1, -prod.curr_pos))
                prod.curr_buy_pos += prod.curr_pos
            else:
                orders.append(Order(prod.name, prod.best_sell - 1, -prod.curr_pos))
                prod.curr_sell_pos += prod.curr_pos


        if prod.name in result:
            result[prod.name].extend(orders)
        else:
            result[prod.name] = orders
 
            

    def market_take(self, prod:Sideways_Product, result:Dict[str,List[Order]]):
        orders: List[Order] = []

        #TODO: IMPLEMENT SMARTER ORDER VOLUME CHOICE
        # This should be much more dynamic.
        # See Resin Class
       
        
        
        #market taking code
        if (prod.best_sell < prod.trade_around):
            orders.append(Order(prod.name, prod.best_sell, prod.mt_bv))
            prod.curr_buy_pos += prod.mt_bv
        if (prod.best_buy > prod.trade_around):
            orders.append(Order(prod.name, prod.best_buy, prod.mt_sv))
            prod.curr_sell_pos += prod.mt_sv
        
        if prod.name in result:
            result[prod.name].extend(orders)
        else:
            result[prod.name] = orders
 
    
    def market_make(self, prod:Sideways_Product, result:Dict[str,List[Order]]):
        orders: List[Order] = []
        
<<<<<<< HEAD
        if (
                prod.gap >= 3 and 
                prod.curr_pos < (prod.pos_lim - prod.mm_bv - prod.curr_buy_pos) and 
                prod.curr_pos > -(prod.pos_lim + prod.mm_sv + prod.curr_sell_pos)
            ):
            
            #TODO: IMPLEMENT SMARTER ORDER VOLUME CHOICE
            orders.append(Order(prod.name, prod.best_buy + 1, prod.mm_bv))
            orders.append(Order(prod.name, prod.best_sell - 1 , prod.mm_sv))
=======
        #TODO: IMPLEMENT SMARTER ORDER VOLUME CHOICE
        #This should be much more dynamic. 
        #See Resin 

        b_bank = prod.pos_lim - prod.curr_buy_pos
        s_bank = prod.pos_lim - prod.curr_sell_pos
        for backoff in range(1,10):

            '''
            bv_ = max(int(0.8 * b_bank), 20)
            sv_ = min(int(0.8 * s_bank), -20)
            '''



            bv_ = int(prod.mm_bv/backoff)
            sv_ = int(prod.mm_sv/backoff)
            if (prod.gap >= 3 and prod.curr_pos < (prod.pos_lim - bv_- prod.curr_buy_pos) and prod.curr_pos > -(prod.pos_lim + sv_ + prod.curr_sell_pos)):
                orders.append(Order(prod.name, prod.best_buy + 1, bv_))
                orders.append(Order(prod.name, prod.best_sell - 1 , sv_))
                break


>>>>>>> origin/rr_optimized
        

        if prod.name in result:
            result[prod.name].extend(orders)
        else:
            result[prod.name] = orders
            
    def market_bully(self, prod:Sideways_Product, result:Dict[str,List[Order]]):
        orders: List[Order] = []
            
        for key, value in prod.od.buy_orders.items():
            if key > prod.best_buy:
                logger.print(f"We are bullying the buy at {key}")
                orders.append(Order(prod.name, key, -value))
                # orders.append(Order(prod.name, prod.best_buy, value))

        for key, value in prod.od.sell_orders.items():
            if key < prod.best_sell:
                logger.print(f"We are bullying the sell at {key}")
                orders.append(Order(prod.name, key, -value))
                # orders.append(Order(prod.name, prod.best_sell, value))

        if prod.name in result:
            result[prod.name].extend(orders)
        else:
            result[prod.name] = orders
    
    def run(self, state: TradingState):

        rr = Resin(state)
        #kl = Kelp(state)
        result: Dict[str, List[Order]] = {}
        
<<<<<<< HEAD
        self.balance(kl, result)
        # self.market_bully(kl, result)
        self.market_make(kl, result)
=======
        #self.balance(kl, result)
        #self.market_make(kl, result)
>>>>>>> origin/rr_optimized

        self.balance(rr, result)
        self.market_make(rr, result)
        self.market_take(rr, result)
        
        traderData = "SAMPLE"  
        conversions = 1
        logger.flush(state, result, conversions, traderData)
        
        return result, conversions, traderData
