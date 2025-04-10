import json
from typing import Any

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
    max_pos = 0
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

class Resin(Sideways_Product):
    def __init__(self, state:TradingState):
        self.name = 'RAINFOREST_RESIN'
        self.od = state.order_depths['RAINFOREST_RESIN']
        self.trade_around = 10000
        self.max_pos = 50
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


class Trader:

    def market_take(self, prod:Sideways_Product, result:Dict[str,List[Order]]) -> Dict[str, List[Order]]:
        orders: List[Order] = []

        #TODO: IMPLEMENT SMARTER ORDER VOLUME CHOICE
        # This should be much more dynamic.
        # See Resin Class
       
        if (prod.curr_pos != 0):
            orders.append(Order(prod.name, prod.trade_around, -prod.curr_pos))
        
        #market taking code
        if (prod.best_sell < prod.trade_around):
            orders.append(Order(prod.name, prod.best_sell, prod.mt_bv))
        if (prod.best_buy > prod.trade_around):
            orders.append(Order(prod.name, prod.best_buy, prod.mt_sv))
        
        result[prod.name] = orders
        return result
    
    def market_make(self, prod:Sideways_Product, result:Dict[str,List[Order]]) -> Dict[str, List[Order]]:
        orders: List[Order] = []
    
       
        
        
        #TODO: IMPLEMENT SMARTER ORDER VOLUME CHOICE
        #This should be much more dynamic. 
     
        
        if (prod.curr_pos != 0):
            orders.append(Order(prod.name, prod.trade_around, -prod.curr_pos))
        
        #market making code
        
        if (prod.gap >= 2):
            orders.append(Order(prod.name,prod.best_buy + 1, prod.mm_bv))
            orders.append(Order(prod.name, prod.best_sell - 1 , prod.mm_sv))
        
        
        result[prod.name] = orders

        return result
    
    def run(self, state: TradingState):

        rr = Resin(state)
        result: Dict[str, List[Order]] = {}

		
        result = self.market_take(rr, result)

        logger.print ("results: ", result)
        
        traderData = "SAMPLE"  
        conversions = 1
        logger.flush(state, result, conversions, traderData)
        
        return result, conversions, traderData