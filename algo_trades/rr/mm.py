import json
from typing import Any

from algo_trades.datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, Dict, List

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
rr_trade_around = 10000
resin_max_pos = 50



class Trader:
    
    def market_make(self, state:TradingState, product:str, result:Dict[str,List[Order]]) -> Dict[str, List[Order]]:
        orders: List[Order] = []
        od = state.order_depths
        
        curr_pos = state.position.get(product, 0)
        buy_len = len(od[product].buy_orders)
        sell_len = len(od[product].sell_orders)
        
        bb = max(od[product].buy_orders) if buy_len else 0
        bs = min(od[product].sell_orders) if sell_len else 0
        gap = bs - bb if bb and bs else -1
        
        
        #TODO: IMPLEMENT SMARTER ORDER VOLUME CHOICE
        #This should be much more dynamic. 
        mbv = 20
        msv = -20
        
        if (curr_pos != 0):
            orders.append(Order(product, rr_trade_around, -curr_pos))
        
        if (gap >= 2):
            orders.append(Order(product, bb + 1, mbv))
            orders.append(Order(product, bs -1 , msv))
        
        result[product] = orders

        return result
    
    def run(self, state: TradingState):

        result: Dict[str, List[Order]] = {}

		
        result = self.market_make(state, 'RAINFOREST_RESIN', result)

        print ("results are")
        print(result)
        
        traderData = "SAMPLE"  
        conversions = 1
        logger.flush(state, result, conversions, traderData)
        
        return result, conversions, traderData
    