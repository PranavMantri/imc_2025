from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict

class Trader:
    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        print("traderData:", state.traderData)
        print("Observations:", state.observations)
        print("Current Position:", state.position)	
        result: Dict[str, List[Order]] = {}
        rr_trade_around = 9999 
        rr_pos_lim = 50

        for product, od in state.order_depths.items():
            if product != 'RAINFOREST_RESIN':
                continue

            orders: List[Order] = []


            curr_pos = state.position.get('RAINFOREST_RESIN', 0)
            buy_len = len(od.buy_orders)
            sell_len = len(od.sell_orders)

            bb = max(od.buy_orders.keys()) if buy_len else 0  # Best bid
            bs = min(od.sell_orders.keys()) if sell_len else 0  # Best ask

            mbp = 0
            mbv = 8
            msp = 0
            msv = -8

            if abs(curr_pos) == rr_pos_lim:
                if curr_pos < 0:
                    orders.append(Order(product, bb, 1))
                    continue
                orders.append(Order(product, bs, -1))
                continue
            # Placing limit orders at strategic price points
            if bb < rr_trade_around:
                mbp = min(bb + 1, rr_trade_around)
                orders.append(Order(product, mbp, mbv))  # Corrected order initialization
                
            if bs > rr_trade_around:
                msp = max(bs - 1, rr_trade_around)
                orders.append(Order(product, msp, msv))  # Corrected order initialization

            result[product] = orders
                
        traderData = "SAMPLE"  # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        conversions = 1
        return result, conversions, traderData

