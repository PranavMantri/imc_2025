from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict

class Trader:
	def run(self, state: TradingState):
		print("Current Position:", state.position)
		result: Dict[str, List[Order]] = {}

		# Configurable parameters
		mid_price_offset = 2  # Spread distance from mid-price
		max_position = 50  # Position limit
		lot_size = 5  # Number of units per order

		for product, od in state.order_depths.items():
			if product != 'RAINFOREST_RESIN':
				continue

			orders: List[Order] = []
			curr_pos = state.position.get(product, 0)

			# Determine best bid and ask from market
			best_bid = max(od.buy_orders.keys(), default=0)
			best_ask = min(od.sell_orders.keys(), default=0)
			mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else 0

			# Adjust prices dynamically
			bid_price = int(mid_price - mid_price_offset)
			ask_price = int(mid_price + mid_price_offset)

			# Ensure we don’t exceed position limits
			if curr_pos < max_position:
				orders.append(Order(product, bid_price, lot_size))  # Buy order

			if curr_pos > -max_position:
				orders.append(Order(product, ask_price, -lot_size))  # Sell order

			result[product] = orders

		traderData = "MARKET_MAKER_STATE"
		return result, 0, traderData
