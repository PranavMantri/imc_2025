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

def moving_avg(self, prod:Sideways_Product, result:Dict[str,List[Order]]):
        orders: List[Order] = []

        bw,sw = prod.get_window()

        # logger.print(*bw)
        # logger.print(*sw)

        bw_mean = np.mean(bw)
        sw_mean = np.mean(sw)

        curr_diff = bw_mean - sw_mean

        #big surpassed small - downward trend prob
        if(prod.prev_diff < 0 and curr_diff > 0):
            logger.print("big passed small")
            orders.append(Order(prod.name, prod.best_sell, prod.ma_bv))
        #small surpassed big - current upward trend
        if(prod.prev_diff > 0 and curr_diff < 0):
            logger.print("small passed big")
            orders.append(Order(prod.name, prod.best_buy, prod.ma_sv))

        prod.prev_diff = curr_diff
        if prod.name in result:
            result[prod.name].extend(orders)
        else:
            result[prod.name] = orders
        return