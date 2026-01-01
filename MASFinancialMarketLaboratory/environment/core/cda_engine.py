from __future__ import annotations
from typing import Callable, Dict, Optional, Tuple, Deque
from sortedcontainers import SortedDict
from collections import deque
import time

from environment.models import Order, Trade, MarketData 
from environment.models.order import OrderType, Side, OrderLifecycle, OrderEndReasons
from environment.configs import get_environment_configuration

from simulation.configs import get_simulation_realtime_data 

from .settlement_ledger import SettlementLedger
from .storage_ledger import StorageLedger



class OrderBook:
    bids:SortedDict[int, Deque[Order]]
    asks:SortedDict[int, Deque[Order]]

    order_map:Dict[int, Order] #OrderID -> Order


    def __init__(self) -> None:
        self.__create_clean_book()

        
    def __create_clean_book(self) -> None:
        bid_sort_key_fn:Callable[[int], int] = lambda k: -k
        self.bids = SortedDict(bid_sort_key_fn)
        self.asks = SortedDict()
        self.order_map = {}
        

    def is_order_exist(self, order_id:int) -> bool:
        return order_id in self.order_map

    
    def add_order(self, order:Order) -> bool:
        #Expectations
        # 1-agent_exist (ASSURED)
        # 2-order_type = Limit
        # 3-price != None && price > 0
        # 4-quantity > 0
        # 5-0 < remaining_quantity < quantity
        # 6-lifecycle = WORKING
        # 7-end_reason = NONE

        assert order.order_type == OrderType.LIMIT #2
        assert order.price is not None #3
        assert order.price > 0 #3
        assert order.quantity > 0 #4
        assert 0 < order.remaining_quantity <= order.quantity
        assert order.lifecycle == OrderLifecycle.WORKING
        assert order.end_reason == OrderEndReasons.NONE
        
        if self.is_order_exist(order.order_id):
            return False

        self.order_map[order.order_id] = order

        target_book = self.bids if order.side == Side.BUY else self.asks
        price_key = order.price

        if price_key not in target_book:
            target_book[price_key] = deque()

        target_book[price_key].append(order)
        
        return True

    
    def remove_order(self, order_id:int) -> Optional[Order]:
        #Expectations
        # 1-agent_exist (ASSURED)
        # 2-order_type = Limit (ASSURED)
        # 3-price != None && price > 0 (ASSURED)
        # 4-quantity > 0 (ASSURED)
        # 5-0 < remaining_quantity < quantity (ASSURED)
        # 6-lifecycle = WORKING (ASSURED)
        # 7-end_reason = NONE (ASSURED)

        if not self.is_order_exist(order_id):
            return None

        order = self.order_map.pop(order_id)

        assert order.price is not None
        target_book = self.bids if order.side == Side.BUY else self.asks
        price_key = order.price

        target_book[price_key].remove(order)

        if not target_book[price_key]:
            del target_book[price_key]

        return order
        

    def expire_book(self) -> Optional[Tuple[Tuple[Order, ...], Tuple[Order, ...]]]:
        #Expectations
        # 1-agent_exist (ASSURED)
        # 2-order_type = Limit (ASSURED)
        # 3-price != None && price > 0 (ASSURED)
        # 4-quantity > 0 (ASSURED)
        # 5-0 < remaining_quantity < quantity (ASSURED)
        # 6-lifecycle = WORKING (ASSURED)
        # 7-end_reason = NONE (ASSURED)

        if not self.order_map:
            return None
        
        bids = [order for level_queue in self.bids.values() for order in level_queue]
        asks = [order for level_queue in self.asks.values() for order in level_queue]

        self.__create_clean_book()

        return tuple(bids), tuple(asks)

    
    def get_best_bid_price(self) -> Optional[int]:
        if not self.bids:
            return None

        return self.bids.keys()[0]

    
    def get_best_ask_price(self) -> Optional[int]:
        if not self.asks:
            return None

        return self.asks.keys()[0]


    def get_best_bid_order(self) -> Optional[Order]:
        #Expectations
        # 1-agent_exist (ASSURED)
        # 2-order_type = Limit (ASSURED)
        # 3-price != None && price > 0 (ASSURED)
        # 4-quantity > 0 (ASSURED)
        # 5-0 < remaining_quantity < quantity (ASSURED)
        # 6-lifecycle = WORKING (ASSURED)
        # 7-end_reason = NONE (ASSURED)

        if not self.bids:
            return None

        return self.bids.peekitem(0)[1][0]
    
    def get_best_ask_order(self) -> Optional[Order]:
        #Expectations
        # 1-agent_exist (ASSURED)
        # 2-order_type = Limit (ASSURED)
        # 3-price != None && price > 0 (ASSURED)
        # 4-quantity > 0 (ASSURED)
        # 5-0 < remaining_quantity < quantity (ASSURED)
        # 6-lifecycle = WORKING (ASSURED)
        # 7-end_reason = NONE (ASSURED)

        if not self.asks:
            return None

        return self.asks.peekitem(0)[1][0]


    def get_l1_bids(self) -> Optional[Tuple[int, int, int]]:
        price = self.get_best_bid_price()
        if price is None:
            return

        bid_queue = self.bids[price]

        number_of_bids = len(bid_queue)
        bid_size = 0
        for bid in bid_queue:
            bid_size += bid.remaining_quantity

        return price, bid_size, number_of_bids


    def get_l1_asks(self) -> Optional[Tuple[int, int, int]]:
        price = self.get_best_ask_price()
        if price is None:
            return

        ask_queue = self.asks[price]

        number_of_asks = len(ask_queue)
        ask_size = 0
        for ask in ask_queue:
            ask_size += ask.remaining_quantity

        return price, ask_size, number_of_asks


    def get_l2_bids(self) -> Optional[Tuple[Tuple[int, int, int], ...]]:
        if not self.bids:
            return

        bid_level_data = []
        for price, bid_queue in self.bids.items():
            number_of_bids = len(bid_queue)
            bid_size = 0
            for bid in bid_queue:
                bid_size += bid.remaining_quantity
                
            bid_level_data.append((price, bid_size, number_of_bids))

        return tuple(bid_level_data)

    
    def get_l2_asks(self) -> Optional[Tuple[Tuple[int, int, int], ...]]:
        if not self.asks:
            return

        ask_level_data = []
        for price, ask_queue in self.asks.items():
            number_of_asks = len(ask_queue)
            ask_size = 0
            for ask in ask_queue:
                ask_size += ask.remaining_quantity

            ask_level_data.append((price, ask_size, number_of_asks))

        return tuple(ask_level_data)

    
class CDAEngine:
    order_book:OrderBook
    storage_ledger:StorageLedger
    settlement_ledger:SettlementLedger

    __next_trade_id:int

    #__trades:List[Tuple[int, int]] #price - volume
    __last_traded_price:Optional[int]
    __last_trade_volume:Optional[int]

    __macro_trade_value:int
    __macro_trade_volume:int
    __micro_trade_count:int
    __micro_trade_value:int
    __micro_trade_volume:int

    
    def __init__(self, storage_ledger:StorageLedger, settlement_ledger:SettlementLedger) -> None:
        self.storage_ledger = storage_ledger
        self.settlement_ledger = settlement_ledger

        self.order_book = OrderBook()

        self.__next_trade_id = 0

        #self.__trades = []
        self.__last_traded_price = None
        self.__last_trade_volume = None

        self.__macro_trade_value = 0
        self.__macro_trade_volume = 0
        self.__micro_trade_count = 0
        self.__micro_trade_value = 0
        self.__micro_trade_volume = 0
    
        
    @property
    def trade_id(self) -> int:
        trade_id = self.__next_trade_id 
        self.__next_trade_id += 1

        return trade_id

    
    def process_new_order(self, order:Order) -> None:
        # Expectations:
        # 1-agent_id exist
        # 2-If Limit -> price != None && price > 0
        # 3-If Market -> price = None
        # 4-quantity > 0
        # 5-remaining_quantity = quantity
        # 6-lifecycle = NEW
        # 7-end_reason = NONE
        # 8-average_trade_price = None
        # 9-trades = {}

        assert self.settlement_ledger.is_account_exist(order.agent_id) #1
        assert order.quantity > 0 #4
        assert order.remaining_quantity == order.quantity #5
        assert order.lifecycle == OrderLifecycle.NEW #6
        assert order.end_reason == OrderEndReasons.NONE #7
        #assert order.average_trade_price is None #8
        assert not order.trades #9
        
        order.lifecycle = OrderLifecycle.WORKING
        
        if order.order_type == OrderType.LIMIT:
            self.__process_new_limit_order(order)

        elif order.order_type == OrderType.MARKET:
            self.__process_new_market_order(order)

        else:
            assert False
        

    def __process_new_limit_order(self, order:Order) -> None:
        # Expectations:
        # 1-agent_exist (ASSURED)
        # 2-order_type = LIMIT  
        # 3-price != None && price > 0
        # 4-quantity > 0 (ASSURED)
        # 5-remaining_quantity = quantity (ASSURED) 
        # 6-lifecycle = WORKING 
        # 7-end_reason = NONE (ASSURED)
        # 8-average_trade_price = None (ASSURED)
        # 9-trades = {} (ASSURED)
        
        assert order.order_type == OrderType.LIMIT #2
        assert order.price is not None #3
        assert order.price > 0 #3
        assert order.lifecycle == OrderLifecycle.WORKING #6

        is_account_available = self.settlement_ledger.limit_check_and_reserve_funds(order)
        if not is_account_available:
            order.lifecycle = OrderLifecycle.DONE
            order.end_reason = OrderEndReasons.REJECTED_INSUFFICIENT_FUND
            return

        wash_trade = False
        insufficient_market_depth = False
        non_crossing = False
        while order.remaining_quantity > 0:
            if order.side == Side.BUY:
                maker_order = self.order_book.get_best_ask_order()
                seller_order = maker_order
                if not maker_order or not seller_order:
                    insufficient_market_depth = True
                    break
                
                assert seller_order.price is not None

                buyer_order = order
                assert buyer_order.price is not None 

                if buyer_order.price < seller_order.price:
                    non_crossing = True
                    break
                
                if buyer_order.agent_id == seller_order.agent_id:
                    wash_trade = True
                    break
                
                trade_price = seller_order.price
                
            elif order.side == Side.SELL:
                maker_order = self.order_book.get_best_bid_order()
                buyer_order = maker_order
                if not maker_order or not buyer_order:
                    insufficient_market_depth = True
                    break

                assert buyer_order.price is not None

                seller_order = order 
                assert seller_order.price is not None

                if buyer_order.price < seller_order.price:
                    non_crossing = True
                    break

                if buyer_order.agent_id == seller_order.agent_id:
                    wash_trade = True
                    break
                
                trade_price = buyer_order.price
                
            else:
                assert False

            trade_quantity = min(buyer_order.remaining_quantity, seller_order.remaining_quantity)

            SIM_REALTIME_DATA = get_simulation_realtime_data()
            ENV_CONFIG = get_environment_configuration()

            trade = Trade(
                trade_id=self.trade_id,
                timestamp=time.time(),
                macro_tick=SIM_REALTIME_DATA.MACRO_TICK,
                micro_tick=SIM_REALTIME_DATA.MICRO_TICK,
                seller_agent_id=seller_order.agent_id,
                sell_order_id=seller_order.order_id,
                buyer_agent_id=buyer_order.agent_id,
                buy_order_id=buyer_order.order_id,
                price=trade_price,
                quantity=trade_quantity,
                fee=trade_price * trade_quantity * ENV_CONFIG.FEE_RATE_PPM // 1000000
            )

            self.__execute_trade(buyer_order, seller_order, trade)

            if maker_order.remaining_quantity == 0:
                maker_order = self.order_book.remove_order(maker_order.order_id)
                assert maker_order is not None
                maker_order.lifecycle = OrderLifecycle.DONE
                maker_order.end_reason = OrderEndReasons.FILLED

        
        if wash_trade:
            if order.side == Side.BUY: self.settlement_ledger.release_cash(order)
            elif order.side == Side.SELL: self.settlement_ledger.release_shares(order)
            else: assert False
            
            order.lifecycle = OrderLifecycle.DONE
            order.end_reason= OrderEndReasons.KILLED_WASH_TRADE
            return 
            
        if insufficient_market_depth or non_crossing:
            assert self.order_book.add_order(order)
            return

        order.lifecycle = OrderLifecycle.DONE
        order.end_reason = OrderEndReasons.FILLED

                          
    def __process_new_market_order(self, order:Order) -> None:
        # Expectations:
        # 1-agent_exist (ASSURED)
        # 2-order_type = MARKET  
        # 3-price = None
        # 4-quantity > 0 (ASSURED)
        # 5-remaining_quantity = quantity (ASSURED) 
        # 6-lifecycle = WORKING 
        # 7-end_reason = NONE (ASSURED)
        # 8-average_trade_price = None (ASSURED)
        # 9-trades = {} (ASSURED)
        assert order.order_type == OrderType.MARKET #2
        assert order.price is None #3
        assert order.lifecycle == OrderLifecycle.WORKING

        wash_trade = False
        insufficient_market_depth = False
        insufficient_funds = False
        while order.remaining_quantity > 0:
            if order.side == Side.BUY:
                maker_order = self.order_book.get_best_ask_order()
                seller_order = maker_order
                if not seller_order or not maker_order:
                    insufficient_market_depth = True
                    break

                assert seller_order.price is not None

                buyer_order = order
                assert buyer_order.price is None

                if buyer_order.agent_id == seller_order.agent_id:
                    wash_trade = True
                    break
                
                trade_price = seller_order.price
                
                possible_shares = self.settlement_ledger.market_calculate_possible_quantities(buyer_order, trade_price=trade_price)
                if possible_shares == 0:
                    insufficient_funds = True
                    break
                
            elif order.side == Side.SELL:
                maker_order = self.order_book.get_best_bid_order()
                buyer_order = maker_order
                if not buyer_order or not maker_order:
                    insufficient_market_depth = True
                    break

                assert buyer_order.price is not None

                seller_order = order
                assert seller_order.price is None

                if buyer_order.agent_id == seller_order.agent_id:
                    wash_trade = True
                    break
                
                trade_price = buyer_order.price
                
                possible_shares = self.settlement_ledger.market_calculate_possible_quantities(seller_order)

                if possible_shares == 0:
                    insufficient_funds = True
                    break
                
            else:
                assert False

            trade_quantity = min(possible_shares, maker_order.remaining_quantity)

            SIM_REALTIME_DATA = get_simulation_realtime_data()
            ENV_CONFIG = get_environment_configuration()
            
            trade = Trade(
                trade_id=self.trade_id,
                timestamp=time.time(),
                macro_tick=SIM_REALTIME_DATA.MACRO_TICK,
                micro_tick=SIM_REALTIME_DATA.MICRO_TICK,
                seller_agent_id=seller_order.agent_id,
                sell_order_id=seller_order.order_id,
                buyer_agent_id=buyer_order.agent_id,
                buy_order_id=buyer_order.order_id,
                price=trade_price,
                quantity=trade_quantity,
                fee=trade_price * trade_quantity * ENV_CONFIG.FEE_RATE_PPM // 1000000
            )

            self.__execute_trade(buyer_order, seller_order, trade)

            if maker_order.remaining_quantity == 0:
                maker_order = self.order_book.remove_order(maker_order.order_id)
                assert maker_order is not None
                maker_order.lifecycle = OrderLifecycle.DONE
                maker_order.end_reason = OrderEndReasons.FILLED

        order.lifecycle = OrderLifecycle.DONE

        if wash_trade:
            order.end_reason = OrderEndReasons.KILLED_WASH_TRADE
            return
        
        if insufficient_market_depth:
            order.end_reason = OrderEndReasons.REJECTED_INSUFFICIENT_MARKET_DEPTH
            return
        
        if insufficient_funds:
            order.end_reason = OrderEndReasons.REJECTED_INSUFFICIENT_FUND
            return

        order.end_reason = OrderEndReasons.FILLED

        
    def __execute_trade(self, buyer_order:Order, seller_order:Order, trade:Trade) -> None:
        self.settlement_ledger.settle_trade(buyer_order, seller_order, trade)
        self.storage_ledger.add_trade(trade)

        #self.__trades.append((trade.price, trade.quantity))
        self.__last_traded_price = trade.price
        self.__last_trade_volume = trade.quantity

        self.__macro_trade_value += trade.price * trade.quantity
        self.__macro_trade_volume += trade.quantity

        self.__micro_trade_count += 1
        self.__micro_trade_value += trade.price * trade.quantity
        self.__micro_trade_volume += trade.quantity
        
    
    def cancel_order(self, order_id:int) -> None:
        #Expectations
        # 1-agent_exist (ASSURED)
        # 2-order.agent_id = agent_id (ASSURED)
        # 3-order_type = Limit (ASSURED)
        # 4-price != None && price > 0 (ASSURED)
        # 5-quantity > 0 (ASSURED)
        # 6-0 < remaining_quantity < quantity (ASSURED)
        # 7-lifecycle = Working (ASSURED)
        # 8-end_reason = None (ASSURED)
        # 9-order in book

        order = self.order_book.remove_order(order_id)
        assert order is not None #9
        
        if order.side == Side.BUY:
            self.settlement_ledger.release_cash(order)
        elif order.side == Side.SELL:
            self.settlement_ledger.release_shares(order)
        else:
            assert False

        order.lifecycle = OrderLifecycle.DONE
        order.end_reason = OrderEndReasons.CANCELLED

        
    def expire_session(self) -> None:
        #Expectations
        # 1-agent_exist (ASSURED)
        # 2-order_type = Limit (ASSURED)
        # 3-price != None && price > 0 (ASSURED)
        # 4-quantity > 0 (ASSURED)
        # 5-0 < remaining_quantity < quantity (ASSURED)
        # 6-lifecycle = WORKING (ASSURED)
        # 7-end_reason = NONE (ASSURED)
        
        book = self.order_book.expire_book()
        if book is None:
            return

        bids, asks = book
        
        for bid in bids:
            self.settlement_ledger.release_cash(bid)
            bid.lifecycle = OrderLifecycle.DONE
            bid.end_reason = OrderEndReasons.EXPIRED

        for ask in asks:
            self.settlement_ledger.release_shares(ask)
            ask.lifecycle = OrderLifecycle.DONE
            ask.end_reason = OrderEndReasons.EXPIRED

        self.__macro_trade_value = 0
        self.__macro_trade_volume = 0
        
            
    def get_market_data(self) -> MarketData:
        SIM_REALTIME_DATA = get_simulation_realtime_data()
        ENV_CONFIG = get_environment_configuration()
        
        l1_bids = self.order_book.get_l1_bids()
        l1_asks = self.order_book.get_l1_asks()

        if l1_bids is not None and l1_asks is not None:
            spread = l1_asks[0] - l1_bids[0]
            mid_price = (l1_asks[0] + l1_bids[0]) // 2
            micro_price = (l1_asks[0] * l1_bids[1] + l1_bids[0] * l1_asks[1]) // (l1_bids[1] + l1_asks[1])
            
        else:
            spread = None
            mid_price = None
            micro_price= None


        l2_bids = self.order_book.get_l2_bids()
        l2_asks = self.order_book.get_l2_asks()

        N = ENV_CONFIG.INSIGHT_L2_DEPTH

        bids_depth_N = 0
        if l2_bids is not None:
            for i in range(min(N, len(l2_bids))):
                bids_depth_N += l2_bids[i][1]

        asks_depth_N = 0
        if l2_asks is not None:
            for i in range(min(N, len(l2_asks))):
                asks_depth_N += l2_asks[i][1]
        
        if bids_depth_N == asks_depth_N == 0:
            imbalance_N = None
        else:
            imbalance_N = (bids_depth_N - asks_depth_N) / (bids_depth_N + asks_depth_N) 

        last_traded_price = self.__last_traded_price
        last_trade_volume = self.__last_trade_volume
            
        trade_count = self.__micro_trade_count
        trade_volume = self.__micro_trade_volume

        if self.__macro_trade_volume == 0:
            vwap_macro = None
        else:
            vwap_macro = self.__macro_trade_value // self.__macro_trade_volume
        
        if self.__micro_trade_volume == 0:
            vwap_micro = None
        else:
            vwap_micro = self.__micro_trade_value // self.__micro_trade_volume
            
        self.__last_traded_price = None
        self.__last_trade_volume = None
        self.__micro_trade_count = 0
        self.__micro_trade_value = 0
        self.__micro_trade_volume = 0
        
        return MarketData(
            timestamp=time.time(),
            macro_tick=SIM_REALTIME_DATA.MACRO_TICK,
            micro_tick=SIM_REALTIME_DATA.MICRO_TICK,
            trade_count=trade_count,
            trade_volume=trade_volume,
            last_traded_price=last_traded_price,
            last_trade_size=last_trade_volume,
            L1_bids=l1_bids,
            L1_asks=l1_asks,
            spread=spread,
            mid_price=mid_price,
            micro_price=micro_price,
            L2_bids=l2_bids,
            L2_asks=l2_asks,
            N=N,
            bids_depth_N=bids_depth_N,
            asks_depth_N=asks_depth_N,
            imbalance_N=imbalance_N,
            vwap_macro=vwap_macro,
            vwap_micro=vwap_micro
        )

