from __future__ import annotations
from typing import Dict, Optional, Tuple
import sqlite3
import json

from environment.models import Account, Deposit, EconomyInsight, MarketData, Order, Trade 
from environment.configs import get_environment_configuration

from simulation.configs import get_simulation_realtime_data



class StorageLedger:
    accounts:Dict[int, Account]
    orders:Dict[int, Order] #OrderID -> Order
    trades:Dict[int, Trade] #TradeID -> Trade
    deposits:Dict[int, Deposit] #DepositID -> Deposit
    economy_insights:Dict[int, EconomyInsight] #macro_tick -> Deposit
    market_data:Dict[Tuple[int, int], MarketData] #(macro_tick, micro_tick) -> MarketData 
    
    db_path:str
    connection:sqlite3.Connection

    __last_flush_macro_tick:int
    
    
    def __init__(self) -> None:
        self.accounts = {}
        self.orders = {}
        self.trades = {}
        self.deposits = {}
        self.economy_insights = {}
        self.market_data = {}
        self.__last_flush_macro_tick = -1

        ENV_CONFIG = get_environment_configuration()
        self.db_path = ENV_CONFIG.DB_PATH
        self.connection = sqlite3.connect(self.db_path)

        self.connection.execute("PRAGMA journal_mode=WAL;")
        self.connection.execute("PRAGMA synchronous=NORMAL;")
        self.connection.execute("PRAGMA foreign_keys = ON;")

        self.__create_sheme()



    def add_account(self, account:Account) -> bool:
        if account.account_id in self.accounts:
            return False

        self.accounts[account.account_id] = account
        return True
    
        
    def add_order(self, order:Order) -> bool:
        if order.order_id in self.orders:
            return False

        self.orders[order.order_id] = order
        return True

    
    def add_trade(self, trade:Trade) -> bool:
        if trade.trade_id in self.trades:
            return False

        self.trades[trade.trade_id] = trade
        return True


    def add_deposit(self, deposit:Deposit) -> bool:
        if deposit.deposit_id in self.deposits:
            return False

        self.deposits[deposit.deposit_id] = deposit
        return True

    
    def add_economy_insight(self, economy_insight:EconomyInsight) -> bool:
        if economy_insight.macro_tick in self.economy_insights:
            return False

        self.economy_insights[economy_insight.macro_tick] = economy_insight
        return True


    def add_market_data(self, market_data:MarketData) -> bool:
        if (market_data.macro_tick, market_data.micro_tick) in self.market_data:
            return False

        self.market_data[(market_data.macro_tick, market_data.micro_tick)] = market_data
        return True
    

    def get_account(self, account_id:int) -> Optional[Account]:
        return self.accounts.get(account_id)

    
    def get_order(self, order_id:int) -> Optional[Order]:
        return self.orders.get(order_id)

    
    def get_trade(self, trade_id:int) -> Optional[Trade]:
        return self.trades.get(trade_id)


    def get_deposit(self, deposit_id:int) -> Optional[Deposit]:
        return self.deposits.get(deposit_id)
    

    def get_economy_insight(self, macro_tick:int) -> Optional[EconomyInsight]:
        return self.economy_insights.get(macro_tick)


    def get_market_data(self, hybrid_time:Tuple[int, int]) -> Optional[MarketData]:
        return self.market_data.get(hybrid_time)
    
    
    def flush(self) -> bool:
        SIM_REALTIME_DATA = get_simulation_realtime_data()
        if self.__last_flush_macro_tick == SIM_REALTIME_DATA.MACRO_TICK:
            return False

        cursor = self.connection.cursor()
        
        for account in self.accounts.values():
            self.__record_account(cursor, account)
        
        for order in self.orders.values():
            self.__record_order(cursor, order)
        
        for trade in self.trades.values():
            self.__record_trade(cursor, trade)

        for deposit in self.deposits.values():
            self.__record_deposit(cursor, deposit)
            
        for economy_insight in self.economy_insights.values():
            self.__record_economy_insight(cursor, economy_insight)

        for market_data in self.market_data.values():
            self.__record_market_data(cursor, market_data)
            
        self.connection.commit()
        
        self.orders.clear()
        self.trades.clear()
        self.deposits.clear()
        self.economy_insights.clear()
        self.market_data.clear()
        
        self.__last_flush_macro_tick = SIM_REALTIME_DATA.MACRO_TICK
        
        return True

    
    def close(self) -> None:
        self.connection.close()


    def __create_sheme(self) -> None:
        cursor = self.connection.cursor()
        
        self.__create_order_table(cursor)
        self.__create_trade_table(cursor)
        self.__create_account_table(cursor)
        self.__create_economy_insight_table(cursor)
        self.__create_deposit_table(cursor)
        self.__create_market_data_table(cursor)
        
        cursor.close()
        self.connection.commit()


    def __create_order_table(self, cursor:sqlite3.Cursor) -> None:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS orders (
            order_id INTEGER PRIMARY KEY,
            agent_id INTEGER  NOT NULL,

            timestamp REAL NOT NULL,
            macro_tick INTEGER NOT NULL,
            micro_tick INTEGER NOT NULL,

            order_type TEXT NOT NULL,
            side TEXT NOT NULL,

            quantity INTEGER NOT NULL,
            price INTEGER,

            lifecycle TEXT NOT NULL,
            end_reason TEXT NOT NULL,

            remaining_quantity INTEGER NOT NULL
            );
            """
        )

        
    def __create_trade_table(self, cursor:sqlite3.Cursor) -> None:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
            trade_id INTEGER PRIMARY KEY,
            timestamp  REAL NOT NULL,
            macro_tick INTEGER NOT NULL,
            micro_tick INTEGER NOT NULL,

            buyer_agent_id INTEGER NOT NULL,
            buy_order_id INTEGER NOT NULL,
            seller_agent_id INTEGER NOT NULL,
            sell_order_id INTEGER NOT NULL,

            price INTEGER NOT NULL,
            quantity INTEGER NOT NULL,
            fee INTEGER NOT NULL
            );
            """
        )


    def __create_account_table(self, cursor:sqlite3.Cursor) -> None:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS accounts (
            macro_tick INTEGER NOT NULL,
            account_id INTEGER NOT NULL,
            agent_id INTEGER NOT NULL,

            cash INTEGER NOT NULL,
            shares INTEGER NOT NULL,

            reserved_cash INTEGER NOT NULL,
            reserved_shares INTEGER NOT NULL,
            deposited_cash INTEGER NOT NULL,

            PRIMARY KEY (macro_tick, account_id)
            );
            """
        )        


    def __create_economy_insight_table(self, cursor:sqlite3.Cursor) -> None:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS economy_insights (
            macro_tick INTEGER PRIMARY KEY,
            true_value INTEGER NOT NULL,
            short_rate REAL NOT NULL,
            width REAL NOT NULL,
            tv_lower_bound INTEGER NOT NULL,
            tv_upper_bound INTEGER NOT NULL,
            deposit_rates TEXT NOT NULL
            );
            """
        )


    def __create_deposit_table(self, cursor:sqlite3.Cursor) -> None:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS deposits (
            deposit_id INTEGER PRIMARY KEY,
            agent_id INTEGER NOT NULL,
            timestamp REAL NOT NULL,
            creation_macro_tick INTEGER NOT NULL,
            maturity_macro_tick INTEGER NOT NULL,
            deposited_cash INTEGER NOT NULL,
            interest_rate REAL NOT NULL,
            matured_cash INTEGER NOT NULL
            );
            """
        )


    def __create_market_data_table(self, cursor:sqlite3.Cursor) -> None:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS market_data (
            macro_tick INTEGER NOT NULL,
            micro_tick INTEGER NOT NULL,
            timestamp REAL NOT NULL,
            trade_count INTEGER NOT NULL,
            trade_volume INTEGER NOT NULL,
            last_traded_price INTEGER,
            last_trade_size INTEGER,
            l1_bids TEXT,
            l1_asks TEXT,
            spread INTEGER,
            mid_price INTEGER,
            micro_price INTEGER,
            l2_bids TEXT,
            l2_asks TEXT,
            N INTEGER NOT NULL,
            bids_depth_N INTEGER NOT NULL,
            asks_depth_N INTEGER NOT NULL,
            imbalance_N REAL,
            vwap_macro INTEGER,
            vwap_micro INTEGER
            );
            """
        )
    
        
    def __record_order(self, cursor:sqlite3.Cursor, order:Order) -> None:
        cursor.execute(
            """
            INSERT INTO orders (
            order_id,
            agent_id,
            timestamp,
            macro_tick,
            micro_tick,
            order_type,
            side,
            quantity,
            price,
            lifecycle,
            end_reason,
            remaining_quantity
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                order.order_id,
                order.agent_id,
                order.timestamp,
                order.macro_tick,
                order.micro_tick,
                order.order_type.name,
                order.side.name,
                order.quantity,
                order.price,
                order.lifecycle.name,
                order.end_reason.name,
                order.remaining_quantity
            )
        )


    def __record_trade(self, cursor:sqlite3.Cursor, trade:Trade) -> None:
        cursor.execute(
            """
            INSERT INTO trades (
            trade_id,
            timestamp,
            macro_tick,
            micro_tick,
            buyer_agent_id,
            buy_order_id,
            seller_agent_id,
            sell_order_id,
            price,
            quantity,
            fee
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                trade.trade_id,
                trade.timestamp,
                trade.macro_tick,
                trade.micro_tick,
                trade.buyer_agent_id,
                trade.buy_order_id,
                trade.seller_agent_id,
                trade.sell_order_id,
                trade.price,
                trade.quantity,
                trade.fee
            )
        )


    def __record_account(self, cursor:sqlite3.Cursor, account:Account) -> None:
        SIM_REALTIME_DATA = get_simulation_realtime_data()
        cursor.execute(
            """
            INSERT INTO accounts (
            macro_tick,
            account_id,
            agent_id,
            cash,
            shares,
            reserved_cash,
            reserved_shares,
            deposited_cash
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                SIM_REALTIME_DATA.MACRO_TICK,
                account.account_id,
                account.agent_id,
                account.cash,
                account.shares,
                account.get_total_reserved_cash(),
                account.get_total_reserved_shares(),
                account.get_total_deposited_cash()
            )
        )


    def __record_economy_insight(self, cursor:sqlite3.Cursor, economy_insight:EconomyInsight) -> None:
        cursor.execute(
            """
            INSERT INTO economy_insights (
            macro_tick,
            true_value,
            short_rate,
            width,
            tv_lower_bound,
            tv_upper_bound,
            deposit_rates
            )
            VALUES(?, ?, ?, ?, ?, ?, ?);
            """,
            (
                economy_insight.macro_tick,
                economy_insight.true_value,
                economy_insight.short_rate,
                economy_insight.width,
                economy_insight.tv_interval[0],
                economy_insight.tv_interval[1],
                json.dumps(economy_insight.deposit_rates)
            )
        )


    def __record_deposit(self, cursor:sqlite3.Cursor, deposit:Deposit) -> None:
        cursor.execute(
            """
            INSERT INTO deposits (
            deposit_id,
            agent_id,
            timestamp,
            creation_macro_tick,
            maturity_macro_tick,
            deposited_cash,
            interest_rate,
            matured_cash
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                deposit.deposit_id,
                deposit.agent_id,
                deposit.timestamp,
                deposit.creation_macro_tick,
                deposit.maturity_macro_tick,
                deposit.deposited_cash,
                deposit.interest_rate,
                deposit.matured_cash
            )
        )


    def __record_market_data(self, cursor:sqlite3.Cursor, market_data:MarketData) -> None:
        cursor.execute(
            """
            INSERT INTO market_data (
            macro_tick,
            micro_tick,
            timestamp,
            trade_count,
            trade_volume,
            last_traded_price,
            last_trade_size,
            l1_bids,
            l1_asks,
            spread,
            mid_price,
            micro_price,
            l2_bids,
            l2_asks,
            N,
            bids_depth_N,
            asks_depth_N,
            imbalance_N,
            vwap_macro,
            vwap_micro
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                market_data.macro_tick,
                market_data.micro_tick,
                market_data.timestamp,
                market_data.trade_count,
                market_data.trade_volume,
                market_data.last_traded_price,
                market_data.last_trade_size,
                json.dumps(market_data.L1_bids),
                json.dumps(market_data.L1_asks),
                market_data.spread,
                market_data.mid_price,
                market_data.micro_price,
                json.dumps(market_data.L2_bids),
                json.dumps(market_data.L2_asks),
                market_data.N,
                market_data.bids_depth_N,
                market_data.asks_depth_N,
                market_data.imbalance_N,
                market_data.vwap_macro,
                market_data.vwap_micro
            )
        )
