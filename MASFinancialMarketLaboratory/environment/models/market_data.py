from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

from environment.views import MarketDataView
from environment.configs import get_environment_configuration



@dataclass(frozen=True)
class MarketData:
    timestamp:float
    macro_tick:int
    micro_tick:int

    trade_count:int
    trade_volume:int
    
    last_traded_price:Optional[int]
    last_trade_size:Optional[int]
    
    L1_bids:Optional[Tuple[int, int, int]] #price - size - #of orders
    L1_asks:Optional[Tuple[int, int, int]] #price - size - #of orders

    spread:Optional[int]
    mid_price:Optional[int]
    micro_price:Optional[int]
    
    L2_bids:Optional[Tuple[Tuple[int, int, int], ...]] # For each price level = price - size - #of orders
    L2_asks:Optional[Tuple[Tuple[int, int, int], ...]] # For each price level = price - size #of orders

    N:int
    bids_depth_N:int
    asks_depth_N:int
    imbalance_N:Optional[float]
    vwap_macro:Optional[int]
    vwap_micro:Optional[int]

    
    def create_view(self) -> MarketDataView:
        ENV_CONFIG = get_environment_configuration()

        last_traded_price = None
        if self.last_traded_price is not None:
            last_traded_price = self.last_traded_price / ENV_CONFIG.PRICE_SCALE

        L1_bids = None
        if self.L1_bids is not None:
            L1_bids = (self.L1_bids[0] / ENV_CONFIG.PRICE_SCALE, self.L1_bids[1], self.L1_bids[2])

        L1_asks = None
        if self.L1_asks is not None:
            L1_asks = (self.L1_asks[0] / ENV_CONFIG.PRICE_SCALE, self.L1_asks[1], self.L1_asks[2])

        spread = None
        if self.spread is not None:
            spread = self.spread / ENV_CONFIG.PRICE_SCALE

        mid_price = None
        if self.mid_price is not None:
            mid_price = self.mid_price / ENV_CONFIG.PRICE_SCALE

        micro_price = None
        if self.micro_price is not None:
            micro_price = self.micro_price / ENV_CONFIG.PRICE_SCALE

        L2_bids = None
        if self.L2_bids is not None:
            L2_bids = tuple((p / ENV_CONFIG.PRICE_SCALE, v, c) for p, v, c in self.L2_bids)

        L2_asks = None
        if self.L2_asks is not None:
            L2_asks = tuple((p / ENV_CONFIG.PRICE_SCALE, v, c) for p, v, c in self.L2_asks)

        vwap_macro = None
        if self.vwap_macro is not None:
            vwap_macro = self.vwap_macro / ENV_CONFIG.PRICE_SCALE

        vwap_micro = None
        if self.vwap_micro is not None:
            vwap_micro = self.vwap_micro / ENV_CONFIG.PRICE_SCALE
            
            
        return MarketDataView(
            timestamp=self.timestamp,
            macro_tick=self.macro_tick,
            micro_tick=self.micro_tick,
            trade_count=self.trade_count,
            trade_volume=self.trade_volume,
            last_traded_price=last_traded_price,
            last_trade_size=self.last_trade_size,
            L1_bids=L1_bids,
            L1_asks=L1_asks,
            spread=spread,
            mid_price=mid_price,
            micro_price=micro_price,
            L2_bids=L2_bids,
            L2_asks=L2_asks,
            N=self.N,
            bids_depth_N=self.bids_depth_N,
            asks_depth_N=self.asks_depth_N,
            imbalance_N=self.imbalance_N,
            vwap_macro=vwap_macro,
            vwap_micro=vwap_micro
        )
