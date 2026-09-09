# MASFinancialMarketLaboratory

A Multi-Agent System (MAS) simulation framework for a financial market. This project provides a controlled laboratory environment designed to simulate continuous double auction (CDA) order book dynamics, time-varying economic signals, and the interactions of heterogeneous trading agents.

## 📄 Paper

**Multi-Agent Financial Market Laboratory: A Controlled Environment for
Market Dynamics Analysis** — single-author paper
· [`MarketDynamicsAnalysis.pdf`](MarketDynamicsAnalysis.pdf)

This laboratory was built to run one controlled experiment: what happens to a
market as trend-following capital grows? Five configurations share an
identical baseline — 100 market makers, 400 value investors, 200 noise
traders — and differ in exactly one dimension, the momentum trader count
(0 → 75 → 150 → 225 → 300). Every mechanism, parameter and economic setting
is held fixed, so any difference in outcome is attributable to composition
alone. Metrics are measured after a burn-in window and aggregated over
multiple independent seeds.

**Findings.** Instability does not grow smoothly with momentum participation;
it appears past a threshold and changes the market's regime. Volatility shifts
from intermittent bursts to a persistent high-volatility state, and liquidity
resilience degrades — mean recovery time after a liquidity shock rises roughly
six-fold from baseline to the densest configuration, while the recovery *rate*
stays near 100%. Markets keep working, but heal progressively more slowly.

At the agent level the effect is non-monotonic. Moderate momentum trading
*helps* value investors by creating exploitable mispricing: realized PnL and
risk-adjusted efficiency both peak at intermediate densities. Past the
threshold, mispricing persists long enough that value investors are forced to
carry larger inventories, efficiency collapses, and worst-case drawdowns reach
total capital loss. Median PnL is negative in *every* configuration while the
mean stays positive — profitability is concentrated in a small subset of
agents, and aggregate averages hide widespread failure.

All results are reproducible from the JSON configurations in this repository;
the paper documents the exact scenario files and seeds used.

## Features

* **Continuous Double Auction (CDA) Engine:** A price-time priority matching engine that generates transaction prices endogenously from agent order flows.
* **Heterogeneous Trading Agents:** Modular agent architectures including:
  * **Market Makers:** Provide liquidity, manage inventory risk, and maintain bid-ask spreads.
  * **Value Investors:** Trade based on fundamental value signals and perceived mispricing.
  * **Momentum Traders:** Trend-following agents that react to short-term price returns and market liquidity.
  * **Noise Traders:** Provide stochastic, uninformed order flow.
* **Economic Module:** Generates exogenous macroeconomic signals (fundamental value intervals, interest rates) without reacting to market outcomes.
* **Ledger System:** * *Settlement Ledger:* Handles trade clearing, cash balances, and inventory updates.
  * *Storage Ledger:* Persists historical market data, order book states, and agent trajectories to an SQLite database for post-simulation analysis.

## Installation

1. Ensure you have Python 3.8+ installed.
2. Clone this repository.
3. Install the required dependencies:

    pip install -r requirements.txt

## Usage

To run a simulation, you **must** provide a configuration file. Execute the `main.py` script and pass the path to your desired JSON configuration file as an argument:

    python main.py path/to/your_config.json

### Output Workflow
Upon execution, the simulation will output its initialization sequence to the console:
1. Loading Configurations
2. Creating Environment
3. Creating Simulation Engine
4. Simulation Running...
5. Simulation Complete ✅

Post-simulation, all generated market and agent data will be saved to the SQLite database path specified in your configuration file.

## Configuration Scope and Scenarios

All experiments and simulations within this laboratory are fully specified through external JSON configuration files. This design ensures absolute reproducibility and allows for controlled comparisons across different experimental setups. 

A configuration file holistically defines:
* **Market Parameters:** Tick sizes, database outputs, and price scaling.
* **Economic Settings:** The trajectory of exogenous signals, such as fundamental value drift, volatility, and interest rates.
* **Agent Population Composition:** The exact counts of each agent stereotype and the statistical distributions that govern their specific behavioral parameters (e.g., risk bounds, aggressiveness, stubbornness).

**Scenario Testing:** The configuration-driven setup is designed to study market stability and regime-dependent dynamics. For example, you can create a "baseline" configuration representing a stable market, and then create derivative configuration files where you systematically vary a single dimension—such as scaling the count of `momentum_trader` agents from 0 to 300—while holding the market structure fixed. This allows you to evaluate how specific trader densities affect volatility, liquidity resilience, and the economic utility of other agents.

## Configuration Parameters Guide

Below is an exhaustive breakdown of all required configuration parameters, their data types, and examples.

### 1. `simulation_config` (Object)
Controls the temporal structure of the simulation.
* `simulation_macro_tick` (Integer): Total number of macroeconomic time steps. *Example: 150*
* `simulation_micro_tick` (Integer): Number of trading events/ticks per macro tick. *Example: 80*
* `init_macro_tick` (Integer): Starting macro tick. *Example: 0*
* `init_micro_tick` (Integer): Starting micro tick. *Example: 0*

### 2. `environment_config` (Object)
Defines the core market rules and economic scenario variables.
* `price_scale` (Integer): Scaling factor for prices. *Example: 100000*
* `db_path` (String): Relative path to save the SQLite output database. *Example: "data/A.db"*
* `insight_l2_depth` (Integer): Limit order book depth tracking level. *Example: 25*
* `fee_rate_ppm` (Integer): Trading fee rate in parts-per-million. *Example: 1000*
* `economy_scenario_seed` (Integer): Random seed for economic generation. *Example: 1923*
* `economy_scenario_tv_initial` (Float): Initial true/fundamental value. *Example: 100.0*
* `economy_scenario_tv_long_run_mean` (Float): Long-run mean for fundamental value. *Example: 150.0*
* `economy_scenario_tv_drift` (Float): Drift term. *Example: 0.08*
* `economy_scenario_tv_mean_reversion` (Float): Mean reversion speed. *Example: 0.008*
* `economy_scenario_tv_vol` (Float): Fundamental value volatility. *Example: 1.2*
* `economy_scenario_r_initial` (Float): Initial interest rate. *Example: 0.02*
* `economy_scenario_r_long_run_mean` (Float): Long-run mean interest rate. *Example: 0.02*
* `economy_scenario_r_mean_reversion` (Float): Rate mean reversion. *Example: 0.2*
* `economy_scenario_r_vol` (Float): Rate volatility. *Example: 0.005*
* `economy_scenario_tv_interval_base_width` (Float): Base width of the value interval. *Example: 10.0*
* `economy_scenario_tv_interval_vol` (Float): Volatility of the interval. *Example: 2.5*
* `economy_scenario_term_curve_slope` (Float): Yield curve slope. *Example: 0.001*
* `economy_scenario_term_curve_curvature` (Float): Yield curve curvature. *Example: 0.0*
* `economy_scenario_deposit_terms` (Array of Integers): Available deposit terms. *Example: [3, 7, 15]*

### 3. `agents_config` (Object)
Defines populations, initial states, and exhaustive agent-specific behavioral parameters.

#### Global Settings & Initial Endowments
* **`global`**:
  * `random_seed` (Integer): Global agent random seed. *Example: 2001*
* **`initial_endowment`**: Sets starting cash and shares.
  * `market_maker` (Object):
    * `cash` (Distribution Object): e.g., `{"distribution": "lognormal", "mean": 1000000, "std": 0.15}`
    * `shares` (Distribution Object): e.g., `{"distribution": "uniform", "min": 8000, "max": 12000}`
  * `others` (Object): (Applied to all non-market-maker agents)
    * `cash` (Distribution Object): e.g., `{"distribution": "lognormal", "mean": 250000, "std": 0.6}`
    * `shares` (Distribution Object): e.g., `{"distribution": "uniform", "min": 2000, "max": 3000}`

#### Agent Stereotype Specific Configurations
Every agent type requires a `count` (Integer) representing the population size, and a `parameters` object where every key maps to a statistical distribution defining agent behaviors.

**`market_maker`**
* `count` (Integer): e.g., *100*
* `parameters` (Object):
  * `target_inventory_fraction`: Ideal portfolio ratio of shares to cash. *Example: {"distribution": "uniform", "min": 0.45, "max": 0.55}*
  * `risk_lower_bound`: Threshold triggering inventory acquisition. *Example: {"distribution": "uniform", "min": 0.15, "max": 0.30}*
  * `risk_upper_bound`: Threshold triggering inventory offloading. *Example: {"distribution": "uniform", "min": 0.70, "max": 0.85}*
  * `stabilization_tolerance`: Allowable deviation before adjusting quotes. *Example: {"distribution": "uniform", "min": 0.07, "max": 0.15}*
  * `spread_size`: Bid-ask spread maintained by the MM. *Example: {"distribution": "discrete_uniform", "values": [4, 5, 6, 7, 8]}*
  * `order_size`: Volume of quotes provided at each level. *Example: {"distribution": "discrete_uniform", "values": [5, 10, 15, 20]}*
  * `wait_time`: Ticks waited before refreshing quotes. *Example: {"distribution": "discrete_uniform", "values": [7, 10, 15, 20]}*
  * `skew_factor`: Pricing skew when inventory boundaries are hit. *Example: {"distribution": "discrete_uniform", "values": [80, 100, 120]}*

**`value_investor`**
* `count` (Integer): e.g., *400*
* `parameters` (Object):
  * `initial_optimism`: Bias for selecting fundamental value within interval. *Example: {"distribution": "uniform", "min": 0.2, "max": 0.8}*
  * `stubbornness`: Resistance to updating beliefs based on PnL. *Example: {"distribution": "uniform", "min": 0.2, "max": 0.7}*
  * `belief_update_rate`: Speed of adjusting optimism. *Example: {"distribution": "uniform", "min": 0.003, "max": 0.02}*
  * `mispricing_threshold`: Minimum perceived mispricing needed to trigger a trade. *Example: {"distribution": "uniform", "min": 0.0005, "max": 0.0015}*
  * `max_position_fraction`: Max portfolio allocation per trade action. *Example: {"distribution": "uniform", "min": 0.4, "max": 0.7}*
  * `deposit_affinity`: Likelihood to park cash in deposits. *Example: {"distribution": "uniform", "min": 0.02, "max": 0.05}*
  * `deposit_allocation_fraction`: Fraction of cash allocated if affinity triggers. *Example: {"distribution": "uniform", "min": 0.3, "max": 0.6}*
  * `deposit_horizon_preference`: Preference scale for longer deposit terms. *Example: {"distribution": "uniform", "min": 0.3, "max": 0.7}*
  * `max_order_size`: Absolute maximum shares per order. *Example: {"distribution": "uniform", "min": 30, "max": 250}*
  * `patience_discount`: Tolerance discount for execution delays. *Example: {"distribution": "uniform", "min": 0.00001, "max": 0.00004}*
  * `patience_premium`: Required premium to justify waiting. *Example: {"distribution": "uniform", "min": 0.00001, "max": 0.00004}*

**`momentum_trader`**
* `count` (Integer): e.g., *0* (to test baselines) or *150*
* `parameters` (Object):
  * `momentum_window`: Time window (in ticks) to calculate price trends. *Example: {"distribution": "discrete_uniform", "values": [3, 5, 7, 10]}*
  * `entry_threshold`: Minimum trend strength to trigger entry. *Example: {"distribution": "uniform", "min": 0.02, "max": 0.05}*
  * `exit_threshold`: Trend weakness threshold to exit position. *Example: {"distribution": "uniform", "min": 0.005, "max": 0.015}*
  * `aggressiveness`: Multiplier for order size and urgency based on signal. *Example: {"distribution": "uniform", "min": 0.4, "max": 0.8}*
  * `max_exposure_fraction`: Maximum portfolio allocation to momentum trades. *Example: {"distribution": "uniform", "min": 0.5, "max": 0.8}*
  * `directional_bias`: Inherent bias towards long or short momentum. *Example: {"distribution": "uniform", "min": -0.1, "max": 0.1}*
  * `liquidity_baseline`: Market depth threshold; falling below this increases their trading intensity. *Example: {"distribution": "discrete_uniform", "values": [80, 100, 120]}*

**`noise_trader`**
* `count` (Integer): e.g., *200*
* `parameters` (Object):
  * `p_trade`: Probability of taking action on any given tick. *Example: {"distribution": "uniform", "min": 0.04, "max": 0.12}*
  * `p_buy`: Probability that the action is a buy (vs sell). *Example: {"distribution": "uniform", "min": 0.45, "max": 0.55}*
  * `p_market_order`: Probability of submitting a market order rather than limit. *Example: {"distribution": "uniform", "min": 0.03, "max": 0.1}*
  * `min_quantity`: Minimum volume for a noise trade. *Example: {"distribution": "discrete_uniform", "values": [1, 2, 5]}*
  * `max_quantity`: Maximum volume for a noise trade. *Example: {"distribution": "uniform", "min": 30, "max": 80}*
  * `price_offset_ticks`: Offset from current market price for limit orders. *Example: {"distribution": "discrete_uniform", "values": [5, 6, 7, 8, 9, 10]}*

---

## 📜 License

This project is licensed under the MIT License. Copyright (c) 2026 Buğrahan İmal. You are free to use, copy, modify, merge, publish, and distribute this software as per the license conditions.
