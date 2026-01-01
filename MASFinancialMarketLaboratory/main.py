import sys
from pathlib import Path
import time

from environment import Environment
from simulation.core import SimulationEngine, SimulationInitializer


def main():
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "config.json"
    
    config_file = Path(config_path)
    if not config_file.exists():
        print(f"Error: Config file not found: {config_path}")
        print(f"Usage: python -m mas_market_laboratory.mas_market_labratory [config_path]")
        sys.exit(1)
    
    print("=" * 75)
    print("MAS Market Laboratory - Multi-Agent Simulation")
    print("=" * 75)
    print(f"Config: {config_file.absolute()}")
    print()
    
    print("Loading Configurations...", end="", flush=True)
    SimulationInitializer.INITIALIZE_CONFIGS(str(config_file.absolute()))
    time.sleep(2)
    print("\b\b\b - Completed ✔️")
    print()
    
    print("Creating Environment...", end="", flush=True)
    env = Environment()
    time.sleep(2)
    print("\b\b\b - Completed ✔️")
    print()    

    print("Creating Simulation Engine...")
    time.sleep(2)
    engine = SimulationEngine(env)
    sys.stdout.write("\033[s")
    sys.stdout.write("\033[F")
    sys.stdout.write("\033[F")
    sys.stdout.write("\033[F")
    sys.stdout.write("\033[F")
    sys.stdout.write("\033[F")
    sys.stdout.write("\033[F")
    sys.stdout.write("\033[F")
    time.sleep(1)
    print("Creating Simulation Engine - Completed ✔️")
    sys.stdout.write("\033[u")
    print()
    
    print("=" * 75)
    print("Simulation Starting...")
    print("=" * 75)
    time.sleep(3)
    sys.stdout.write("\033[s")
    sys.stdout.write("\033[F")
    print(" " * 75)
    sys.stdout.write("\033[F")
    sys.stdout.write("\033[F")
    print("\rSimulation Running...", flush=True)
    sys.stdout.write("\033[u")
    print("=" * 75)
    sys.stdout.write("\033[F")
    sys.stdout.write("\033[F")
    engine.run()
    sys.stdout.write("\033[F")
    print("Simulation Complete ✅")
    print("=" * 75)

    
if __name__ == "__main__":
    main()
