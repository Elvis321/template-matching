import csv
import logging
from pathlib import Path
from datetime import datetime, timezone
from config import config

logger = logging.getLogger("paper_broker")


class PaperBroker:
    def __init__(self, starting_balance, position_size_usd, spread_pct, slippage_pct,
                 equity_log_path):
        self.balance = starting_balance
        self.position_size_usd = position_size_usd
        self.spread_pct = spread_pct
        self.slippage_pct = slippage_pct
        self.equity_log_path = Path(equity_log_path)
        self.open_position = None  # only one position at a time

        self.equity_log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.equity_log_path.exists():
            with open(self.equity_log_path, "w", newline="") as f:
                csv.writer(f).writerow(["timestamp", "balance", "event"])
            self._log_equity("initialized")

    def _log_equity(self, event=""):
        with open(self.equity_log_path, "a", newline="") as f:
            csv.writer(f).writerow([datetime.now(timezone.utc).isoformat(), self.balance, event])

    def _execution_cost_pct(self):
        return self.spread_pct + self.slippage_pct

    def open_trade(self, direction, market_price):
        """direction: 'buy' or 'sell'. Returns the simulated fill price after cost."""
        if self.open_position is not None:
            raise RuntimeError("A position is already open -- close it before opening another.")

        cost_pct = self._execution_cost_pct() / 100
        if direction == "buy":
            fill_price = market_price * (1 + cost_pct)
        else:
            fill_price = market_price * (1 - cost_pct)

        self.open_position = {
            "direction": direction,
            "entry_price": fill_price,
            "size_usd": self.position_size_usd,
            "opened_at": datetime.now(timezone.utc).isoformat(),
        }
        logger.info("Paper trade opened: %s @ %.6f (size=$%.2f)",
                     direction, fill_price, self.position_size_usd)
        return fill_price

    def close_trade(self, market_price):
        if self.open_position is None:
            raise RuntimeError("No open position to close.")

        cost_pct = self._execution_cost_pct() / 100
        direction = self.open_position["direction"]
        entry_price = self.open_position["entry_price"]
        size_usd = self.open_position["size_usd"]

        if direction == "buy":
            fill_price = market_price * (1 - cost_pct)
            pct_move = (fill_price / entry_price - 1)
        else:
            fill_price = market_price * (1 + cost_pct)
            pct_move = (entry_price - fill_price) / entry_price

        pnl_usd = size_usd * pct_move
        self.balance += pnl_usd

        logger.info("Paper trade closed: %s @ %.6f, pnl=$%.2f, new balance=$%.2f",
                     direction, fill_price, pnl_usd, self.balance)

        self.open_position = None
        self._log_equity(event=f"trade_closed pnl={pnl_usd:.2f}")

        return {"exit_price": fill_price, "pnl_usd": pnl_usd, "balance": self.balance}

    def has_open_position(self):
        return self.open_position is not None

    def equity(self, current_market_price=None):
        """Balance plus unrealized PnL on any open position, if a current price is given."""
        if self.open_position is None or current_market_price is None:
            return self.balance

        direction = self.open_position["direction"]
        entry_price = self.open_position["entry_price"]
        size_usd = self.open_position["size_usd"]

        if direction == "buy":
            pct_move = (current_market_price / entry_price - 1)
        else:
            pct_move = (entry_price - current_market_price) / entry_price

        return self.balance + size_usd * pct_move