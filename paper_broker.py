import csv
import logging
from pathlib import Path
from datetime import datetime, timezone
from config import config

logger = logging.getLogger("paper_broker")


class PaperBroker:
    def __init__(self, starting_balance, position_size_usd, spread_pct, slippage_pct,
                 equity_log_path):
        self.position_size_usd = position_size_usd
        self.spread_pct = spread_pct
        self.slippage_pct = slippage_pct
        self.equity_log_path = Path(equity_log_path)
        self.open_positions = {}  # symbol -> position dict

        self.equity_log_path.parent.mkdir(parents=True, exist_ok=True)

        if self.equity_log_path.exists():
            # restore balance from the last recorded value instead of resetting --
            # otherwise every restart silently wipes any realized P&L
            try:
                with open(self.equity_log_path, "r", newline="") as f:
                    rows = list(csv.reader(f))
                if len(rows) > 1:  # header + at least one data row
                    last_balance = float(rows[-1][1])
                    self.balance = last_balance
                    logger.info("Restored balance from %s: $%.2f (was $%.2f at startup default)",
                                 self.equity_log_path, last_balance, starting_balance)
                else:
                    self.balance = starting_balance
                    self._log_equity("initialized")
            except (ValueError, IndexError) as e:
                logger.warning("Could not parse existing equity log (%s) -- starting fresh at $%.2f",
                                e, starting_balance)
                self.balance = starting_balance
                self._log_equity("initialized_after_parse_error")
        else:
            self.balance = starting_balance
            with open(self.equity_log_path, "w", newline="") as f:
                csv.writer(f).writerow(["timestamp", "balance", "event"])
            self._log_equity("initialized")

    def _log_equity(self, event=""):
        with open(self.equity_log_path, "a", newline="") as f:
            csv.writer(f).writerow([datetime.now(timezone.utc).isoformat(), self.balance, event])

    def _execution_cost_pct(self):
        return self.spread_pct + self.slippage_pct

    def open_trade(self, symbol, direction, market_price):
        if symbol in self.open_positions:
            raise RuntimeError(f"A position is already open on {symbol}.")

        cost_pct = self._execution_cost_pct() / 100
        if direction == "buy":
            fill_price = market_price * (1 + cost_pct)
        else:
            fill_price = market_price * (1 - cost_pct)

        self.open_positions[symbol] = {
            "direction": direction,
            "entry_price": fill_price,
            "size_usd": self.position_size_usd,
            "opened_at": datetime.now(timezone.utc).isoformat(),
        }
        logger.info("Paper trade opened: %s %s @ %.6f (size=$%.2f)",
                     symbol, direction, fill_price, self.position_size_usd)
        return fill_price

    def close_trade(self, symbol, market_price):
        if symbol not in self.open_positions:
            raise RuntimeError(f"No open position on {symbol} to close.")

        position = self.open_positions[symbol]
        cost_pct = self._execution_cost_pct() / 100
        direction = position["direction"]
        entry_price = position["entry_price"]
        size_usd = position["size_usd"]

        if direction == "buy":
            fill_price = market_price * (1 - cost_pct)
            pct_move = (fill_price / entry_price - 1)
        else:
            fill_price = market_price * (1 + cost_pct)
            pct_move = (entry_price - fill_price) / entry_price

        pnl_usd = size_usd * pct_move
        self.balance += pnl_usd

        logger.info("Paper trade closed: %s %s @ %.6f, pnl=$%.2f, new balance=$%.2f",
                     symbol, direction, fill_price, pnl_usd, self.balance)

        del self.open_positions[symbol]
        self._log_equity(event=f"{symbol}_closed pnl={pnl_usd:.2f}")

        return {"exit_price": fill_price, "pnl_usd": pnl_usd, "balance": self.balance}

    def has_open_position(self, symbol):
        return symbol in self.open_positions

    def equity(self, current_prices=None):
        total = self.balance
        if not current_prices:
            return total

        for symbol, position in self.open_positions.items():
            price = current_prices.get(symbol)
            if price is None:
                continue
            direction = position["direction"]
            entry_price = position["entry_price"]
            size_usd = position["size_usd"]
            if direction == "buy":
                pct_move = (price / entry_price - 1)
            else:
                pct_move = (entry_price - price) / entry_price
            total += size_usd * pct_move

        return total