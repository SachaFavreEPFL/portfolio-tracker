from typing import Protocol

from pydantic import BaseModel, ConfigDict


class StockData(BaseModel):
    """Contrat de données financières pour une action."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    name: str
    currency: str | None = None
    price: float | None = None
    fifty_two_week_low: float | None = None
    fifty_two_week_high: float | None = None
    market_cap: float | None = None
    trailing_pe: float | None = None
    forward_pe: float | None = None
    debt_to_equity: float | None = None
    profit_margins: float | None = None
    revenue_growth: float | None = None
    recommendation_key: str | None = None


class MarketDataProvider(Protocol):
    """Contrat d'interface pour tout fournisseur de données boursières."""

    def get(self, symbol: str) -> StockData: ...
