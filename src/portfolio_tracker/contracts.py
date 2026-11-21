from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

Verdict = Literal["ACHAT", "CONSERVER", "VENDRE"]


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


class StockEvaluation(BaseModel):
    """Contrat de résultat d'évaluation qualitative et quantitative d'une action."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    verdict: Verdict
    score: int = Field(ge=1, le=10)
    summary: str
    strengths: list[str]
    risks: list[str]
    model: str
    sources: list[str] = Field(default_factory=list)

    @field_validator("score", mode="before")
    @classmethod
    def _reject_bool_score(cls, value: Any) -> Any:
        # En Python, bool est une sous-classe de int : True/False ne doivent pas passer comme 1/0.
        if isinstance(value, bool):
            raise ValueError("Le score doit être un entier compris entre 1 et 10, pas un booléen.")
        return value


class MarketDataProvider(Protocol):
    """Contrat d'interface pour tout fournisseur de données boursières."""

    def get(self, symbol: str, *, force_refresh: bool = False) -> StockData: ...


class StockEvaluator(Protocol):
    """Contrat d'interface pour tout évaluateur d'action."""

    def evaluate(self, stock: StockData) -> StockEvaluation: ...
