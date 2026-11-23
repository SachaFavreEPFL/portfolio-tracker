from pathlib import Path
from typing import Any, Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_COMMENT_LENGTH = 200

Verdict = Literal["ACHAT", "CONSERVER", "VENDRE"]

VERDICT_DISPLAY_LABELS: dict[Verdict, str] = {
    "ACHAT": "Achat",
    "CONSERVER": "Conserver",
    "VENDRE": "Vendre",
}


def format_verdict(verdict: Verdict) -> str:
    """Formate un verdict canonique pour l'affichage dans un tableur."""
    return VERDICT_DISPLAY_LABELS[verdict]


def expected_verdict_for_score(score: int) -> Verdict:
    """Retourne le verdict canonique attendu pour un score compris entre 0 et 10."""
    if score <= 3:
        return "VENDRE"
    if score <= 6:
        return "CONSERVER"
    return "ACHAT"


class StockData(BaseModel):
    """Contrat de données financières pour une action."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

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

    @field_validator("symbol", "name", mode="before")
    @classmethod
    def _require_non_empty_text(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("Une chaîne de caractères non vide est attendue.")
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Ce champ ne peut pas être vide.")
        return cleaned


class StockEvaluation(BaseModel):
    """Contrat de résultat d'évaluation qualitative et quantitative d'une action."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    verdict: Verdict
    score: int = Field(ge=0, le=10)
    comment: str
    summary: str
    strengths: list[str]
    risks: list[str]
    model: str
    sources: list[str] = Field(default_factory=list)

    @field_validator("symbol", "summary", "model", mode="before")
    @classmethod
    def _require_non_empty_text(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("Une chaîne de caractères non vide est attendue.")
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Ce champ ne peut pas être vide.")
        return cleaned

    @field_validator("score", mode="before")
    @classmethod
    def _reject_bool_score(cls, value: Any) -> Any:
        # En Python, bool est une sous-classe de int : True/False ne doivent pas passer comme 1/0.
        if isinstance(value, bool):
            raise ValueError("Le score doit être un entier compris entre 0 et 10, pas un booléen.")
        return value

    @field_validator("comment", mode="before")
    @classmethod
    def _normalize_comment(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("Le commentaire doit être une chaîne de caractères.")
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("Le commentaire ne peut pas être vide.")
        if len(normalized) <= MAX_COMMENT_LENGTH:
            return normalized
        limit = MAX_COMMENT_LENGTH - 1
        truncated = normalized[:limit]
        # Évite de couper un mot au milieu lorsqu'un espace est disponible avant la limite.
        if normalized[limit] != " " and " " in truncated:
            candidate = truncated.rsplit(" ", 1)[0].rstrip(" ,;:-")
            if candidate:
                return f"{candidate}…"
        return f"{truncated.rstrip()}…"

    @model_validator(mode="after")
    def _validate_score_verdict_coherence(self) -> Self:
        expected = expected_verdict_for_score(self.score)
        if self.verdict != expected:
            raise ValueError(
                f"Verdict '{self.verdict}' incohérent avec le score {self.score} "
                f"(attendu : '{expected}' pour 0-3=VENDRE, 4-6=CONSERVER, 7-10=ACHAT)."
            )
        return self

    @property
    def decision_label(self) -> str:
        return format_verdict(self.verdict)


class PortfolioRow(BaseModel):
    """Contrat d'une ligne d'action lue depuis un fichier portefeuille."""

    model_config = ConfigDict(frozen=True)

    row_index: int
    symbol: str


class PortfolioRowResult(BaseModel):
    """Contrat du résultat d'analyse d'une ligne de portefeuille."""

    model_config = ConfigDict(frozen=True)

    row_index: int
    symbol: str
    score: int | None = None
    comment: str | None = None
    decision: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class PortfolioUpdateResult(BaseModel):
    """Contrat de résultat global de la mise à jour d'un portefeuille Numbers."""

    model_config = ConfigDict(frozen=True)

    input_path: Path
    output_path: Path
    sheet_name: str
    table_name: str
    decision_column_created: bool = False
    saved: bool = True
    results: list[PortfolioRowResult] = Field(default_factory=list)

    @property
    def succeeded(self) -> list[PortfolioRowResult]:
        return [item for item in self.results if item.ok]

    @property
    def failed(self) -> list[PortfolioRowResult]:
        return [item for item in self.results if not item.ok]


class MarketDataProvider(Protocol):
    """Contrat d'interface pour tout fournisseur de données boursières."""

    def get(self, symbol: str, *, force_refresh: bool = False) -> StockData: ...


class StockEvaluator(Protocol):
    """Contrat d'interface pour tout évaluateur d'action."""

    def evaluate(self, stock: StockData) -> StockEvaluation: ...
