from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

NonNegativeFloat = Annotated[float, Field(ge=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]


class Applicant(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "RevolvingUtilizationOfUnsecuredLines": 0.82,
        "age": 42,
        "NumberOfTime30-59DaysPastDueNotWorse": 1,
        "DebtRatio": 0.47,
        "MonthlyIncome": 5200,
        "NumberOfOpenCreditLinesAndLoans": 8,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 2,
    }})

    RevolvingUtilizationOfUnsecuredLines: NonNegativeFloat
    age: Annotated[int, Field(ge=18, le=120)]
    NumberOfTime30_59DaysPastDueNotWorse: NonNegativeInt = Field(
        alias="NumberOfTime30-59DaysPastDueNotWorse"
    )
    DebtRatio: NonNegativeFloat
    MonthlyIncome: NonNegativeFloat | None = None
    NumberOfOpenCreditLinesAndLoans: NonNegativeInt
    NumberOfTimes90DaysLate: NonNegativeInt
    NumberRealEstateLoansOrLines: NonNegativeInt
    NumberOfTime60_89DaysPastDueNotWorse: NonNegativeInt = Field(
        alias="NumberOfTime60-89DaysPastDueNotWorse"
    )
    NumberOfDependents: NonNegativeFloat | None = None

    def feature_dict(self) -> dict[str, float | int | None]:
        return self.model_dump(by_alias=True)


class Reason(BaseModel):
    code: str
    feature: str
    description: str
    contribution: float


class ScoreResponse(BaseModel):
    request_id: str
    default_probability: float
    credit_score: int
    risk_tier: Literal["A", "B", "C", "D", "E"]
    decision: Literal["APPROVE", "REFER", "DECLINE"]
    adverse_action_reasons: list[Reason]
    model_version: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None

