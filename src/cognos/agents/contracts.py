"""Output contracts: the typed recommendation each agent must return.

Contracts use plain JSON types (str / float / bool / lists / nested objects, no free-form maps) so
they convert cleanly to the JSON Schema handed to every backend — `claude -p --json-schema`,
Anthropic structured outputs, and the OpenAI-compatible `submit_answer` function. ``extra="forbid"``
makes an agent that invents fields fail validation (and retry with the error).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Confidence = Literal["low", "medium", "high"]
Severity3 = Literal["low", "medium", "high"]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Claim(Contract):
    statement: str
    evidence: list[str] = Field(description="Ids from context.facts that support the statement")
    confidence: Confidence = "medium"


class ChallengeResponse(Contract):
    challenge_id: str = Field(description="The id of a challenge in context.challenges")
    response: str = Field(description="How you addressed it, or why you disagree")
    changed_recommendation: bool


class AgentOutput(Contract):
    summary: str = Field(description="Your recommendation in 2-4 plain sentences")
    uncertainties: list[str] = Field(
        default_factory=list, description="What you are unsure about and why")
    responses_to_challenges: list[ChallengeResponse] = Field(
        default_factory=list, description="Exactly one entry per item in context.challenges")


# --- explore: data analyst ---------------------------------------------------------------
class ColumnDecision(Contract):
    column: str
    decision: Literal["keep", "exclude"]
    reason: str
    evidence: list[str] = Field(default_factory=list)


class DataAnalystOutput(AgentOutput):
    column_decisions: list[ColumnDecision] = Field(
        description="One decision per leakage suspect and per column you recommend excluding")
    data_quality: list[Claim] = Field(default_factory=list)
    questions_for_sponsor: list[str] = Field(default_factory=list)


# --- ideate: design lead --------------------------------------------------------------------
class FrameworkChoice(Contract):
    framework: str = Field(description="A framework id from context.framework_assessment")
    decision: Literal["primary", "candidate", "challenger", "rejected"]
    reason: str


class SlateItem(Contract):
    family: str = Field(description="One of context.fittable_families")
    feature_strategy: Literal["top", "all"]
    role: Literal["candidate", "challenger"]
    priority: float = Field(ge=0.0, le=1.0)
    rationale: str


class TransformProposal(Contract):
    name: str
    expr: str = Field(description="Expression over existing feature columns using np.<fn> and arithmetic")
    rationale: str


class SponsorQuestion(Contract):
    question: str
    category: Literal["design", "data"]
    design_field: Literal["use_case", "horizon", "default_definition", "segment", "none"] = "none"
    why_it_matters: str


class DesignLeadOutput(AgentOutput):
    frameworks: list[FrameworkChoice]
    slate: list[SlateItem]
    transforms: list[TransformProposal] = Field(default_factory=list)
    sponsor_questions: list[SponsorQuestion] = Field(default_factory=list)


# --- model: modeler ---------------------------------------------------------------------------
class SignCheck(Contract):
    feature: str
    expected: Literal["+", "-", "none"]
    observed: Literal["+", "-", "0", "n/a"]
    assessment: Literal["consistent", "wrong_sign", "no_prior", "not_applicable"]


class ModelerOutput(AgentOutput):
    champion: str = Field(description="The label of one candidate in context.admissible_set")
    rationale: str
    sign_checks: list[SignCheck] = Field(default_factory=list)
    concerns: list[Claim] = Field(default_factory=list)


class HyperParam(Contract):
    name: str
    value: str = Field(description="The value as text, e.g. '0.1' or '200'")


class ExperimentProposal(Contract):
    stop: bool = Field(description="true when no further experiment is worth running")
    family: str = ""
    hyperparams: list[HyperParam] = Field(default_factory=list)
    transforms: list[TransformProposal] = Field(default_factory=list)
    rationale: str = ""


# --- backtest: outcomes analyst -------------------------------------------------------------
class AgentFinding(Contract):
    id: str
    severity: Severity3
    category: str
    message: str
    evidence: list[str] = Field(description="Ids from context.facts")


class OutcomesAnalystOutput(AgentOutput):
    discrimination: str
    calibration: str
    stability: str
    findings: list[AgentFinding] = Field(default_factory=list)


# --- validate: independent validator --------------------------------------------------------
class ValidatorFinding(AgentFinding):
    target_stage: Literal["explore", "ideate", "model", "none"] = Field(
        description="The stage whose work must change to resolve this finding")
    remedy: str


class ValidatorOutput(AgentOutput):
    assessment: str
    findings: list[ValidatorFinding] = Field(default_factory=list)
    recommendation: Literal["approve", "approve_with_conditions", "send_back"]
    conditions: list[str] = Field(default_factory=list)


# --- comply: model-risk analyst -------------------------------------------------------------
class RiskAnalystOutput(AgentOutput):
    readiness: Literal["ready", "ready_with_actions", "not_ready"]
    narrative: str
    priority_actions: list[Claim] = Field(default_factory=list)


# --- document: technical writer -------------------------------------------------------------
SECTIONS = ["executive_summary", "methodology_rationale", "alternatives_considered",
            "limitations", "use_and_monitoring"]


class Section(Contract):
    section: Literal["executive_summary", "methodology_rationale", "alternatives_considered",
                     "limitations", "use_and_monitoring"]
    markdown: str = Field(description="Prose; every number must be a {{fact:<id>}} placeholder")


class WriterOutput(AgentOutput):
    sections: list[Section]


CONTRACTS: dict[str, type[Contract]] = {
    "data_analyst": DataAnalystOutput,
    "design_lead": DesignLeadOutput,
    "modeler": ModelerOutput,
    "experiment": ExperimentProposal,
    "outcomes_analyst": OutcomesAnalystOutput,
    "validator": ValidatorOutput,
    "risk_analyst": RiskAnalystOutput,
    "writer": WriterOutput,
}

# Which stage each agent serves (the experiment proposer is the modeler's guided-search role).
AGENT_STAGE = {"data_analyst": "explore", "design_lead": "ideate", "modeler": "model",
               "experiment": "model", "outcomes_analyst": "backtest", "validator": "validate",
               "risk_analyst": "comply", "writer": "document"}
STAGE_AGENT = {"explore": "data_analyst", "ideate": "design_lead", "model": "modeler",
               "backtest": "outcomes_analyst", "validate": "validator", "comply": "risk_analyst",
               "document": "writer"}
FRIENDLY = {"data_analyst": "Data Analyst", "design_lead": "Design Lead", "modeler": "Modeler",
            "experiment": "Modeler (guided search)", "outcomes_analyst": "Outcomes Analyst",
            "validator": "Independent Validator", "risk_analyst": "Model-Risk Analyst",
            "writer": "Technical Writer"}
