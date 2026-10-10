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


# --- intake: intake analyst ----------------------------------------------------------------
BriefFieldId = Literal[
    "objective", "use_case", "segment", "default_definition", "horizon", "intended_use",
    "out_of_scope_use", "interpretability", "success_criteria", "constraints", "data_sources",
    "stakeholders", "prior_model", "update_reason", "requested_changes", "must_not_change",
    "known_issues"]


class BriefEntry(Contract):
    field: BriefFieldId = Field(description="A field id from context.fields")
    value: str = Field(description="The sponsor's position in one to three sentences, in their "
                                   "own words where possible; empty when basis is 'missing'")
    basis: Literal["stated", "inferred", "missing"] = Field(
        description="stated: the documents say it; inferred: you read it between the lines; "
                    "missing: the documents do not address it")
    quote: str = Field(default="", description="For 'stated': a short passage copied verbatim "
                                               "from the documents that says it")


class InterviewQuestion(Contract):
    question: str = Field(description="One question the sponsor can answer in a sentence or two")
    field: Literal[
        "objective", "use_case", "segment", "default_definition", "horizon", "intended_use",
        "out_of_scope_use", "interpretability", "success_criteria", "constraints", "data_sources",
        "stakeholders", "prior_model", "update_reason", "requested_changes", "must_not_change",
        "known_issues", "none"] = Field(
        default="none", description="The brief field the answer fills, or 'none'")
    why_it_matters: str
    blocking: bool = Field(description="true when development should not start without the answer")


class ChangeItem(Contract):
    change: str = Field(description="One requested change, restated plainly")
    type: Literal["data_refresh", "recalibration", "re_estimation", "redevelopment",
                  "scope_change", "remediation", "documentation"]
    affects: Literal["explore", "ideate", "model", "none"] = Field(
        description="The earliest stage whose work this change alters")


class IntakeAnalystOutput(AgentOutput):
    restated_objective: str = Field(description="The business goal in your own words, in one or "
                                                "two sentences the sponsor would agree with")
    clarity: Literal["clear", "needs_clarification", "unclear"]
    brief: list[BriefEntry] = Field(description="One entry per field in context.fields")
    interview: list[InterviewQuestion] = Field(default_factory=list)
    update_scope: Literal["not_applicable", "recalibrate", "re_estimate", "redevelop"] = Field(
        default="not_applicable", description="Model update only: how deep the change goes")
    scope_rationale: str = ""
    change_items: list[ChangeItem] = Field(default_factory=list)
    incumbent_family: str = Field(
        default="", description="Model update only: the family in context.engine_families the "
                                "existing model uses, or empty when the artifacts do not say")


# --- explore: data analyst ---------------------------------------------------------------
class ColumnDecision(Contract):
    column: str
    decision: Literal["keep", "exclude"]
    reason: str
    evidence: list[str] = Field(default_factory=list)


class FeatureCandidate(Contract):
    column: str = Field(description="A column in context.features")
    relationship: Literal["+", "-", "nonlinear", "unknown"] = Field(
        description="How the target is expected to move as this feature rises")
    rationale: str = Field(description="Why the business intent makes this a driver worth testing")
    evidence: list[str] = Field(default_factory=list)


class DataAnalystOutput(AgentOutput):
    target: str = Field(default="", description="The dependent variable: context.target when it "
                                                "is fixed, else the column you recommend")
    target_rationale: str = Field(default="", description="Why this column is the outcome the "
                                                          "business intent describes")
    feature_candidates: list[FeatureCandidate] = Field(
        default_factory=list, description="The features worth considering, strongest first")
    column_decisions: list[ColumnDecision] = Field(
        description="One decision per leakage suspect and per column you recommend excluding")
    data_quality: list[Claim] = Field(default_factory=list)
    questions_for_sponsor: list[str] = Field(default_factory=list)


class ToolParam(Contract):
    name: str
    value: str


class AnalysisRequest(Contract):
    purpose: str = Field(description="The question this analysis answers, in one sentence")
    tool: str = Field(default="", description="A tool name from context.tools; empty when you "
                                              "write code instead")
    params: list[ToolParam] = Field(default_factory=list)
    code: str = Field(default="", description="Python, only when no tool fits (see the prompt "
                                              "for what a script may use)")


class DataScoutOutput(Contract):
    target_column: str = Field(default="", description="Only when context.target is empty: the "
                                                       "column you take as the dependent variable")
    target_rationale: str = ""
    requests: list[AnalysisRequest] = Field(default_factory=list)
    done: bool = Field(description="true when you need no further analysis round")
    notes: str = ""


class ToolRequest(Contract):
    purpose: str = Field(description="The question this run answers, in one sentence")
    tool: str = Field(description="A tool name from context.tools")
    params: list[ToolParam] = Field(default_factory=list)


class ToolRequestOutput(Contract):
    """Any stage agent's request step: which of the tools offered to its stage to run."""

    requests: list[ToolRequest] = Field(default_factory=list)
    done: bool = Field(description="true when you need no further round of tool runs")
    notes: str = ""


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
    "intake_analyst": IntakeAnalystOutput,
    "data_analyst": DataAnalystOutput,
    "data_scout": DataScoutOutput,
    "design_lead": DesignLeadOutput,
    "modeler": ModelerOutput,
    "experiment": ExperimentProposal,
    "outcomes_analyst": OutcomesAnalystOutput,
    "validator": ValidatorOutput,
    "risk_analyst": RiskAnalystOutput,
    "writer": WriterOutput,
}

# Which stage each agent serves (the experiment proposer is the modeler's guided-search role).
AGENT_STAGE = {"intake_analyst": "intake", "data_analyst": "explore", "data_scout": "explore", "design_lead": "ideate", "modeler": "model",
               "experiment": "model", "outcomes_analyst": "backtest", "validator": "validate",
               "risk_analyst": "comply", "writer": "document"}
STAGE_AGENT = {"intake": "intake_analyst", "explore": "data_analyst", "ideate": "design_lead", "model": "modeler",
               "backtest": "outcomes_analyst", "validate": "validator", "comply": "risk_analyst",
               "document": "writer"}
# Agents that get a tool-request step (``<agent>_tools``) before they recommend. The Data
# Analyst has its own (``data_scout``), which may also write code.
TOOL_USERS = ("intake_analyst", "design_lead", "modeler", "outcomes_analyst", "validator",
              "risk_analyst", "writer")


def tool_step_of(agent: str) -> str | None:
    """The role agent a tool-request step belongs to, or None."""
    base = agent[:-len("_tools")] if agent.endswith("_tools") else ""
    return base if base in TOOL_USERS else None


FRIENDLY = {"intake_analyst": "Intake Analyst", "data_analyst": "Data Analyst",
            "data_scout": "Data Analyst (analysis requests)", "design_lead": "Design Lead", "modeler": "Modeler",
            "experiment": "Modeler (guided search)", "outcomes_analyst": "Outcomes Analyst",
            "validator": "Independent Validator", "risk_analyst": "Model-Risk Analyst",
            "writer": "Technical Writer"}

for _agent in TOOL_USERS:
    CONTRACTS[f"{_agent}_tools"] = ToolRequestOutput
    AGENT_STAGE[f"{_agent}_tools"] = AGENT_STAGE[_agent]
    FRIENDLY[f"{_agent}_tools"] = f"{FRIENDLY[_agent]} (tool requests)"
