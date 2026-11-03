"""Output contracts: the typed recommendation each agent must return.

Contracts use plain JSON types (str / float / bool / lists / nested objects, no free-form maps) so
they convert cleanly to the JSON Schema handed to every backend — `claude -p --json-schema`,
Anthropic structured outputs, and the OpenAI-compatible `submit_answer` function. ``extra="forbid"``
makes an agent that invents fields fail validation (and retry with the error).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .ids import LEGACY

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


# --- intake: Intake agent ----------------------------------------------------------------
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


class IntakeAgentOutput(AgentOutput):
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


# --- explore: Data agent ---------------------------------------------------------------
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


class DataAgentOutput(AgentOutput):
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


class DataRequestsOutput(Contract):
    target_column: str = Field(default="", description="Only when context.target is empty: the "
                                                       "column you take as the dependent variable")
    target_rationale: str = ""
    requests: list[AnalysisRequest] = Field(default_factory=list)
    done: bool = Field(description="true when you need no further analysis round")
    notes: str = ""


class JoinChoice(Contract):
    link: str = Field(description="A link id from context.links")
    many: Literal["none", "aggregate", "first"] = Field(
        description="When a key can match several rows of the table being added: aggregate "
                    "(numbers averaged, text takes the first value, a row count is added) or "
                    "first (keep the first row). none when the link's added side is unique")
    unmatched: Literal["keep", "drop"] = Field(
        description="A base row with no match in the added table: keep it with missing values "
                    "(most model families then fail on it) or drop it (the engine reports how "
                    "many)")
    unmatched_reason: str = Field(
        description="Why keep or drop: how many rows have no match and whether their outcome "
                    "differs from the matched rows (the link's a_outcome / b_outcome)")
    reason: str = Field(description="Why these two tables connect through this key, in "
                                    "business terms")


class LeftOutTable(Contract):
    table: str
    reason: str


class DataJoinsOutput(Contract):
    """The Data agent's join plan when a run has several sources."""

    base_table: str = Field(description="The table whose rows are the modelling observations")
    base_rationale: str
    joins: list[JoinChoice] = Field(default_factory=list)
    left_out: list[LeftOutTable] = Field(
        default_factory=list, description="Every source you do not join, and why")
    concerns: list[str] = Field(
        default_factory=list, description="What the developer should check before accepting: "
                                          "low match rates, aggregation that hides timing, "
                                          "columns recorded after the outcome")


class ToolRequest(Contract):
    purpose: str = Field(description="The question this run answers, in one sentence")
    tool: str = Field(description="A tool name from context.tools")
    params: list[ToolParam] = Field(default_factory=list)


class ToolRequestOutput(Contract):
    """Any stage agent's request step: which of the tools offered to its stage to run."""

    requests: list[ToolRequest] = Field(default_factory=list)
    done: bool = Field(description="true when you need no further round of tool runs")
    notes: str = ""


# --- ideate: Design agent --------------------------------------------------------------------
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


class PlanChoice(Contract):
    """The development plan: every value is one of the options in context.plan.options."""

    test: Literal["out_of_time", "random", "none"] = Field(
        description="How the test sample is held back; the candidate pool is compared on it")
    validation: Literal["none", "kfold", "stratified_kfold", "group_kfold", "rolling_origin",
                        "sliding_window"] = Field(
        description="How the development sample is re-sampled to score a model on rows it was "
                    "not fitted on")
    sealed_holdout: bool = Field(
        description="Keep a final sample nobody chooses on (scored once, for the champion)")
    methods: list[Literal["grid", "forward", "backward", "stepwise", "best_subset"]] = Field(
        description="How single factors become multi-factor models; one or several")
    criterion: Literal["validation", "in_sample", "aic", "bic", "adj_r2"] = Field(
        description="What a selection step has to improve")
    rationale: str = Field(description="Why this plan fits the data and the intended use")


class DesignAgentOutput(AgentOutput):
    frameworks: list[FrameworkChoice]
    slate: list[SlateItem]
    plan: PlanChoice | None = Field(
        default=None, description="Your development plan; omit to accept context.plan as it is")
    transforms: list[TransformProposal] = Field(default_factory=list)
    sponsor_questions: list[SponsorQuestion] = Field(default_factory=list)


# --- model: Modeling agent ---------------------------------------------------------------------------
class SignCheck(Contract):
    feature: str
    expected: Literal["+", "-", "none"]
    observed: Literal["+", "-", "0", "n/a"]
    assessment: Literal["consistent", "wrong_sign", "no_prior", "not_applicable"]


class ModelingAgentOutput(AgentOutput):
    pool: list[str] = Field(description="Ids from context.leaders: the candidates that go on "
                                        "to outcomes analysis, at most context.pool_size")
    rationale: str = Field(description="Why these, and what makes them different from each "
                                       "other")
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


# --- backtest: Backtest agent -------------------------------------------------------------
class AgentFinding(Contract):
    id: str
    severity: Severity3
    category: str
    message: str
    evidence: list[str] = Field(description="Ids from context.facts")


class BacktestAgentOutput(AgentOutput):
    champion: str = Field(description="The id of one candidate in context.candidates: the "
                                      "model you recommend on the outcomes evidence")
    champion_rationale: str = Field(
        description="Why this one and not the others, on what context.candidates shows")
    discrimination: str
    calibration: str
    stability: str
    findings: list[AgentFinding] = Field(default_factory=list)


# --- validate: Validation agent --------------------------------------------------------
class ValidatorFinding(AgentFinding):
    target_stage: Literal["explore", "ideate", "model", "none"] = Field(
        description="The stage whose work must change to resolve this finding")
    remedy: str


class ValidationAgentOutput(AgentOutput):
    assessment: str
    findings: list[ValidatorFinding] = Field(default_factory=list)
    recommendation: Literal["approve", "approve_with_conditions", "send_back"]
    conditions: list[str] = Field(default_factory=list)


# --- comply: Compliance agent -------------------------------------------------------------
class ComplianceAgentOutput(AgentOutput):
    readiness: Literal["ready", "ready_with_actions", "not_ready"]
    narrative: str
    priority_actions: list[Claim] = Field(default_factory=list)


# --- document: Writing agent -------------------------------------------------------------
SECTIONS = ["executive_summary", "methodology_rationale", "alternatives_considered",
            "limitations", "use_and_monitoring"]


class Section(Contract):
    section: Literal["executive_summary", "methodology_rationale", "alternatives_considered",
                     "limitations", "use_and_monitoring"]
    markdown: str = Field(description="Prose; every number must be a {{fact:<id>}} placeholder")


class WritingAgentOutput(AgentOutput):
    sections: list[Section]


# --- any stage: a question from the page (ADR-0021) ----------------------------------------
class AskAnswer(Contract):
    answer: str = Field(description="The answer in plain sentences, for the person reading the "
                                    "page; every number is a {{fact:<id>}} placeholder")
    cites: list[str] = Field(default_factory=list,
                             description="Ids from context.facts the answer relies on")
    suggested_challenge: str = Field(
        default="", description="When the question shows something this stage's work should "
                                "reconsider: what to ask the agent to reconsider, in one or two "
                                "sentences a person could send back at the gate. Empty otherwise")


CONTRACTS: dict[str, type[Contract]] = {
    "intake_agent": IntakeAgentOutput,
    "data_agent": DataAgentOutput,
    "data_agent_requests": DataRequestsOutput,
    "data_agent_joins": DataJoinsOutput,
    "design_agent": DesignAgentOutput,
    "modeling_agent": ModelingAgentOutput,
    "modeling_agent_search": ExperimentProposal,
    "backtest_agent": BacktestAgentOutput,
    "validation_agent": ValidationAgentOutput,
    "compliance_agent": ComplianceAgentOutput,
    "writing_agent": WritingAgentOutput,
}

# Which stage each agent serves (modeling_agent_search is the Modeling agent's guided-search step).
AGENT_STAGE = {"intake_agent": "intake", "data_agent": "explore", "data_agent_requests": "explore", "data_agent_joins": "explore", "design_agent": "ideate", "modeling_agent": "model",
               "modeling_agent_search": "model", "backtest_agent": "backtest", "validation_agent": "validate",
               "compliance_agent": "comply", "writing_agent": "document"}
STAGE_AGENT = {"intake": "intake_agent", "explore": "data_agent", "ideate": "design_agent", "model": "modeling_agent",
               "backtest": "backtest_agent", "validate": "validation_agent", "comply": "compliance_agent",
               "document": "writing_agent"}
# Agents that get a tool-request step (``<agent>_tools``) before they recommend. The Data
# Analyst has its own (``data_agent_requests``), which may also write code.
TOOL_USERS = ("intake_agent", "design_agent", "modeling_agent", "backtest_agent", "validation_agent",
              "compliance_agent", "writing_agent")


def tool_step_of(agent: str) -> str | None:
    """The role agent a tool-request step belongs to, or None."""
    base = agent[:-len("_tools")] if agent.endswith("_tools") else ""
    return base if base in TOOL_USERS else None


FRIENDLY = {"intake_agent": "Intake agent", "data_agent": "Data agent",
            "data_agent_requests": "Data agent (analysis requests)",
            "data_agent_joins": "Data agent (joining sources)", "design_agent": "Design agent", "modeling_agent": "Modeling agent",
            "modeling_agent_search": "Modeling agent (guided search)", "backtest_agent": "Backtest agent",
            "validation_agent": "Validation agent", "compliance_agent": "Compliance agent",
            "writing_agent": "Writing agent"}

for _agent in TOOL_USERS:
    CONTRACTS[f"{_agent}_tools"] = ToolRequestOutput
    AGENT_STAGE[f"{_agent}_tools"] = AGENT_STAGE[_agent]
    FRIENDLY[f"{_agent}_tools"] = f"{FRIENDLY[_agent]} (tool requests)"

# Every stage agent answers questions about its page (``<agent>_ask``, ADR-0021): with its own
# scope and prompt, never a decision.
ASKABLE = tuple(STAGE_AGENT.values())
for _agent in ASKABLE:
    CONTRACTS[f"{_agent}_ask"] = AskAnswer
    AGENT_STAGE[f"{_agent}_ask"] = AGENT_STAGE[_agent]
    FRIENDLY[f"{_agent}_ask"] = f"{FRIENDLY[_agent]} (questions)"


# Pages and files written before the ids matched the names still read: an old id shows the
# agent's current name (agents/ids.py).
for _old, _new in LEGACY.items():
    for _suffix in ("", "_tools", "_ask"):
        if _new + _suffix in FRIENDLY:
            FRIENDLY.setdefault(_old + _suffix, FRIENDLY[_new + _suffix])


def ask_step_of(agent: str) -> str | None:
    """The role agent a question step belongs to, or None."""
    base = agent[:-len("_ask")] if agent.endswith("_ask") else ""
    return base if base in ASKABLE else None


def role_of(agent: str) -> str:
    """The role agent behind a step: its own name, or the agent a tool or question step serves."""
    return tool_step_of(agent) or ask_step_of(agent) or agent
