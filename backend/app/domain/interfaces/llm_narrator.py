from abc import ABC, abstractmethod


class LLMNarrator(ABC):
    """Port for the read-only, natural-language explanation layer over the
    gate (reconstruction 2026-09, Fase 7 - docs/quant_methodology.md §25):
    translates an already-computed `levels_engine.GateResult` into a short,
    plain-language summary a non-technical reader can act on - never a
    second opinion, never a score, never a decision of its own.

    Kept a plain, primitive-typed port (no `GateResult`/`EntryTrigger`
    import here) for the same reason `MarketDataProvider` never imports a
    concrete provider's own types: `domain/` depends on nothing in
    `services/` or `infrastructure/` (see CLAUDE.md's layering rule) - the
    caller (a service, which already imports `levels_engine`) translates a
    `GateResult` into these primitives before calling this port, not the
    other way around.

    Structurally incapable of scoring or deciding anything: every method
    here only ever receives facts the gate/exit engine already computed
    and settled - there is no parameter through which this port could feed
    a verdict back into anything upstream, and every implementation must
    return `None` (never raise, never fabricate a plausible-sounding
    answer) when it isn't configured or the call fails for any reason - a
    narrative is optional color on top of a decision already made, never a
    dependency the decision itself can be blocked or corrupted by."""

    @abstractmethod
    def explain_gate(
        self,
        ticker: str,
        gate_passes: bool,
        conditions: list[tuple[str, bool]],
        trend_label: str,
        stage_label: str | None,
        entry_trigger_summary: str | None,
        stop_and_target_summary: str | None,
    ) -> str | None:
        """A short (2-4 sentence) plain-language summary of why the gate did
        or didn't pass and what the entry trigger/stop-target mean in
        practice, in Spanish (matching the rest of this app's user-facing
        text) - or `None` when unconfigured or on any failure. `conditions`
        is `[(label, passed), ...]`, the same shape `GateCondition` already
        carries; `entry_trigger_summary`/`stop_and_target_summary` are
        already-formatted strings (e.g. "ruptura en 125.40, ya disparado"),
        not raw numbers, so no implementation of this port ever needs to
        know this app's currency/rounding conventions."""
        ...
