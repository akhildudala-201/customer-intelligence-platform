from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "campaign_rules_reasoncodes.yaml"

MIN_PRIORITY = 1
MAX_PRIORITY = 5

RuleKey = tuple[str, str, str]  # (risk_tier, segment, value_tier)


@dataclass(frozen=True)
class CampaignRule:
    campaign_name: str
    campaign_priority: int
    reason_code: str


@dataclass(frozen=True)
class CampaignRulesConfig:
    risk_tiers: tuple[str, ...]
    segments: tuple[str, ...]
    value_tiers: tuple[str, ...]
    reason_codes: dict[str, str]
    default_rule: CampaignRule
    rules: dict[RuleKey, CampaignRule]

    def lookup(self, risk_tier: str, segment: str, value_tier: str) -> tuple[CampaignRule, bool]:
       
        key = (risk_tier, segment, value_tier)
        if key in self.rules:
            return self.rules[key], True
        return self.default_rule, False


@functools.lru_cache(maxsize=1)
def load_campaign_rules_config(config_path: Path = CONFIG_PATH) -> CampaignRulesConfig:

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    risk_tiers = tuple(raw.get("risk_tiers") or [])
    segments = tuple(raw.get("segments") or [])
    value_tiers = tuple(raw.get("value_tiers") or [])
    reason_codes = dict(raw.get("reason_codes") or {})

    _require_non_empty(risk_tiers, "risk_tiers", config_path)
    _require_non_empty(segments, "segments", config_path)
    _require_non_empty(value_tiers, "value_tiers", config_path)
    _require_non_empty(reason_codes, "reason_codes", config_path)

    default_rule = _parse_rule(
        raw.get("default_rule"), "default_rule", reason_codes, config_path
    )

    rules: dict[RuleKey, CampaignRule] = {}
    for i, entry in enumerate(raw.get("rules") or []):
        label = f"rules[{i}]"
        risk_tier = _require_str(entry, "risk_tier", label, config_path)
        segment = _require_str(entry, "segment", label, config_path)
        value_tier = _require_str(entry, "value_tier", label, config_path)

        _require_member(risk_tier, risk_tiers, "risk_tier", label, config_path)
        _require_member(segment, segments, "segment", label, config_path)
        _require_member(value_tier, value_tiers, "value_tier", label, config_path)

        key = (risk_tier, segment, value_tier)
        if key in rules:
            raise ValueError(
                f"{config_path}: {label} duplicates an earlier rule for "
                f"(risk_tier={risk_tier!r}, segment={segment!r}, "
                f"value_tier={value_tier!r}) — each combination may only "
                "be listed once."
            )

        rules[key] = _parse_rule(entry, label, reason_codes, config_path)

    return CampaignRulesConfig(
        risk_tiers=risk_tiers,
        segments=segments,
        value_tiers=value_tiers,
        reason_codes=reason_codes,
        default_rule=default_rule,
        rules=rules,
    )


def _require_non_empty(value, field_name: str, config_path: Path) -> None:
    if not value:
        raise ValueError(f"{config_path}: `{field_name}` is missing or empty.")


def _require_str(entry: dict, field_name: str, label: str, config_path: Path) -> str:
    value = entry.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{config_path}: {label} is missing a non-empty string `{field_name}`."
        )
    return value


def _require_member(value: str, allowed: tuple[str, ...], field_name: str, label: str, config_path: Path) -> None:
    if value not in allowed:
        raise ValueError(
            f"{config_path}: {label} has {field_name}={value!r}, which is not "
            f"one of the declared `{field_name}s` {list(allowed)}. Fix the typo, "
            f"or add it to `{field_name}s:` at the top of the file if it's new."
        )


def _parse_rule(entry: dict | None, label: str, reason_codes: dict[str, str], config_path: Path) -> CampaignRule:
    if not entry:
        raise ValueError(f"{config_path}: {label} is missing.")

    campaign_name = _require_str(entry, "campaign_name", label, config_path)
    reason_code = _require_str(entry, "reason_code", label, config_path)
    _require_member(reason_code, tuple(reason_codes), "reason_code", label, config_path)

    priority = entry.get("campaign_priority")
    if not isinstance(priority, int) or isinstance(priority, bool):
        raise ValueError(f"{config_path}: {label} is missing an integer `campaign_priority`.")
    if not (MIN_PRIORITY <= priority <= MAX_PRIORITY):
        raise ValueError(
            f"{config_path}: {label} has campaign_priority={priority}, expected "
            f"{MIN_PRIORITY}-{MAX_PRIORITY} (1 = highest priority, {MAX_PRIORITY} = lowest)."
        )

    return CampaignRule(
        campaign_name=campaign_name.strip(),
        campaign_priority=priority,
        reason_code=reason_code,
    )
