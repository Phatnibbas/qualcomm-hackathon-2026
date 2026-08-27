"""Model-family registry for the later Colab worker.

Nothing here selects anything. The registry declares which families are in
scope, which of them are importable in the current environment, how each maps
onto a portable export type, and where the search space lives (the config file,
not a Python branch).

Hard rule for this phase: no test or evaluation outcome may influence which
family, parameterization, feature set or threshold is chosen. That decision
belongs to the Colab phase, executed against the fold protocol in
``splits.py``.
"""

from __future__ import annotations

import importlib.util
from typing import Any, Callable

from . import load_config

__all__ = [
    "FAMILY_TO_EXPORT_TYPE",
    "TrialBudgetError",
    "XGBoostPolicyError",
    "available_families",
    "family_status",
    "build_estimator",
    "curated_trials",
    "trial_budget_report",
    "xgboost_policy_status",
    "assert_xgboost_eligible",
    "parameterization_contract",
    "registry_report",
]

FAMILY_TO_EXPORT_TYPE: dict[str, str] = {
    "ridge": "linear",
    "elastic_net": "linear",
    "gradient_boosting": "tree_ensemble",
    "extra_trees": "tree_ensemble",
    "xgboost_optional": "tree_ensemble",
    "mlp_relu": "mlp_relu",
}


def _sklearn_builders() -> dict[str, Callable[..., Any]]:
    from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
    from sklearn.linear_model import ElasticNet, Ridge
    from sklearn.neural_network import MLPRegressor

    return {
        "ridge": Ridge,
        "elastic_net": ElasticNet,
        "gradient_boosting": GradientBoostingRegressor,
        "extra_trees": ExtraTreesRegressor,
        "mlp_relu": MLPRegressor,
    }


def _module_present(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def family_status(config: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Per-family availability in *this* environment.

    ``xgboost_optional`` reports ``absent`` when XGBoost is not installed. It is
    never installed by this package.
    """
    resolved = config or load_config()
    declared: list[str] = list(resolved["models"]["learned_families"])
    sklearn_present = _module_present("sklearn")
    xgboost_present = _module_present("xgboost")

    status: dict[str, dict[str, Any]] = {}
    for family in declared:
        if family == "xgboost_optional":
            available = xgboost_present
            provider = "xgboost"
        else:
            available = sklearn_present
            provider = "scikit-learn"
        status[family] = {
            "provider": provider,
            "local_status": "available" if available else "absent",
            "export_type": FAMILY_TO_EXPORT_TYPE[family],
            "trainable_here": False,
            "trainable_here_reason": (
                "Local phase performs preparation and parity checks only. "
                "Training and selection happen in the Colab phase."
            ),
        }
    return status


def available_families(config: dict[str, Any] | None = None) -> list[str]:
    return [
        name
        for name, entry in family_status(config).items()
        if entry["local_status"] == "available"
    ]


def build_estimator(
    family: str,
    params: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> Any:
    """Instantiate a declared family with the configured seed.

    Raises for a family that is not declared, is forbidden this phase, or whose
    provider is not importable. This function never installs anything.
    """
    resolved = config or load_config()
    models = resolved["models"]
    if family in models["forbidden_families_this_phase"]:
        raise ValueError(f"family {family!r} is forbidden in this phase")
    if family not in models["learned_families"]:
        raise ValueError(f"family {family!r} is not in the declared registry")

    seed = int(resolved["seed"])
    kwargs = dict(params or {})

    if family == "xgboost_optional":
        if not _module_present("xgboost"):
            raise ModuleNotFoundError(
                "xgboost is not installed in this environment and must not be "
                "installed here. Train XGBoost only in the Colab phase."
            )
        import xgboost  # noqa: PLC0415 - optional dependency, imported on demand

        kwargs.setdefault("random_state", seed)
        return xgboost.XGBRegressor(**kwargs)

    builders = _sklearn_builders()
    builder = builders[family]
    if family in {"ridge", "elastic_net", "gradient_boosting", "extra_trees", "mlp_relu"}:
        kwargs.setdefault("random_state", seed)
    if family == "mlp_relu":
        kwargs.setdefault("activation", "relu")
        kwargs.setdefault("solver", "adam")
    if family == "extra_trees":
        kwargs.setdefault("n_jobs", 1)
    return builder(**kwargs)


class TrialBudgetError(ValueError):
    """Raised when the curated trial list violates its declared budget."""


class XGBoostPolicyError(RuntimeError):
    """Raised when an XGBoost candidate is used without clearing its gate."""


def curated_trials(
    config: dict[str, Any] | None = None,
    *,
    enable_xgboost: bool = False,
) -> list[dict[str, Any]]:
    """The explicit trial list, in file order, with the budget enforced.

    There is no runtime Cartesian expansion. Every configuration that will ever
    be fitted — including its direct/residual parameterization — is one entry in
    ``models.trials``, so the budget is reviewable in the file rather than being
    an emergent property of multiplying value lists together.

    XGBoost trials are excluded unless ``enable_xgboost`` is explicitly true.
    """
    resolved = config or load_config()
    models = resolved["models"]
    budget = models["trial_budget"]
    declared: list[dict[str, Any]] = list(models["trials"])

    ids = [str(trial["trial_id"]) for trial in declared]
    if len(set(ids)) != len(ids):
        duplicates = sorted({name for name in ids if ids.count(name) > 1})
        raise TrialBudgetError(f"duplicate trial_id(s) in the curated list: {duplicates}")

    known_families = set(models["learned_families"])
    known_parameterizations = set(models["parameterizations"])
    for trial in declared:
        if trial["family"] not in known_families:
            raise TrialBudgetError(
                f"trial {trial['trial_id']!r} names family {trial['family']!r}, "
                f"which is not in the declared registry"
            )
        if trial["family"] in models["forbidden_families_this_phase"]:
            raise TrialBudgetError(
                f"trial {trial['trial_id']!r} names family {trial['family']!r}, "
                f"which is forbidden this phase"
            )
        if trial["parameterization"] not in known_parameterizations:
            raise TrialBudgetError(
                f"trial {trial['trial_id']!r} names unknown parameterization "
                f"{trial['parameterization']!r}"
            )

    non_xgboost = [t for t in declared if t["family"] != "xgboost_optional"]
    maximum_default = int(budget["max_default_trials_excluding_xgboost"])
    if len(non_xgboost) > maximum_default:
        raise TrialBudgetError(
            f"{len(non_xgboost)} non-XGBoost trials exceeds "
            f"max_default_trials_excluding_xgboost={maximum_default}"
        )

    selected = list(declared) if enable_xgboost else non_xgboost
    low, high = int(budget["min_total_trials"]), int(budget["max_total_trials"])
    if not low <= len(selected) <= high:
        raise TrialBudgetError(
            f"{len(selected)} trials (enable_xgboost={enable_xgboost}) is outside "
            f"the declared budget [{low}, {high}]"
        )
    return selected


def trial_budget_report(
    config: dict[str, Any] | None = None,
    *,
    enable_xgboost: bool = False,
) -> dict[str, Any]:
    """Serialisable description of what will be fitted, and what will not."""
    resolved = config or load_config()
    budget = resolved["models"]["trial_budget"]
    every = list(resolved["models"]["trials"])
    selected = curated_trials(resolved, enable_xgboost=enable_xgboost)
    selected_ids = {t["trial_id"] for t in selected}
    return {
        "policy": budget["policy"],
        "why_not_a_grid": budget["why_not_a_grid"],
        "ordering": budget["ordering"],
        "adjustment_rule": budget["adjustment_rule"],
        "min_total_trials": budget["min_total_trials"],
        "max_total_trials": budget["max_total_trials"],
        "declared_total": len(every),
        "declared_excluding_xgboost": len([t for t in every if t["family"] != "xgboost_optional"]),
        "declared_xgboost": len([t for t in every if t["family"] == "xgboost_optional"]),
        "enable_xgboost": enable_xgboost,
        "selected_total": len(selected),
        "selected_trial_ids": [t["trial_id"] for t in selected],
        "excluded_trial_ids": [t["trial_id"] for t in every if t["trial_id"] not in selected_ids],
        "per_family_selected": {
            family: len([t for t in selected if t["family"] == family])
            for family in resolved["models"]["learned_families"]
        },
        "no_hidden_trials": (
            "The fitted set is exactly selected_trial_ids. Nothing expands it at "
            "runtime and nothing may be appended after a metric is seen."
        ),
    }


def xgboost_policy_status(
    config: dict[str, Any] | None = None,
    *,
    enable_xgboost: bool = False,
) -> dict[str, Any]:
    """Whether XGBoost is even a candidate, and why.

    The default is ``skipped_by_policy``, which is not the same as "absent". An
    importable XGBoost is still skipped unless the operator opted in, and even
    then it is not eligible until the version check and the dedicated parity
    probe pass.
    """
    resolved = config or load_config()
    policy = resolved["models"]["xgboost_policy"]
    present = _module_present("xgboost")

    if not enable_xgboost:
        return {
            "status": policy["default_status"],
            "enable_flag": policy["enable_flag"],
            "module_importable": present,
            "eligible": False,
            "reason": (
                "XGBoost is opt-in. It is excluded from the trial list and from "
                "every export path until the operator passes the enable flag."
            ),
            "boundary": policy["boundary"],
        }

    installed_version: str | None = None
    if present:
        try:
            import xgboost  # noqa: PLC0415 - optional dependency, imported on demand

            installed_version = str(xgboost.__version__)
        except Exception as exc:  # noqa: BLE001 - a broken install is a status, not a crash
            return {
                "status": "enabled_but_unusable",
                "module_importable": True,
                "eligible": False,
                "reason": f"xgboost is importable but failed to load: {exc}",
                "boundary": policy["boundary"],
            }

    expected = str(policy["expected_version"])
    return {
        "status": "enabled" if present else "enabled_but_absent",
        "enable_flag": policy["enable_flag"],
        "module_importable": present,
        "installed_version": installed_version,
        "expected_version": expected,
        "version_matches": installed_version == expected,
        "eligible": False,
        "reason": (
            "Opt-in recorded. Eligibility still requires the version match and "
            "the dedicated neutral-export parity probe; neither is decided here."
        ),
        "eligibility_gate": list(policy["eligibility_gate"]),
        "boundary": policy["boundary"],
    }


def assert_xgboost_eligible(
    parity_max_abs_error: float,
    config: dict[str, Any] | None = None,
    *,
    enable_xgboost: bool = False,
) -> dict[str, Any]:
    """Fail closed unless every XGBoost gate is satisfied.

    Called by the Colab worker before an XGBoost candidate may be exported.
    Never called on this machine: XGBoost is not installed here.
    """
    resolved = config or load_config()
    status = xgboost_policy_status(resolved, enable_xgboost=enable_xgboost)
    tolerance = float(resolved["tolerances"]["xgboost_tree_ensemble_fp64_max_abs_error"])

    if not enable_xgboost:
        raise XGBoostPolicyError(
            "XGBoost is skipped by policy. Pass the enable flag to consider it."
        )
    if not status["module_importable"]:
        raise XGBoostPolicyError("XGBoost was enabled but is not importable.")
    if not status.get("version_matches"):
        raise XGBoostPolicyError(
            f"XGBoost version {status.get('installed_version')!r} does not match the "
            f"expected {status.get('expected_version')!r}. A different build can "
            f"change split semantics, so parity proven against one version does not "
            f"transfer."
        )
    if not float(parity_max_abs_error) <= tolerance:
        raise XGBoostPolicyError(
            f"XGBoost neutral-export parity error {parity_max_abs_error} exceeds "
            f"tolerance {tolerance}; the candidate is ineligible."
        )
    return {**status, "eligible": True, "parity_max_abs_error": float(parity_max_abs_error)}


def parameterization_contract(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Both prediction parameterizations, with honest provenance."""
    resolved = config or load_config()
    models = resolved["models"]
    return {
        "parameterizations": {
            "direct": {
                "learns": "AT(t+30)",
                "final_prediction": "model output",
            },
            "residual": {
                "learns": "AT(t+30) - AT(t)",
                "final_prediction": "AT(t) + predicted residual",
            },
        },
        "parameterization_provenance": dict(models["parameterization_provenance"]),
        "status": "both are retrospective hypotheses; neither is selected on this machine",
    }


def registry_report(
    config: dict[str, Any] | None = None,
    *,
    enable_xgboost: bool = False,
) -> dict[str, Any]:
    """Serialisable registry state for the run manifest and the Colab packet."""
    resolved = config or load_config()
    models = resolved["models"]
    return {
        "artifact": "model_registry",
        "baselines": list(models["baselines"]),
        "learned_families": list(models["learned_families"]),
        "forbidden_families_this_phase": list(models["forbidden_families_this_phase"]),
        "family_to_export_type": dict(FAMILY_TO_EXPORT_TYPE),
        "family_status": family_status(resolved),
        "trial_source": "prototype/halo_safeshift/config/experiment.v1.json -> models.trials",
        "trial_budget": trial_budget_report(resolved, enable_xgboost=enable_xgboost),
        "trials": curated_trials(resolved, enable_xgboost=enable_xgboost),
        "xgboost_policy": xgboost_policy_status(resolved, enable_xgboost=enable_xgboost),
        **parameterization_contract(resolved),
        "selection_authority": models["selection_authority"],
        "boundary": [
            "No family, parameterization or hyperparameter is selected here.",
            "No test or evaluation outcome informed this registry.",
        ],
    }
