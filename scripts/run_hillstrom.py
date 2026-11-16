"""Run the stats core on Hillstrom and print the results.

All numbers come from functions in readout/. This script only formats them.
"""

from readout.config import GUARDRAILS, MIN_PRACTICAL_LIFT
from readout.decision import decide
from readout.guardrails import guardrail_check
from readout.loaders.hillstrom import load_hillstrom
from readout.metrics import compare_all_to_control, mean_summary
from readout.srm import srm_check
from readout.variance_reduction import cuped_compare

PRIMARY = "visit"
FOCUS_ARM = "Womens E-Mail"


def show_srm(title, result):
    verdict = "ok" if result.passed else "MISMATCH"
    print(f"{title}: chi2={result.chi2:.3f}, df={result.dof}, p={result.p_value:.4f} -> {verdict}")
    for arm, n in result.observed.items():
        print(f"    {arm:<14} observed {n:>6,}   expected {result.expected[arm]:>9,.1f}")


def show_effect(title, e):
    print(f"{title}")
    print(f"    control {e.control_estimate:.4f}   treatment {e.treatment_estimate:.4f}")
    print(
        f"    abs diff {e.abs_diff:+.4f}  SE {e.abs_se:.5f}  "
        f"{e.ci_level:.1%} CI [{e.abs_ci[0]:+.4f}, {e.abs_ci[1]:+.4f}]"
    )
    print(
        f"    rel lift {e.rel_lift:+.2%}  {e.ci_level:.1%} CI "
        f"[{e.rel_ci[0]:+.2%}, {e.rel_ci[1]:+.2%}]  z={e.z:.2f}  "
        f"p={e.p_value:.3g}  p_adj={e.p_adjusted:.3g}"
    )


def main():
    exp = load_hillstrom()
    df = exp.df
    print(f"Hillstrom: {len(df):,} users, arms {exp.arm_counts()}\n")

    print("== SRM ==")
    srm_all = srm_check(exp.arm_counts(), exp.intended_allocation)
    show_srm("All 3 arms (the gate)", srm_all)
    pair = [exp.control_arm, FOCUS_ARM]
    show_srm(
        "Womens vs No E-Mail (diagnostic)",
        srm_check(exp.arm_counts(), exp.intended_allocation, pair),
    )

    print(f"\n== {FOCUS_ARM} vs {exp.control_arm}, metric: {PRIMARY} ==")
    y, arm = df[PRIMARY].to_numpy(), df[exp.arm_col].to_numpy()
    for covariates in (["history"], ["recency"], ["history", "recency"]):
        result = cuped_compare(
            y, df[covariates].to_numpy(), arm, FOCUS_ARM, exp.control_arm, tuple(covariates)
        )
        if covariates == ["history"]:
            show_effect("Raw (no adjustment)", result.raw)
        theta = ", ".join(f"{t:.3g}" for t in result.theta)
        show_effect(f"CUPED with {' + '.join(covariates)} (theta = {theta})", result.adjusted)
        print(f"    variance reduction: {result.variance_reduction:.2%}")

    print("\n== Both e-mail arms vs control, Holm-adjusted, with decision ==")
    control = mean_summary(exp.values(PRIMARY, exp.control_arm))
    primary = compare_all_to_control(
        {a: mean_summary(exp.values(PRIMARY, a)) for a in exp.treatment_arms}, control
    )
    for treatment in exp.treatment_arms:
        show_effect(f"{treatment}: {PRIMARY}", primary[treatment])
        guardrails = []
        for metric, cfg in GUARDRAILS[exp.name].items():
            effect = compare_all_to_control(
                {treatment: mean_summary(exp.values(metric, treatment))},
                mean_summary(exp.values(metric, exp.control_arm)),
            )[treatment]
            g = guardrail_check(metric, effect, **cfg)
            guardrails.append(g)
            print(
                f"    guardrail {metric}: lift {g.estimate:+.1%}, one-sided lower bound "
                f"{g.lower_bound:+.1%} vs margin {-g.margin:+.0%} -> {g.status}"
            )
        decision = decide(srm_all, guardrails, primary[treatment], MIN_PRACTICAL_LIFT[exp.name])
        print(f"    DECISION: {decision.label}")
        for reason in decision.reasons:
            print(f"      - {reason}")


if __name__ == "__main__":
    main()
