import argparse

from src.config.ct_config import (
    MIN_IMPROVEMENT,
)
from src.pipeline.training_pipeline import (
    TrainingPipeline,
)
from src.utils.model_registry import (
    get_champion_metric,
    get_champion_version,
    promote_model,
)


def should_promote(
    champion_f1: float | None,
    candidate_f1: float,
) -> bool:
    if champion_f1 is None:
        return True

    improvement = (
        candidate_f1 - champion_f1
    )

    return improvement >= MIN_IMPROVEMENT


def run_ct(trigger: str) -> None:
    print("\n")
    print("========================================")
    print("      CONTINUOUS TRAINING PIPELINE")
    print("========================================")

    print(
        f"\nCT Trigger: {trigger}"
    )

    # --------------------------------------------------
    # 1. GET CURRENT CHAMPION
    # --------------------------------------------------

    champion = get_champion_version()
    champion_f1 = get_champion_metric()

    if champion is None:
        print(
            "\nCurrent champion: NONE"
        )
        print(
            "This is the first CT run."
        )
    else:
        print(
            "\nCurrent champion version:",
            champion.version,
        )

        print(
            "Current champion F1:",
            f"{champion_f1:.4f}",
        )

    # --------------------------------------------------
    # 2. TRAIN CANDIDATE
    # --------------------------------------------------

    print(
        "\nStarting candidate training..."
    )

    pipeline = TrainingPipeline()

    candidate = pipeline.run_pipeline()

    candidate_f1 = candidate["f1"]
    candidate_version = candidate["version"]

    print("\n")
    print("========================================")
    print("        CT PROMOTION GATE")
    print("========================================")

    print(
        "Candidate version:",
        candidate_version,
    )

    print(
        "Candidate F1:",
        f"{candidate_f1:.4f}",
    )

    # --------------------------------------------------
    # 3. COMPARE CANDIDATE VS CHAMPION
    # --------------------------------------------------

    if champion_f1 is not None:
        improvement = (
            candidate_f1 - champion_f1
        )

        print(
            "Champion F1:",
            f"{champion_f1:.4f}",
        )

        print(
            "Improvement:",
            f"{improvement:.4f}",
        )

        print(
            "Required improvement:",
            f"{MIN_IMPROVEMENT:.4f}",
        )

    # --------------------------------------------------
    # 4. PROMOTE OR REJECT
    # --------------------------------------------------

    if should_promote(
        champion_f1,
        candidate_f1,
    ):
        promote_model(
            candidate_version
        )

        print("\n✅ CANDIDATE PROMOTED")

        print(
            f"Version {candidate_version} "
            "is now the champion."
        )
    else:
        print("\n❌ CANDIDATE REJECTED")

        print(
            "Current champion remains "
            "in production."
        )

    print("\n")
    print("========================================")
    print("          CT RUN COMPLETE")
    print("========================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--trigger",
        choices=[
            "manual",
            "drift",
        ],
        required=True,
    )

    args = parser.parse_args()

    run_ct(
        trigger=args.trigger
    )