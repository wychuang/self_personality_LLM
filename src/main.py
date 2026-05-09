"""CLI entry point for self_personality_LLM experiments.

Usage:
    python -m src.main experiment=experiment4
    python -m src.main experiment=experiment1 config.lambda=0.1
    python -m src.main experiment=experiment2
    python -m src.main experiment=experiment3
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config, save_config


EXPERIMENT_HELP = {
    "experiment4": "Personality crystallization curve measurement (observation only)",
    "experiment1": "Identity Direction Anchoring (IDA) training",
    "experiment2": "Self-consistency RL (DPO) training",
    "experiment3": "Drive-embedded multi-objective training",
}


def main():
    # Parse CLI: python main.py experiment=X key=value key2=value2 ...
    experiment = None
    overrides = []

    for arg in sys.argv[1:]:
        if arg.startswith("experiment="):
            experiment = arg.split("=", 1)[1]
        elif arg == "--help" or arg == "-h":
            print("Available experiments:")
            for name, desc in EXPERIMENT_HELP.items():
                print(f"  {name}: {desc}")
            print("\nUsage: python -m src.main experiment=<name> [key=value ...]")
            return
        elif "=" in arg:
            overrides.append(arg)

    if experiment is None:
        print("Please specify an experiment: python -m src.main experiment=<name>")
        print("Available:", ", ".join(EXPERIMENT_HELP.keys()))
        return

    cfg = load_config(experiment=experiment, overrides=overrides)

    print(f"Running {experiment} with config:")
    print(f"  Model: {cfg.model.name}")
    print(f"  Output: {cfg.training.output_dir}")

    if experiment == "experiment4":
        _run_experiment4(cfg)
    elif experiment == "experiment1":
        _run_experiment1(cfg)
    elif experiment == "experiment2":
        _run_experiment2(cfg)
    elif experiment == "experiment3":
        _run_experiment3(cfg)
    else:
        print(f"Unknown experiment: {experiment}")


def _run_experiment4(cfg):
    from src.experiment4.probe_suite import ProbeSuite
    from src.experiment4.crystallization_plotter import plot_crystallization_curves

    suite = ProbeSuite(
        model_name=cfg.model.name,
        output_dir=cfg.output.results_dir,
        num_style_samples=cfg.probes.style.num_samples,
        style_prompt=cfg.probes.style.prompt,
        max_new_tokens=cfg.probes.style.max_new_tokens,
        temperature=cfg.probes.style.temperature,
    )

    df = suite.run(
        start_step=cfg.checkpoint.step_start,
        end_step=cfg.checkpoint.step_end,
        step_interval=cfg.checkpoint.step_interval,
    )

    plot_crystallization_curves(df, cfg.output.results_dir)
    print(f"Experiment 4 complete. Results in {cfg.output.results_dir}")


def _run_experiment1(cfg):
    from src.core.model_utils import load_model_and_tokenizer
    from src.experiment1.seed_generator import load_seed_narratives
    from src.experiment1.activation_extractor import ActivationExtractor
    from src.experiment1.identity_direction import compute_v_id, analyze_v_id
    from src.experiment1.dual_loss_trainer import ida_train

    model, tokenizer = load_model_and_tokenizer(cfg)

    narratives = load_seed_narratives()
    print(f"Loaded {len(narratives)} seed narratives")

    extractor = ActivationExtractor(model, layers=["last"])
    activations = extractor.extract_from_texts(narratives, tokenizer)
    v_id = compute_v_id(activations, method="mean")
    print(f"Computed v_id: norm={v_id.norm():.4f}")

    analysis = analyze_v_id(v_id, activations)
    print(f"v_id analysis: {analysis}")

    # Quick IDA training on seed narratives themselves as demo
    model = ida_train(
        model, tokenizer, v_id, cfg,
        train_texts=narratives * 10,  # Repeat for demo
        id_lambda=0.1,
        experiment_name="ida_demo",
    )

    print(f"Experiment 1 complete. Model saved to {cfg.training.output_dir}/ida_demo")


def _run_experiment2(cfg):
    from src.core.model_utils import load_model_and_tokenizer
    from src.experiment2.self_statement_generator import SelfStatementGenerator
    from src.experiment2.consistency_scorer import ConsistencyScorer
    from src.experiment2.preference_dataset import build_preference_dataset
    from src.experiment2.dpo_trainer import train_dpo

    model, tokenizer = load_model_and_tokenizer(cfg)

    generator = SelfStatementGenerator()
    pairs = generator.generate_pairs(model, tokenizer, num_pairs=100)
    print(f"Generated {len(pairs)} self-statement pairs")

    scorer = ConsistencyScorer(method="embedding", embed_model=model, embed_tokenizer=tokenizer)
    scores = scorer.score_all(pairs)
    print(f"Scored {len(scores)} pairs: mean={sum(scores)/len(scores):.3f}")

    ref_model, _ = load_model_and_tokenizer(cfg)
    ref_model.eval()

    dataset = build_preference_dataset(pairs, scores)
    model = train_dpo(model, ref_model, tokenizer, dataset, cfg)

    print(f"Experiment 2 complete. Model saved to {cfg.training.output_dir}/dpo_consistency")


def _run_experiment3(cfg):
    from src.core.model_utils import load_model_and_tokenizer
    from src.experiment3.multi_objective_trainer import drive_embedded_train
    from src.experiment1.seed_generator import load_seed_narratives

    model, tokenizer = load_model_and_tokenizer(cfg)
    narratives = load_seed_narratives()

    model = drive_embedded_train(
        model, tokenizer, cfg,
        train_texts=narratives * 10,
        alpha=0.01,
        beta=0.01,
        gamma=0.005,
    )

    print(f"Experiment 3 complete. Model saved to {cfg.training.output_dir}/drive_embedded")


if __name__ == "__main__":
    main()
