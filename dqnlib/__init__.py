"""DQN homework library: networks, replay, agent, environments, training, experiments, plots."""
from dqnlib.utils import (set_seed, get_device, epsilon_at, moving_average, atomic_savez,
                          atomic_save_json, atomic_save_torch, format_seconds)
from dqnlib.networks import (QNetwork, NatureCNN, NatureEncoder, DuelingQNetwork,
                             DuelingNatureCNN, make_q_network, CNN_FEATURES)
from dqnlib.replay import ReplayMemory, Transition
from dqnlib.agent import DQNAgent
from dqnlib.envs import ENV_SPECS, make_env, make_eval_env, spec_for
from dqnlib.train import TrainResult, train, evaluate
from dqnlib.experiments import (VARIANT_SWITCHES, EX1_VARIANTS, EX2_VARIANTS, SEEDS, VARIANT_LABELS,
                                variant_label, agent_kwargs, build_agent, run_one, load_run,
                                load_matrix, run_matrix, run_paths)
from dqnlib.plotting import (curve_stats, plot_curves, plot_q_curves, strip_plot, summary_table,
                             compare)

__all__ = [
    "set_seed", "get_device", "epsilon_at", "moving_average", "atomic_savez", "atomic_save_json",
    "atomic_save_torch", "format_seconds", "QNetwork", "NatureCNN", "NatureEncoder",
    "DuelingQNetwork", "DuelingNatureCNN", "make_q_network", "CNN_FEATURES", "ReplayMemory",
    "Transition", "DQNAgent", "ENV_SPECS", "make_env", "make_eval_env", "spec_for", "TrainResult",
    "train", "evaluate", "VARIANT_SWITCHES", "EX1_VARIANTS", "EX2_VARIANTS", "SEEDS", "VARIANT_LABELS",
    "variant_label", "agent_kwargs", "build_agent", "run_one", "load_run", "load_matrix",
    "run_matrix", "run_paths", "curve_stats", "plot_curves", "plot_q_curves", "strip_plot",
    "summary_table", "compare",
]
