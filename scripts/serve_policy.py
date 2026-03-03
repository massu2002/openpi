import dataclasses
import enum
import logging
import socket
import tyro
from typing import Any, Tuple

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False
    
    train_seed: int | None = None

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)
    ckpt_steps: list[int] = dataclasses.field(default_factory=lambda: [30000, 20000, 10000])

def _build_ckpt_dir_candidates(dir_template: str, train_seed: int | None, ckpt_steps: list[int]) -> list[str]:
    """
    dir_template に {seed}/{step} が含まれていれば埋めて候補を返す。
    - {step} がある場合: ckpt_steps を順に当てた複数候補
    - {step} がない場合: 単一候補
    """
    needs_seed = "{seed" in dir_template
    needs_step = "{step" in dir_template

    if needs_seed and train_seed is None:
        raise ValueError(
            "Checkpoint dir contains '{seed}' but --train_seed was not provided. "
            "Example: --train_seed 7"
        )

    if needs_step:
        out = []
        for s in ckpt_steps:
            if needs_seed:
                out.append(dir_template.format(seed=train_seed, step=s))
            else:
                out.append(dir_template.format(step=s))
        return out

    # stepなし（従来通り）
    if needs_seed:
        return [dir_template.format(seed=train_seed)]
    return [dir_template]


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="./checkpoints/teacher/pi05_libero/fineturne_libero/seed{seed}/{step}",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi0_libero_distill_6",
        dir="./checkpoints/student/pi0_libero_distill_6/libero_distill_6/seed{seed}/{step}",
    ),
}


@dataclasses.dataclass(frozen=True)
class _LoadPlan:
    config: str
    ckpt_dirs: Tuple[str, ...]
    source: str          # "default" or "checkpoint"
    env: str             # args.env.value


@dataclasses.dataclass(frozen=True)
class _LoadedSpec:
    config: str
    ckpt_dir: str
    source: str
    env: str
    tried_ckpt_dirs: Tuple[str, ...]


def _resolve_load_plan(args: Args) -> _LoadPlan:
    match args.policy:
        case Checkpoint():
            dirs = _build_ckpt_dir_candidates(args.policy.dir, args.train_seed, args.ckpt_steps)
            return _LoadPlan(
                config=args.policy.config,
                ckpt_dirs=tuple(dirs),
                source="checkpoint",
                env=args.env.value,
            )
        case Default():
            checkpoint = DEFAULT_CHECKPOINT.get(args.env)
            if checkpoint is None:
                raise ValueError(f"Unsupported environment mode: {args.env}")
            dirs = _build_ckpt_dir_candidates(checkpoint.dir, args.train_seed, args.ckpt_steps)
            return _LoadPlan(
                config=checkpoint.config,
                ckpt_dirs=tuple(dirs),
                source="default",
                env=args.env.value,
            )


def create_policy(args: Args) -> tuple[_policy.Policy, _LoadedSpec]:
    plan = _resolve_load_plan(args)

    last_err: Exception | None = None
    for ckpt_dir in plan.ckpt_dirs:
        try:
            pol = _policy_config.create_trained_policy(
                _config.get_config(plan.config),
                ckpt_dir,
                default_prompt=args.default_prompt,
            )
            spec = _LoadedSpec(
                config=plan.config,
                ckpt_dir=ckpt_dir,
                source=plan.source,
                env=plan.env,
                tried_ckpt_dirs=plan.ckpt_dirs,
            )
            return pol, spec
        except Exception as e:
            last_err = e
            logging.warning(
                "Failed to load ckpt_dir=%s (%s). Trying next if available...",
                ckpt_dir,
                type(e).__name__,
            )

    raise RuntimeError(f"Failed to load policy from any candidates: {plan.ckpt_dirs}") from last_err


def main(args: Args) -> None:
    policy, spec = create_policy(args)

    base_md = policy.metadata
    if isinstance(base_md, dict):
        policy_metadata: dict[str, Any] = dict(base_md)
    else:
        policy_metadata = {"policy_metadata": str(base_md)}

    policy_metadata.update(
        {
            "ckpt_dir": spec.ckpt_dir,
            "ckpt_config": spec.config,
            "ckpt_source": spec.source,
            "env_mode": spec.env,
            "train_seed": args.train_seed,
            "ckpt_tried_dirs": list(spec.tried_ckpt_dirs),
        }
    )

    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)
    logging.info("Serving ckpt_dir=%s (config=%s)", spec.ckpt_dir, spec.config)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
