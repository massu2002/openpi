import collections
import dataclasses
import json
import logging
import math
import pathlib
from typing import Any, Dict, List, Tuple

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suites: Tuple[str, ...] = (
        "libero_spatial",
        "libero_object",
        "libero_goal",
        "libero_10",
        # "libero_90",
    )
    
    task_suite_name: str = "libero_spatial"
    # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90

    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Multi-seed
    #################################################################################################################
    seeds: Tuple[int, ...] = (7, 8, 9)  # 複数シード

    #################################################################################################################
    # Output root
    #################################################################################################################
    result_root: str = "result/libero"  # result/libero/{task_suite}/(json|videos)

    #################################################################################################################
    # Utils
    #################################################################################################################
    seed: int = 7  # (単一seed評価のときに使用; multi-seedでは内部で上書き)


def _max_steps_for_suite(task_suite_name: str) -> int:
    if task_suite_name == "libero_spatial":
        return 220
    if task_suite_name == "libero_object":
        return 280
    if task_suite_name == "libero_goal":
        return 300
    if task_suite_name == "libero_10":
        return 520
    if task_suite_name == "libero_90":
        return 400
    raise ValueError(f"Unknown task suite: {task_suite_name}")


def eval_libero_seed(args: Args, *, seed: int) -> Dict[str, Any]:
    """
    1つの seed で task_suite を全タスク評価し、結果dictを返す（保存は呼び出し側）。
    JSON構造：
      {
        "task_suite": "...",
        "seed": 7,
        "suite_summary": {...},
        "tasks": {
           "<task_description>": {...},
           ...
        }
      }
    """
    np.random.seed(seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name} | seed={seed}")

    max_steps = _max_steps_for_suite(args.task_suite_name)

    # output dirs
    suite_dir = pathlib.Path(args.result_root) / args.task_suite_name
    video_dir = suite_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    total_episodes, total_successes = 0, 0
    tasks_out: Dict[str, Any] = {}

    for task_id in tqdm.tqdm(range(num_tasks_in_suite), desc=f"tasks(seed={seed})"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, seed)

        task_episodes, task_successes = 0, 0
        episode_details: List[Dict[str, Any]] = []
        
        saved_success = False
        saved_failure = False

        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task), desc="episodes", leave=False):
            env.reset()
            action_plan = collections.deque()

            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            record_video = not (saved_success and saved_failure)
            replay_images = []
            done = False
            error_str = None

            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    if record_video:
                        replay_images.append(img)

                    if not action_plan:
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        action_chunk = client.infer(element)["actions"]
                        if len(action_chunk) < args.replan_steps:
                            raise RuntimeError(
                                f"Policy predicts only {len(action_chunk)} steps < replan_steps={args.replan_steps}"
                            )
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()
                    obs, reward, done, info = env.step(action.tolist())

                    if done:
                        break
                    t += 1

                except Exception as e:
                    error_str = str(e)
                    logging.error(f"[seed={seed}] task='{task_description}' episode={episode_idx} exception: {e}")
                    break

            # per-episode bookkeeping
            task_episodes += 1
            total_episodes += 1
            if done:
                task_successes += 1
                total_successes += 1

            episode_details.append(
                {
                    "episode_idx": int(episode_idx),
                    "success": bool(done),
                    "steps": int(t),
                    "error": error_str,
                }
            )

            # 最初の成功/失敗だけ動画保存
            if record_video and len(replay_images) > 0:
                want_save = False
                suffix = None

                if done and (not saved_success):
                    want_save = True
                    suffix = "success"
                elif (not done) and (not saved_failure):
                    want_save = True
                    suffix = "failure"

                if want_save:
                    safe_task = _sanitize_name(task_description)
                    out_mp4 = video_dir / (
                        f"seed{seed:04d}_task{task_id:03d}_{safe_task}_{suffix}.mp4"
                    )
                    try:
                        imageio.mimwrite(out_mp4, [np.asarray(x) for x in replay_images], fps=10)
                        if suffix == "success":
                            saved_success = True
                        else:
                            saved_failure = True
                    except Exception as e:
                        logging.error(f"Failed to write video {out_mp4}: {e}")

        # task-level summary
        task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0.0
        tasks_out[task_description] = {
            "task_id": int(task_id),
            "task_description": str(task_description),
            "num_episodes": int(task_episodes),
            "num_successes": int(task_successes),
            "success_rate": float(task_success_rate),
            "episodes": episode_details,  # seed別jsonにだけ入る（平均jsonには使わない想定）
        }

        logging.info(
            f"[seed={seed}] task done: '{task_description}' success_rate={task_success_rate:.3f}"
        )

    suite_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0
    out = {
        "task_suite": str(args.task_suite_name),
        "seed": int(seed),
        "suite_summary": {
            "num_tasks": int(num_tasks_in_suite),
            "num_trials_per_task": int(args.num_trials_per_task),
            "num_episodes_total": int(total_episodes),
            "num_successes_total": int(total_successes),
            "success_rate_total": float(suite_success_rate),
        },
        "tasks": tasks_out,
    }
    return out


def eval_libero_multi_seed(args: Args) -> None:
    """
    複数seedを回して:
      - result/libero/{suite}/json/result_{seed}.json
      - result/libero/{suite}/json/result.json (seed平均)
    を作る。
    """
    suite_dir = pathlib.Path(args.result_root) / args.task_suite_name
    json_dir = suite_dir / "json"
    video_dir = suite_dir / "videos"
    json_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    seed_results: Dict[int, Dict[str, Any]] = {}

    for seed in args.seeds:
        r = eval_libero_seed(args, seed=seed)
        seed_results[int(seed)] = r

        # write seed json
        out_path = json_dir / f"result_{int(seed)}.json"
        _write_json(out_path, r)
        logging.info(f"Wrote: {out_path}")

    # build averaged result.json
    avg = _aggregate_seed_results(args.task_suite_name, args.num_trials_per_task, seed_results)
    out_path = json_dir / "result.json"
    _write_json(out_path, avg)
    logging.info(f"Wrote: {out_path}")
    
    
def eval_libero_all_suites(args: Args) -> None:
    """
    args.task_suites に書かれた suite を全て実行して、
    suiteごとに result/libero/{suite}/(json|videos) に保存する。
    """
    suites = list(args.task_suites)

    # もし誤って空で来た場合は task_suite_name を使う
    if len(suites) == 0:
        suites = [args.task_suite_name]

    for suite in suites:
        logging.info("=" * 80)
        logging.info(f"Running suite: {suite} | seeds={args.seeds} | trials={args.num_trials_per_task}")
        logging.info("=" * 80)

        # suite を差し替えて 1 suite 実行（multi-seed 保存までやる）
        args.task_suite_name = suite
        eval_libero_multi_seed(args)


def _aggregate_seed_results(
    task_suite_name: str,
    num_trials_per_task: int,
    seed_results: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    seed_results から seed平均のJSONを作る。
    - suite_summary: total success_rate の平均/標準偏差
    - tasks: task_descriptionごとに success_rate の平均/標準偏差、成功数合計など
    """
    seeds = sorted(seed_results.keys())
    n_seeds = len(seeds)

    # suite-level
    suite_rates = [
        float(seed_results[s]["suite_summary"]["success_rate_total"]) for s in seeds
    ]
    suite_rate_mean = float(np.mean(suite_rates)) if n_seeds > 0 else 0.0
    suite_rate_std = float(np.std(suite_rates, ddof=0)) if n_seeds > 0 else 0.0

    # tasks union by description
    all_task_desc = set()
    for s in seeds:
        all_task_desc.update(seed_results[s]["tasks"].keys())

    tasks_out: Dict[str, Any] = {}
    for desc in sorted(all_task_desc):
        rates = []
        succ = []
        eps = []
        task_id_any = None
        for s in seeds:
            t = seed_results[s]["tasks"].get(desc)
            if t is None:
                continue
            task_id_any = t.get("task_id", task_id_any)
            rates.append(float(t["success_rate"]))
            succ.append(int(t["num_successes"]))
            eps.append(int(t["num_episodes"]))

        rate_mean = float(np.mean(rates)) if len(rates) > 0 else 0.0
        rate_std = float(np.std(rates, ddof=0)) if len(rates) > 0 else 0.0

        tasks_out[desc] = {
            "task_id": task_id_any,
            "task_description": desc,
            "num_seeds": int(n_seeds),
            "seeds": seeds,
            "num_trials_per_task": int(num_trials_per_task),
            "success_rate_mean": float(rate_mean),
            "success_rate_std": float(rate_std),
            "success_rate_per_seed": {str(s): float(seed_results[s]["tasks"].get(desc, {}).get("success_rate", 0.0)) for s in seeds},
            "num_successes_sum": int(np.sum(succ)) if len(succ) > 0 else 0,
            "num_episodes_sum": int(np.sum(eps)) if len(eps) > 0 else 0,
        }

    out = {
        "task_suite": str(task_suite_name),
        "seeds": seeds,
        "suite_summary": {
            "num_seeds": int(n_seeds),
            "num_trials_per_task": int(num_trials_per_task),
            "success_rate_total_mean": float(suite_rate_mean),
            "success_rate_total_std": float(suite_rate_std),
            "success_rate_total_per_seed": {str(s): float(seed_results[s]["suite_summary"]["success_rate_total"]) for s in seeds},
        },
        "tasks": tasks_out,
    }
    return out


def _write_json(path: pathlib.Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _sanitize_name(s: str, max_len: int = 120) -> str:
    # ファイル名に使えない文字を雑に除去
    bad = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for b in bad:
        s = s.replace(b, "_")
    s = s.replace(" ", "_")
    if len(s) > max_len:
        s = s[:max_len]
    return s


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite:
    https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero_all_suites)
