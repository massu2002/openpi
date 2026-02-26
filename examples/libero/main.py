import collections
import dataclasses
import json
import logging
import math
import pathlib
from typing import Any, Dict, List, Optional, Tuple

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import pathlib
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro
import copy

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    # Model server parameters
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    # LIBERO environment-specific parameters
    task_suites: Tuple[str, ...] = (
        "libero_spatial",
        "libero_object",
        "libero_goal",
        "libero_10",
    )
    task_suite_name: str = "libero_spatial"  # 1つだけ指定する場合の引数（複数指定の task_suites と両方書かれたときは task_suites が優先される）

    num_steps_wait: int = 10
    num_trials_per_task: int = 50

    # Multi-seed
    seeds: Tuple[int, ...] = (7, 8, 9)
    result_root: str = "AUTO"

def _derive_result_root_from_checkpoint_dir(ckpt_dir: str) -> str:
    """
    例:
      ckpt:   ./checkpoints/teacher/pi05_libero/fineturne_libero/seed7/30000
      result: ./result/libero/teacher/pi05_libero/finetune_libero/seed7/30000
    """
    s = ckpt_dir.replace("\\", "/")

    if s.startswith("./"):
        s_rel = s[2:]
    else:
        s_rel = s

    if s_rel.startswith("checkpoints/"):
        s_rel = s_rel[len("checkpoints/") :]

    s_rel = s_rel.replace("fineturne_libero", "finetune_libero")

    return "./" + str(pathlib.Path("result/libero") / s_rel)

def _resolve_result_root_from_server(args: Args) -> str:
    if args.result_root != "AUTO":
        return args.result_root

    ws = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    md = ws.get_server_metadata()  # ★サーバから metadata 取得 :contentReference[oaicite:2]{index=2}

    ckpt_dir = md.get("ckpt_dir")
    if not ckpt_dir:
        raise RuntimeError(
            f"Server metadata does not include 'ckpt_dir'. metadata={md}"
        )

    return _derive_result_root_from_checkpoint_dir(str(ckpt_dir))

def _fetch_server_metadata(args: "Args") -> Dict[str, Any]:
    """
    WebSocket サーバの metadata から、評価JSONに入れたい情報を抽出して返す。
    最低でも ckpt_dir, ckpt_config を入れる（無い場合は None）。
    """
    try:
        c = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
        md = c.get_server_metadata()
        if not isinstance(md, dict):
            return {"ckpt_dir": None, "ckpt_config": None, "_raw": str(md)}
    except Exception as e:
        logging.warning(f"Failed to fetch server metadata: {e}")
        return {"ckpt_dir": None, "ckpt_config": None, "_error": str(e)}

    keep_keys = ("ckpt_dir", "ckpt_config", "ckpt_source", "env_mode", "train_seed")
    out = {k: md.get(k, None) for k in keep_keys}

    # もし将来必要になったら丸ごと保存できる（今は肥大化防止で入れない）
    # out["_full"] = md

    return out

def _resolve_result_root_from_server_md(args: "Args", server_md: Dict[str, Any]) -> str:
    """
    result_root=AUTO のとき、server_md["ckpt_dir"] から result_root を導出する。
    """
    if args.result_root != "AUTO":
        return args.result_root

    ckpt_dir = server_md.get("ckpt_dir")
    if not ckpt_dir:
        raise RuntimeError(
            "result_root is AUTO but server metadata does not contain ckpt_dir. "
            f"server_metadata={server_md}"
        )
    return _derive_result_root_from_checkpoint_dir(str(ckpt_dir))

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


def eval_libero_seed(args: Args, *, seed: int, server_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
        "server_metadata": copy.deepcopy(server_metadata) if server_metadata is not None else None,
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
    複数 eval_seed を回して:
      - {result_root}/{suite}/json/result_{eval_seed}.json
      - {result_root}/{suite}/json/result.json (eval_seed平均)
    を作る。result_root は ckpt_dir（サーバmetadata）由来で AUTO 生成できる。
    さらに保存JSONに server_metadata（ckpt_dir, ckpt_config など）を埋め込む。
    """
    server_md = _fetch_server_metadata(args)
    resolved_root = _resolve_result_root_from_server_md(args, server_md)

    suite_dir = pathlib.Path(resolved_root) / args.task_suite_name
    json_dir = suite_dir / "json"
    video_dir = suite_dir / "videos"
    json_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    seed_results: Dict[int, Dict[str, Any]] = {}

    for eval_seed in args.seeds:
        args_run = dataclasses.replace(args, result_root=resolved_root)

        r = eval_libero_seed(args_run, seed=int(eval_seed), server_metadata=server_md)
        seed_results[int(eval_seed)] = r

        out_path = json_dir / f"result_{int(eval_seed)}.json"
        _write_json(out_path, r)
        logging.info(f"Wrote: {out_path}")

    avg = _aggregate_seed_results(args.task_suite_name, args.num_trials_per_task, seed_results)

    avg["server_metadata"] = copy.deepcopy(server_md)
    avg["result_root"] = str(resolved_root)

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
