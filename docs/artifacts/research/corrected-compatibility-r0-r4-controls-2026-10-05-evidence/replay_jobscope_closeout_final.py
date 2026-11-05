"""Bounded attempt04 terminal replay, frozen evaluator plus reviewed live launcher."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

import psutil

REPO = Path('C:/Users/Admin/PROJECTS/iTRADER/itrader-azr')
ROOT = Path('C:/Users/Admin/ray_results/corrected-compat-r0-r4v1-20261005')
FROZEN = ROOT / 'frozen_source_944d7d30'
EXPECTED_SOURCE = '944d7d308654b93b090fec83a8097728157405671ae8aff858d2d00658fcf233'
TRIALS = [
    'training_fn_ITraderEnv_55cb4_00000_0_2026-10-05_12-41-34',
    'training_fn_ITraderEnv_2c514_00000_0_2026-10-05_12-47-34',
    'training_fn_ITraderEnv_01e66_00000_0_2026-10-05_12-53-32',
    'training_fn_ITraderEnv_dfa55_00000_0_2026-10-05_12-59-44',
    'training_fn_ITraderEnv_be0d6_00000_0_2026-10-05_13-05-58',
    'training_fn_ITraderEnv_9415f_00000_0_2026-10-05_13-11-57',
    'training_fn_ITraderEnv_6b27c_00000_0_2026-10-05_13-17-58',
    'training_fn_ITraderEnv_48faf_00000_0_2026-10-05_13-24-10',
]


def write_new(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cells', required=True)
    parser.add_argument('--output-root', type=Path,
                        default=ROOT / 'study_attempt04_terminal_replay_jobscope20261005')
    args = parser.parse_args()
    cells = [int(value) for value in args.cells.split(',')]
    if len(set(cells)) != len(cells) or not cells or any(value not in range(1, 9) for value in cells):
        parser.error('cells must be unique values from 1 through 8')
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    from itrader.rl_lib.evaluation.provenance import source_tree_fingerprint, checkpoint_sha256, sha256_file
    from itrader.rl_lib.evaluation.run_checkpoint_replay import load_checkpoint_selection
    from itrader.rl_lib.evaluation.run_replay_replication import launch_replay
    from itrader.rl_lib.evaluation.replay_evidence import verify_receipt, replay_launch_succeeded, _validate_episode_replay_contract
    from itrader.rl_lib.evaluation.policy_baselines import summarize_control_episodes
    from itrader.scripts.experiments.analyze_r4_v1_ablation import _terminal_replay_matches

    assert source_tree_fingerprint(FROZEN)['source_tree_sha256'] == EXPECTED_SOURCE
    launcher_source = source_tree_fingerprint(REPO)
    output = args.output_root.resolve()
    if not output.is_relative_to(ROOT.resolve()) or output == ROOT.resolve():
        parser.error('output-root must be a child of the declared run root')
    output.mkdir(exist_ok=True)
    # Child-only numerical resource factor, matching training's Ray-node allocation.
    os.environ['RAY_OVERRIDE_RESOURCES'] = '{"CPU":2}'
    wrapper = (
        'import sys,runpy; '
        f'sys.path.insert(0,{str(FROZEN)!r}); '
        'from itrader.rl_lib.evaluation.provenance import source_tree_fingerprint; '
        f'assert source_tree_fingerprint({str(FROZEN)!r})["source_tree_sha256"]=={EXPECTED_SOURCE!r}; '
        'runpy.run_module("itrader.rl_lib.evaluation.run_checkpoint_replay",run_name="__main__")'
    )
    for cell in cells:
        startup = psutil.virtual_memory().available
        if startup < 5 * 2**30:
            raise RuntimeError(f'Insufficient measured startup headroom: {startup / 2**30:.3f} GiB; no launch')
        trial = TRIALS[cell - 1]
        stable = ROOT / '_stable_metrics' / trial
        selected = load_checkpoint_selection(stable, 'terminal')
        if checkpoint_sha256(selected['stable_checkpoint']) != selected['checkpoint_sha256']:
            raise RuntimeError('Indexed checkpoint bytes changed')
        resolved = json.loads((stable / 'resolved_config.json').read_text())
        if resolved['source_tree_fingerprint']['source_tree_sha256'] != EXPECTED_SOURCE:
            raise RuntimeError('Training source identity changed')
        folder = output / f'cell{cell:02d}'
        folder.mkdir(exist_ok=False)
        declaration = {'cell': cell, 'trial': trial, 'evaluator_source': EXPECTED_SOURCE,
                       'launcher_source': launcher_source, 'operator_sha256': sha256_file(Path(__file__)),
                       'startup_free_bytes': startup, 'ram_hard_floor_bytes': 2**30,
                       'ram_proactive_stop_bytes': int(1.25 * 2**30),
                       'ray_override_resources': {'CPU': 2}, 'timeout_seconds': 300,
                       'checkpoint_sha256': selected['checkpoint_sha256']}
        write_new(folder / 'declaration.json', declaration)
        try:
            command = [sys.executable, '-c', wrapper, '--stable-trial-dir', str(stable),
                       '--out-dir', str(folder / 'out'), '--checkpoint-selection', 'terminal']
            launch = launch_replay(command, log_path=folder / 'replay.log', timeout_seconds=300,
                                   minimum_available_memory_bytes=int(1.25 * 2**30))
            write_new(folder / 'launch.json', launch)
            if not replay_launch_succeeded(launch):
                raise RuntimeError('Replay launch failed its lifetime/resource gate')
            receipt_path = folder / 'out' / f'{trial}.json'
            receipt = verify_receipt(receipt_path)
            _validate_episode_replay_contract(receipt, receipt_path)
            producer = receipt['producer']
            if (producer['launch_id'] != launch['launch_id'] or not any(
                row['pid'] == producer['pid'] and row['process_started_unix'] == producer['process_started_unix']
                for row in launch['observed_processes'])):
                raise RuntimeError('Producer not witnessed by this launch')
            episodes_path = receipt_path.parent / receipt['artifacts'][0]['file']
            episodes = [json.loads(line) for line in episodes_path.read_text().splitlines()]
            with (stable / 'report_rows.csv').open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            terminal = max(rows, key=lambda row: int(row['pbt_progress_updates']))
            if len(episodes) != 32 or not _terminal_replay_matches(
                receipt_path, episodes_path, episodes, stable, resolved, terminal
            ):
                raise RuntimeError('Terminal identity, episode, score or payload mismatch')
            result = {'status': 'accepted_terminal_replay', 'cell': cell, 'trial': trial,
                      'episode_count': 32, 'producer_pid': producer['pid'],
                      'checkpoint_sha256': selected['checkpoint_sha256'],
                      'evaluation_payload_sha256': receipt['evaluation_payload_sha256'],
                      'minimum_free_gib': launch['minimum_available_memory_bytes'] / 2**30,
                      'summary': summarize_control_episodes(episodes), 'economic_promotion': False}
            write_new(folder / 'acceptance.json', result)
            print(json.dumps(result), flush=True)
        except BaseException as exc:
            write_new(folder / 'failure.json', {'cell': cell, 'error': repr(exc), 'status': 'failed_no_retry'})
            raise


if __name__ == '__main__':
    main()
