#!/usr/bin/env python3
"""Run the experimental search matrix sequentially and retain failures as evidence.

Build first: cargo build --locked --release --features npu --example npu_search_probe
The per-case limit is enforced between completed NPU chunks, never by killing work.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', default='target/release/examples/npu_search_probe')
    parser.add_argument('--output', default='results/npu-qualification.json')
    parser.add_argument('--max-seconds', type=int, default=75)
    args = parser.parse_args()
    report_path = Path(args.output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for name in ('HRX_RUNTIME_DIR', 'HRX_BUNDLE_MANIFEST', 'HRX_LOOM_LIBRARY', 'HRX_FABRIC_LIBRARY'):
        env.pop(name, None)
    cases = [(1033, 129, 1), (4096, 768, 1), (256, 384, 60)]
    cases += [(n, d, b) for n in (100_000, 1_000_000) for d in (384, 768) for b in (1, 60)]
    report = {
        'hrx_version': '0.8.11', 'bundle': 'native-20260927-244cd3801b',
        'scope': 'Published bundle; sequential cases; 3 warmups and 10 samples; CPU validation precedes timings; failures are not performance measurements.',
        'cases': [],
    }
    for rows, dimensions, batch in cases:
        # Unique output per run avoids reading an old successful result on failure.
        with tempfile.TemporaryDirectory(prefix='hrxdb-npu-') as directory:
            output = Path(directory) / 'result.json'
            command = [args.binary, '--rows', str(rows), '--dimensions', str(dimensions),
                       '--batch', str(batch), '--k', '10', '--samples', '10', '--warmups', '3',
                       '--max-seconds', str(args.max_seconds), '--compare-gpu', '--output', str(output)]
            print(f'Qualifying {rows} x {dimensions}, batch {batch}', flush=True)
            start = time.monotonic()
            run = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
            case = {'rows': rows, 'dimensions': dimensions, 'batch': batch,
                    'command': command[:-2], 'exit_code': run.returncode,
                    'wall_seconds': time.monotonic() - start, 'stderr': run.stderr}
            if run.returncode == 0:
                case['status'] = 'passed'
                case['result'] = json.loads(output.read_text())
            elif 'probe time limit reached' in run.stderr:
                case['status'] = 'time_limit'
            else:
                case['status'] = 'failed'
            report['cases'].append(case)
            report_path.write_text(json.dumps(report, indent=2) + '\n')
            print(case['status'], flush=True)
    return int(any(case['status'] != 'passed' for case in report['cases']))


if __name__ == '__main__':
    raise SystemExit(main())
