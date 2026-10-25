"""Verify P006 evidence in a temporary directory with no datasets or models."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


TASK = Path(__file__).resolve().parents[1]


def main() -> None:
    output = TASK / 'report/p006_portable_verification.json'
    if output.exists():
        raise FileExistsError(f'Portable verification already saved: {output}')
    with tempfile.TemporaryDirectory(prefix='t_lab_p006_portable_') as temporary:
        isolated = Path(temporary)
        for name in ['verify_final_snapshot.py', 'verify_auto_checkpoint.py']:
            shutil.copyfile(TASK / 'src' / name, isolated / name)
        shutil.copytree(TASK / 'report/p006_snapshot', isolated / 'snapshot')
        assert not list(isolated.rglob('*.parquet'))
        assert not list(isolated.rglob('*.npy'))
        assert not list(isolated.rglob('*.npz'))
        assert not list(isolated.rglob('*.cbm'))
        result = subprocess.run(
            [sys.executable, str(isolated / 'verify_final_snapshot.py'),
             '--snapshot', str(isolated / 'snapshot')],
            cwd=isolated, capture_output=True, text=True, check=True,
        )
        verified = json.loads(result.stdout)
        if verified['status'] != 'PASS':
            raise AssertionError(verified)
        report = {
            'experiment': 'P006', 'isolated_copy': True,
            'copied_modules': ['verify_final_snapshot.py', 'verify_auto_checkpoint.py'],
            'copied_data': 'compact snapshot only', 'dataset_model_paths_present': False,
            'model_training_performed': False, 'verification': verified,
        }
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
