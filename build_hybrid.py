"""Keep original 15-degree solutions; solve missing epochs with 10-degree inputs.

This selects independently solved positions, without interpolation or copying
neighbouring epochs. Not a claim that either mask is more accurate.
"""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import run_benchmark as run

ROOT=Path(__file__).resolve().parent


def read(path):
    with path.open(newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        return reader.fieldnames,list(reader)


def key(row):
    return int(row['gps_week']),round(float(row['gps_tow_s']),6)


def main():
    out=ROOT/'experiments/hybrid/results'
    out.mkdir(parents=True,exist_ok=True)
    fields,baseline=read(ROOT/'results/solution_details.csv')
    _,relaxed=read(ROOT/'experiments/el10/results/solution_details.csv')
    original={key(r):r for r in baseline}
    extra={key(r):r for r in relaxed if key(r) not in original}
    combined={**original,**extra}
    keys=sorted(combined)
    _,epochs=run.load(ROOT/'benchmark/preprocessed_gps')
    expected={(w,round(t,6)) for w,t in epochs}
    assert set(keys)<=expected
    assert len(keys)==1444 and len(extra)==7
    for k in original:
        assert combined[k]==original[k]
    rows=[]
    provenance=[]
    for k in keys:
        r=combined[k]
        xyz=[float(r[c]) for c in run.HEADER[2:]]
        assert np.isfinite(xyz).all() and 6.30e6<np.linalg.norm(xyz)<6.42e6
        rows.append([k[0],f'{k[1]:.1f}',*[f'{v:.3f}' for v in xyz]])
        provenance.append([k[0],f'{k[1]:.6f}',10 if k in extra else 15])
    dest=out/'GXX_benchmark.csv'
    run.write_csv(dest,run.HEADER,rows)
    run.write_csv(out/'solution_details.csv',fields,[[combined[k][f] for f in fields] for k in keys])
    run.write_csv(out/'epoch_sources.csv',['gps_week','gps_tow_s','elevation_mask_deg'],provenance)
    _,diagnostics=read(ROOT/'experiments/el10/inputs/conversion_diagnostics.csv')
    failures={(int(r['week']),round(float(r['tow']),6)):r['reason'] for r in diagnostics}
    missing=sorted(expected-set(keys))
    run.write_csv(out/'omitted_epochs.csv',['gps_week','gps_tow_s','reason'],
                  [[w,f'{t:.6f}',failures[(w,t)]] for w,t in missing])
    baseline_export=read(ROOT/'results/GXX_benchmark.csv')[1]
    hybrid_export=read(dest)[1]
    by_key={(int(r['gps_week']),float(r['gps_tow_s'])):r for r in hybrid_export}
    assert all(by_key[(int(r['gps_week']),float(r['gps_tow_s']))]==r for r in baseline_export)
    assert len(by_key)==len(hybrid_export)==1444
    assert [float(r['gps_tow_s']) for r in hybrid_export]==sorted(float(r['gps_tow_s']) for r in hybrid_export)
    # Also run every course content check; the placeholder group is the sole naming exception.
    import importlib.util
    spec=importlib.util.spec_from_file_location('course_checker',ROOT/'reference_code/check_submission.py')
    checker=importlib.util.module_from_spec(spec); spec.loader.exec_module(checker)
    errors,count,coverage=checker.check(dest,week=2158,epoch_file=ROOT/'benchmark/preprocessed_gps/epochs.csv')
    assert errors==['Use filename G01..G20_benchmark.csv or G01..G20_owndata.csv'],errors
    assert count==1444 and coverage>=90
    summary=dict(input_epochs=len(epochs),solved_epochs=len(keys),omitted_epochs=len(missing),
                 availability_percent=100*len(keys)/len(epochs),
                 unchanged_baseline_positions=len(original),recovered_at_10_deg=len(extra),
                 method='15-degree baseline first; independent 10-degree solution only for absent epochs',
                 interpolated_or_carried_positions=0,benchmark_rmse=None,group_number_pending=True,
                 course_content_checks_passed=True,
                 sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
