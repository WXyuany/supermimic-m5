"""Export all scalar curves, readable plots and per-clip results from measured runs."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--retuned', type=Path, required=True)
    p.add_argument('--benchmark', type=Path, required=True)
    p.add_argument('--out', type=Path, default=Path('results'))
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    records = {}
    for label, run in [('baseline', a.baseline), ('retuned', a.retuned)]:
        ev = EventAccumulator(str(run / 'tb'), size_guidance={'scalars': 0}).Reload()
        records[label] = {tag: [{'step': e.step, 'value': e.value} for e in ev.Scalars(tag)]
                          for tag in ev.Tags()['scalars']}
    (a.out / 'tensorboard_scalars.json').write_text(json.dumps(records))
    with (a.out / 'tensorboard_scalars.csv').open('w', newline='') as f:
        w = csv.writer(f, lineterminator="\n"); w.writerow(['stage', 'tag', 'iteration', 'value'])
        for stage, tags in records.items():
            for tag, rows in tags.items():
                for row in rows: w.writerow([stage, tag, row['step'], row['value']])
    tags = ['train/episode_norm_return', 'wm/online_nse_8step', 'policy/imag_return_undisc', 'sampler/top10_probability']
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout='constrained')
    for ax, tag in zip(axes.flat, tags):
        for label, color in [('baseline', '#9b8ac4'), ('retuned', '#147f93')]:
            rows = records[label].get(tag, [])
            if rows: ax.plot([r['step'] for r in rows], [r['value'] for r in rows], color=color, lw=1, label=label)
        ax.axvline(12500, color='#777', linestyle='--', lw=.8)
        ax.set(title=tag, xlabel='Training iteration'); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('Apple M5 / LAFAN1 — measured training scalars (not benchmark scores)')
    fig.savefig(a.out / 'training_curves.png', dpi=160); plt.close(fig)
    from matplotlib.backends.backend_pdf import PdfPages
    all_tags = sorted(set().union(*(set(v) for v in records.values())))
    with PdfPages(a.out / 'all_training_curves.pdf') as pdf:
        for offset in range(0, len(all_tags), 6):
            fig, axes = plt.subplots(3, 2, figsize=(11.7, 8.3), layout='constrained')
            for ax, tag in zip(axes.flat, all_tags[offset:offset+6]):
                for label, values in records.items():
                    rows = values.get(tag, [])
                    if rows: ax.plot([r['step'] for r in rows], [r['value'] for r in rows], lw=.8, label=label)
                ax.set(title=tag, xlabel='Training iteration'); ax.grid(alpha=.2); ax.legend(fontsize=7)
            for ax in list(axes.flat)[len(all_tags[offset:offset+6]):]: ax.set_visible(False)
            fig.suptitle('Apple M5 - all recorded TensorBoard scalars')
            pdf.savefig(fig); plt.close(fig)
    benchmark = json.loads(a.benchmark.read_text())
    rows = sorted(benchmark['clips'], key=lambda r: r['norm_return'])
    with (a.out / 'per_clip.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['name', 'norm_return', 'steps', 'frames', 'failed', 'fail_step'], extrasaction='ignore', lineterminator='\n')
        w.writeheader(); w.writerows(rows)
    fig, ax = plt.subplots(figsize=(12, 10), layout='constrained')
    ax.barh([r['name'] for r in rows], [r['norm_return'] for r in rows], color=['#c77568' if r['failed'] else '#147f93' for r in rows])
    ax.tick_params(axis='y', labelsize=8)
    ax.set(xlabel='Return normalized over executed steps', title=f"M5 / {len(rows)} clips / mean {benchmark['summary']['mean_norm_return']:.4f} / {benchmark['summary']['failed']} terminated early")
    fig.savefig(a.out / 'evaluation.png', dpi=150); plt.close(fig)
    print(f'Exported {len(all_tags)} scalar tags and {len(rows)} clip results to {a.out}')


if __name__ == '__main__': main()
