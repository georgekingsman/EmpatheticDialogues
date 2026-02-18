"""Quick score distribution analysis."""
import sys, os, json
sys.path.insert(0, ".")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from collections import Counter, defaultdict

for fname in ['gpt2_finetuned_judge.jsonl', 'empathy_chain_judge.jsonl']:
    path = f'outputs/judge/{fname}'
    records = [json.loads(l) for l in open(path)]
    scored = [r for r in records if 'scores' in r]

    by_sample = defaultdict(list)
    for r in scored:
        by_sample[r['sample_id']].append(r)

    avgs = []
    for sid, repeats in by_sample.items():
        avg_overall = sum(r['overall'] for r in repeats) / len(repeats)
        avgs.append((sid, avg_overall, repeats[0].get('notes', '')))
    avgs.sort(key=lambda x: x[1], reverse=True)

    print(f'\n=== Top-5 scored samples from {fname} ===')
    for sid, avg, notes in avgs[:5]:
        print(f'  {sid}: overall={avg:.1f} | {notes[:150]}')

    print(f'  Score distribution (overall):')
    overall_vals = [r['overall'] for r in scored]
    for score, count in sorted(Counter(overall_vals).items()):
        print(f'    {score}: {count} ({count/len(overall_vals)*100:.0f}%)')
