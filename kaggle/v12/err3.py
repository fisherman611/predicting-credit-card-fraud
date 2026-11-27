"""How much of the fraud class looks exactly like legit rows? Mixture estimate on v8 OOF ranks."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'kaggle/v11'); from residual_lm import find, lr, zones
from sklearn.metrics import roc_auc_score
S = sys.argv[1]; root = S + '/v11in'
tr = pd.read_csv(find(root, 'train.csv', 'playground-series-s6e11')); y = tr.is_fraud.values.astype(int); z = zones(tr)
w = lr(np.load(find(root, 'wide_lm_stack_nested.npy'))); dv = lr(pd.read_csv(find(root, 'oof_wall_stack.csv')).sort_values('id').stack_all.values)
v8 = (w + dv) / 2 - 100 * z
q = (pd.Series(v8).rank().values - 0.5) / len(v8)
# Legit-rank of every row: share of legits scored below it
leg = np.sort(v8[y == 0]); u = np.searchsorted(leg, v8) / len(leg)
uf = u[y == 1]
print('fraud legit-rank deciles:', np.round(np.histogram(uf, bins=10, range=(0, 1))[0] / len(uf), 4))
# If a share pi of frauds is distributed like legits, their legit-ranks are uniform; the lower bins hold only them.
for cut in (0.2, 0.3, 0.4, 0.5):
    pi = (uf < cut).mean() / cut
    print(f'pi from legit-rank < {cut}: {pi:.3f}  -> AUC ceiling if all other frauds were perfect: {1 - pi / 2:.4f}')
print('v8 AUC', roc_auc_score(y, v8))
# within the 'signal' part, how good is v8: AUC restricted to frauds above legit-rank 0.5 vs all legits
