# ============================================================
# ÖDEV 5 — KESİN MÜDAHALECİ SHAPLEY DEĞERLERİ
# Tek dosya: Step 0–5, tekrarlar kaldırıldı
# ============================================================

import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import warnings
from math import factorial
warnings.filterwarnings('ignore')

print(f"shap sürümü: {shap.__version__}")

# ============================================================
# BÖLÜM 0 — HAZIRLIK
# ============================================================

bb_model      = joblib.load('bb_model.pkl')
X_train       = joblib.load('X_train.pkl')
X_test        = joblib.load('X_test.pkl')
y_train       = joblib.load('y_train.pkl')
y_test        = joblib.load('y_test.pkl')
feature_names = joblib.load('feature_names.pkl')

# Feature seçimi: top 10
importances = bb_model.feature_importances_
order       = np.argsort(importances)[::-1]
top_10_features = [feature_names[i] for i in order[:10]]
fixed_features  = [feature_names[i] for i in order[10:]]

X_train_10 = X_train[top_10_features].copy()
X_test_10  = X_test[top_10_features].copy()
bg_means_fixed = X_train[fixed_features].mean().to_dict()

print("\n=== FEATURE SEÇİMİ ===")
print(f"Top 10 (SHAP hesaplanacak): {top_10_features}")
print(f"Fixed at bg mean: {fixed_features}")
print(f"Sabit değerler: {bg_means_fixed}")

# Wrapper: 10-feature df → olasılık
def predict_wrapper(X_10):
    if isinstance(X_10, np.ndarray):
        X_10 = pd.DataFrame(X_10, columns=top_10_features)
    else:
        X_10 = X_10[top_10_features].copy()
    for f in fixed_features:
        X_10[f] = bg_means_fixed[f]
    return bb_model.predict_proba(X_10[feature_names])[:, 1]

# 3 kişi: yüksek / orta / düşük risk
test_proba = bb_model.predict_proba(X_test)[:, 1]
persons = {
    'Yüksek risk': int(np.argmax(test_proba)),
    'Orta risk':   int(np.argmin(np.abs(test_proba - 0.5))),
    'Düşük risk':  int(np.argmin(test_proba)),
}
print("\n=== SEÇİLEN 3 KİŞİ ===")
for label, idx in persons.items():
    print(f"  {label:<12} (idx={idx:>2}): proba = {test_proba[idx]:.4f}")

# Sabit arka plan (Step 1–2 için)
bg_sample = X_train_10.sample(n=100, random_state=42).reset_index(drop=True)


# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================

def interventional_shapley(predict_fn, x, bg_df, cols, return_v=False):
    """Exact interventional Shapley. return_v=True ise v(∅) ve v(F) de döner."""
    n  = len(cols)
    bg = bg_df[cols].values
    x  = np.asarray(x, dtype=float)

    # Tüm 2^n alt küme değerleri
    v = np.zeros(2 ** n)
    for mask in range(2 ** n):
        hybrid = bg.copy()
        for j in range(n):
            if (mask >> j) & 1:
                hybrid[:, j] = x[j]
        v[mask] = predict_fn(pd.DataFrame(hybrid, columns=cols)).mean()

    # Shapley değerleri
    phi = np.zeros(n)
    for j in range(n):
        for mask in range(2 ** n):
            if (mask >> j) & 1:
                continue
            s = bin(mask).count('1')
            w = factorial(s) * factorial(n - s - 1) / factorial(n)
            phi[j] += w * (v[mask | (1 << j)] - v[mask])

    if return_v:
        return phi, v
    return phi


def get_ev(exp):
    """expected_value → skaler float (scalar / size-1 / size-2 hepsi)."""
    ev = exp.expected_value
    if isinstance(ev, (np.ndarray, list)):
        ev = np.asarray(ev).flatten()
        return float(ev[1]) if ev.size >= 2 else float(ev[0])
    return float(ev)


def extract_class1(sv):
    """shap_values → (n, features). list / 3D / 2D hepsi."""
    if isinstance(sv, list):
        return sv[1]
    if sv.ndim == 3:
        return sv[:, :, 1]
    return sv


# ============================================================
# BÖLÜM 1 — KENDİ SHAPLEY + 4 AKSİYOM
# ============================================================
print("\n" + "=" * 60)
print("BÖLÜM 1: INTERVENTIONAL SHAPLEY — 3 PERSONS")
print("=" * 60)

results = {}
for label, idx in persons.items():
    x = X_test_10.iloc[idx].values
    phi, v = interventional_shapley(predict_wrapper, x, bg_sample, top_10_features,
                                    return_v=True)
    results[label] = {'phi': phi, 'v': v, 'x': x, 'idx': idx}
    n = len(top_10_features)
    print(f"\n{label} (idx={idx}):  f(x) = {v[-1]:.6f},  E[f(bg)] = {v[0]:.6f}")
    for j, f in enumerate(top_10_features):
        print(f"  {f:<10}: φ = {phi[j]:+.6f}")

# --- Aksiyom testleri ---
print("\n" + "=" * 60)
print("AXIOM TESTS")
print("=" * 60)
axiom_rows = []
n = len(top_10_features)

# 1) Verimlilik
for label, res in results.items():
    phi, v = res['phi'], res['v']
    diff = abs(phi.sum() - (v[-1] - v[0]))
    axiom_rows.append({'Aksiyom': 'Verimlilik', 'Durum': label, 'Fark': diff})
    print(f"[Verimlilik] {label:<12} |fark| = {diff:.2e}")

# 2) Kukla
rng    = np.random.RandomState(42)
bg_dum = bg_sample.copy(); bg_dum['dummy'] = 0.0
x_dum  = np.append(X_test_10.iloc[persons['Yüksek risk']].values, 0.0)
cols_11 = top_10_features + ['dummy']
phi_dum, _ = interventional_shapley(
    lambda X_df: predict_wrapper(X_df[top_10_features]),
    x_dum, bg_dum, cols_11, return_v=True)
print(f"[Kukla]      φ_dummy = {phi_dum[-1]:+.2e}")
axiom_rows.append({'Aksiyom': 'Kukla', 'Durum': 'Yüksek risk',
                   'Fark': abs(phi_dum[-1])})

# 3) Simetri (bg_a = bg_b garantisiyle)
bg_common = rng.normal(1.0, 0.5, 100)
bg_sym    = pd.DataFrame({'a': bg_common, 'b': bg_common})
phi_sym, _ = interventional_shapley(
    lambda X_df: X_df['a'].values * X_df['b'].values,
    np.array([2.0, 2.0]), bg_sym, ['a', 'b'], return_v=True)
sym_diff = abs(phi_sym[0] - phi_sym[1])
print(f"[Simetri]    |φ_a − φ_b| = {sym_diff:.2e}")
axiom_rows.append({'Aksiyom': 'Simetri', 'Durum': 'sentetik a·b',
                   'Fark': sym_diff})

# 4) Toplamsallık
v2_fn = lambda X_df: 0.5 * X_df['age'].values + 0.3 * X_df['thalach'].values / 100
x_main = X_test_10.iloc[persons['Yüksek risk']].values
phi_v1  = interventional_shapley(predict_wrapper, x_main, bg_sample, top_10_features)
phi_v2  = interventional_shapley(v2_fn,           x_main, bg_sample, top_10_features)
phi_v12 = interventional_shapley(lambda X_df: predict_wrapper(X_df) + v2_fn(X_df),
                                 x_main, bg_sample, top_10_features)
add_diff = np.abs(phi_v12 - (phi_v1 + phi_v2)).max()
print(f"[Toplamsallık] max fark = {add_diff:.2e}")
axiom_rows.append({'Aksiyom': 'Toplamsallık', 'Durum': 'Yüksek risk',
                   'Fark': add_diff})

axiom_df = pd.DataFrame(axiom_rows)
axiom_df['Geçti mi?'] = axiom_df['Fark'].apply(lambda x: '✓' if x < 1e-6 else '✗')
print("\n" + axiom_df.to_string(index=False))
axiom_df.to_csv('axiom_tests.csv', index=False)


# ============================================================
# BÖLÜM 2 — KÜTÜPHANE KARŞILAŞTIRMASI (shap.Exact)
# ============================================================
print("\n" + "=" * 60)
print("BÖLÜM 2: KÜTÜPHANE — shap.Exact")
print("=" * 60)

ExactClass = getattr(shap, 'Exact', None) or shap.explainers.Exact
explainer  = ExactClass(predict_wrapper, masker=bg_sample)
print(f"Explainer tipi: {type(explainer).__name__}")

comparison_rows = []
shap_lib_values = []
for label, idx in persons.items():
    x_df = pd.DataFrame([X_test.iloc[idx][top_10_features].values],
                        columns=top_10_features)
    sv = explainer(x_df).values[0]
    shap_lib_values.append(sv)

    phi_ours = results[label]['phi']
    diff     = np.abs(phi_ours - sv)
    max_diff = diff.max()
    print(f"\n{label}: max |φ_ours − φ_shap| = {max_diff:.2e}")

    comparison_rows.append({'Person': label, 'Max_fark': max_diff,
                            'Ort_fark': diff.mean(),
                            '10^-6_altı': max_diff < 1e-6})

cmp_df = pd.DataFrame(comparison_rows)
print("\n" + cmp_df.to_string(index=False))
cmp_df.to_csv('library_comparison.csv', index=False)
np.save('shap_lib_values.npy', np.array(shap_lib_values))


# ============================================================
# BÖLÜM 3 — ARKA PLAN BOYUTU DUYARLILIĞI
# ============================================================
print("\n" + "=" * 60)
print("BÖLÜM 3: ARKA PLAN BOYUTU {1,10,100,500} × 5 seed")
print("=" * 60)

# Referans: tüm X_train
ref_phi = {}
for label, idx in persons.items():
    x = X_test.iloc[idx][top_10_features].values
    ref_phi[label] = interventional_shapley(predict_wrapper, x,
                                            X_train_10, top_10_features)
    print(f"  {label:<12} ||φ_ref||₂ = {np.linalg.norm(ref_phi[label]):.6f}")

bg_sizes = [1, 10, 100, 500]
seeds    = [42, 43, 44, 45, 46]
all_results = {label: {s: [] for s in bg_sizes} for label in persons}

for label, idx in persons.items():
    x = X_test.iloc[idx][top_10_features].values
    for size in bg_sizes:
        for seed in seeds:
            rng  = np.random.RandomState(seed)
            idxs = rng.choice(len(X_train_10), size=size,
                              replace=(size > len(X_train_10)))
            bg   = X_train_10.iloc[idxs].reset_index(drop=True)
            phi  = interventional_shapley(predict_wrapper, x, bg, top_10_features)
            rel  = np.linalg.norm(phi - ref_phi[label]) / (np.linalg.norm(ref_phi[label]) + 1e-12)
            all_results[label][size].append(rel)
        print(f"  {label:<12} size={size:>3}: {np.mean(all_results[label][size]):.4f}"
              f" ± {np.std(all_results[label][size]):.4f}")

# Grafik
fig, ax = plt.subplots(figsize=(10, 6))
for label in persons:
    means = [np.mean(all_results[label][s]) for s in bg_sizes]
    stds  = [np.std(all_results[label][s])  for s in bg_sizes]
    ax.errorbar(bg_sizes, means, yerr=stds, marker='o', capsize=5,
                label=label, linewidth=2, markersize=8)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Arka plan boyutu (log)'); ax.set_ylabel('Göreli sapma (L2, log)')
ax.set_title('Referanstan göreli sapma vs arka plan boyutu')
ax.legend(); ax.grid(alpha=0.3, which='both')
plt.tight_layout(); plt.savefig('bg_size_sensitivity.png', dpi=150); plt.show()

summary = pd.DataFrame({
    label: [f"{np.mean(all_results[label][s]):.4f} ± {np.std(all_results[label][s]):.4f}"
            for s in bg_sizes] for label in persons
}, index=[f"size={s}" for s in bg_sizes])
print("\n" + summary.to_string())
summary.to_csv('bg_size_sensitivity.csv')


# ============================================================
# BÖLÜM 4 — REFERANS GRUBU DUYARLILIĞI
# ============================================================
print("\n" + "=" * 60)
print("BÖLÜM 4: REFERANS GRUBU DUYARLILIĞI")
print("=" * 60)

pred_train = bb_model.predict_proba(X_train)[:, 1]
MIN_GROUP_SIZE = 20

def build_groups(person_full):
    age_bin = int(person_full['age'] // 10) * 10
    age_bins_train = (X_train_10['age'] // 10 * 10).astype(int)
    return {
        'Tümü':          X_train_10,
        'Onaylananlar':  X_train_10[pred_train > 0.5],
        'Reddedilenler': X_train_10[pred_train < 0.5],
        'Aynı cinsiyet': X_train_10[X_train_10['sex'] == person_full['sex']],
        'Aynı yaş grubu':X_train_10[age_bins_train == age_bin],
    }

all_person_results = {}
for label, idx in persons.items():
    x_10   = X_test_10.iloc[idx].values
    x_full = X_test.iloc[idx]
    groups = build_groups(x_full)
    pres   = {'groups': {}, 'phis': {}, 'bases': {}}
    print(f"\n{label} (idx={idx}):")

    for gname, gdf in groups.items():
        n_bg = len(gdf)
        if n_bg < MIN_GROUP_SIZE:
            print(f"  [{gname:<20}] n={n_bg} — ATLANDI")
            continue
        if n_bg > 200:
            gdf = gdf.sample(n=200, random_state=42).reset_index(drop=True)
        phi, v = interventional_shapley(predict_wrapper, x_10, gdf,
                                        top_10_features, return_v=True)
        pres['groups'][gname] = gdf
        pres['phis'][gname]   = phi
        pres['bases'][gname]  = v[0]
        print(f"  [{gname:<20}] n={n_bg:>3}  base={v[0]:+.4f}")

    all_person_results[label] = pres

# İşaret değişimi
print("\n" + "=" * 60 + "\nİŞARET DEĞİŞİMİ ANALİZİ\n" + "=" * 60)
sign_flip_summary = []
for label, pres in all_person_results.items():
    group_names = list(pres['phis'].keys())
    print(f"\n{label}:")
    print(f"  {'Öznitelik':<10} " + " ".join(f"{g[:9]:>10}" for g in group_names))
    for j, f in enumerate(top_10_features):
        vals   = [pres['phis'][g][j] for g in group_names]
        signs  = [np.sign(v) for v in vals]
        n_flip = int(np.sum(np.diff(signs) != 0))
        row_str = f"  {f:<10} " + " ".join(f"{v:>+10.4f}" for v in vals)
        if n_flip > 0:
            row_str += "  ⚠️ FLIP"
        print(row_str)
        sign_flip_summary.append({'Person': label, 'Feature': f,
                                  'n_sign_flips': n_flip,
                                  'values': [round(v, 4) for v in vals]})

print("\n" + "=" * 60 + "\nÖZET — İşaret Değiştiren Öznitelikler\n" + "=" * 60)
for label in persons:
    flips = [r['Feature'] for r in sign_flip_summary
             if r['Person'] == label and r['n_sign_flips'] > 0]
    print(f"  {label:<14}: {flips if flips else 'YOK'}")

# Heatmap
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, label in zip(axes, persons):
    pres = all_person_results[label]
    gnames = list(pres['phis'].keys())
    mat = np.array([pres['phis'][g] for g in gnames])
    im = ax.imshow(mat, cmap='RdBu_r', aspect='auto',
                   vmin=-np.abs(mat).max(), vmax=np.abs(mat).max())
    ax.set_xticks(range(len(top_10_features)))
    ax.set_xticklabels(top_10_features, rotation=90, fontsize=8)
    ax.set_yticks(range(len(gnames)))
    ax.set_yticklabels(gnames, fontsize=8)
    ax.set_title(f'{label} (base = {pres["bases"]})')
    plt.colorbar(im, ax=ax, fraction=0.04, label='φ')
plt.suptitle('Referans Grubu × Öznitelik — Shapley Değerleri', fontsize=13)
plt.tight_layout(); plt.savefig('reference_group_heatmap.png', dpi=150); plt.show()

joblib.dump(all_person_results, 'reference_group_results.pkl')
pd.DataFrame(sign_flip_summary).to_csv('sign_flip_summary.csv', index=False)


# ============================================================
# BÖLÜM 5 — PATH-DEPENDENT vs INTERVENTIONAL TREESHAP
# ============================================================
print("\n" + "=" * 60)
print("BÖLÜM 5: PATH-DEPENDENT vs INTERVENTIONAL TREESHAP")
print("=" * 60)

explainer_path = shap.TreeExplainer(bb_model,
                                    feature_perturbation='tree_path_dependent')
bg_full = X_train.sample(n=100, random_state=42).reset_index(drop=True)
explainer_interv = shap.TreeExplainer(bb_model, data=bg_full,
                                      feature_perturbation='interventional')

ev_path   = get_ev(explainer_path)
ev_interv = get_ev(explainer_interv)
print(f"Path  expected_value : {ev_path:.6f}")
print(f"Interv expected_value: {ev_interv:.6f}")

X_explain  = X_test.iloc[list(persons.values())].copy()
shap_path  = extract_class1(explainer_path.shap_values(X_explain))
shap_interv= extract_class1(explainer_interv.shap_values(X_explain))

print(f"\nshap_path shape:   {shap_path.shape}")
print(f"shap_interv shape: {shap_interv.shape}")

diff = np.abs(shap_path - shap_interv)
comparison_rows = []
for i, (label, idx) in enumerate(persons.items()):
    print(f"\n{label} (idx={idx}):  "
          f"max={diff[i].max():.6f}  mean={diff[i].mean():.6f}")
    comparison_rows.append({'Person': label, 'idx': idx,
                            'max_abs_diff':  diff[i].max(),
                            'mean_abs_diff': diff[i].mean()})

# Completeness
log_odds = bb_model.decision_function(X_explain)
print(f"\n{'Person':<14} {'f(x)':>12} {'err_path':>12} {'err_interv':>12}")
for i, (label, _) in enumerate(persons.items()):
    err_p = abs(shap_path[i].sum()   + ev_path   - log_odds[i])
    err_i = abs(shap_interv[i].sum() + ev_interv - log_odds[i])
    print(f"{label:<14} {log_odds[i]:>+12.6f} {err_p:>12.2e} {err_i:>12.2e}")

# Görsel
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
for i, label in enumerate(persons.keys()):
    axes[0].scatter(shap_path[i], shap_interv[i],
                    s=80, alpha=0.7, edgecolor='black', label=label)
lim = max(np.abs(shap_path).max(), np.abs(shap_interv).max()) * 1.1
axes[0].plot([-lim, lim], [-lim, lim], 'r--', alpha=0.5, label='y = x')
axes[0].set_xlabel('Path-dependent SHAP'); axes[0].set_ylabel('Interventional SHAP')
axes[0].set_title('Path vs Interventional — Tüm 3 Kişi × 13 Öznitelik')
axes[0].legend(); axes[0].grid(alpha=0.3)
axes[0].axhline(0, color='k', linewidth=0.4); axes[0].axvline(0, color='k', linewidth=0.4)

x_pos = np.arange(len(feature_names))
axes[1].bar(x_pos - 0.2, shap_path[0],   width=0.4, label='Path',
            color='steelblue', edgecolor='black')
axes[1].bar(x_pos + 0.2, shap_interv[0], width=0.4, label='Interv',
            color='darkorange', edgecolor='black')
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(feature_names, rotation=60, ha='right', fontsize=9)
axes[1].axhline(0, color='k', linewidth=0.4)
axes[1].set_ylabel('SHAP değeri (log-odds)')
axes[1].set_title(f'Yüksek riskli kişi (idx={persons["Yüksek risk"]})')
axes[1].legend()
plt.tight_layout(); plt.savefig('treeshap_comparison.png', dpi=150); plt.show()

cmp_df = pd.DataFrame(comparison_rows)
print("\n" + "=" * 60 + "\nÖZET TABLO\n" + "=" * 60)
print(cmp_df.to_string(index=False))

cmp_df.to_csv('treeshap_comparison.csv', index=False)
np.save('shap_path_values.npy',   shap_path)
np.save('shap_interv_values.npy', shap_interv)

print("\n=== TÜM ADIMLAR TAMAMLANDI ===")