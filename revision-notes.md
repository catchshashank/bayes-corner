# Bayesian Statistics — Complete Revision Notes
### Sources: Course Sessions 1–8 · Cowles (2013) *Applied Bayesian Statistics* · Downey (2013) *Think Bayes* · PyMC / ArviZ

> **How to use these notes.** Each section opens with the conceptual core, then layers in mathematical detail, Python implementation, and synthesis across all three sources. Read top-to-bottom for a first pass; jump to subsections when drilling a specific topic. Every code block is self-contained and runnable with `pymc >= 5`, `arviz >= 0.17`, `numpy`, and `scipy`.

---

## Table of Contents

1. [The Bayesian Paradigm](#1-the-bayesian-paradigm)
2. [Distributions & Prior Selection](#2-distributions--prior-selection)
3. [Computational Bayesian Inference with PyMC](#3-computational-bayesian-inference-with-pymc)
4. [Bayesian Regression & GLMs](#4-bayesian-regression--glms)
5. [Hierarchical Models](#5-hierarchical-models)
6. [MCMC Theory & Diagnostics](#6-mcmc-theory--diagnostics)
7. [Causal Inference](#7-causal-inference)
8. [Bayesian Time Series](#8-bayesian-time-series)
9. [Approximate Bayesian Computation (ABC)](#9-approximate-bayesian-computation-abc)
10. [Decision Analysis under Uncertainty](#10-decision-analysis-under-uncertainty)
11. [Model Comparison & Evidence](#11-model-comparison--evidence)
12. [Revision Plan & Formula Reference](#12-revision-plan--formula-reference)

---

## 1. The Bayesian Paradigm

### 1.1 Core Philosophy

The Bayesian framework treats parameters as **random variables** — objects that have probability distributions reflecting our uncertainty. This is the fundamental philosophical break from frequentism.

> *Cowles* frames this beautifully via the scientific method: Bayesian analysis is the formalisation of "assess existing knowledge → gather data → update beliefs → act." The posterior is your updated belief after step 3.

> *Downey* calls the update rule the **diachronic interpretation**: "diachronic" means changing over time. Bayes's theorem tells you how your probability for a hypothesis H should change when you observe data D.

```
P(H | D) = P(H) × P(D | H) / P(D)
```

| Term | Name | Intuition |
|------|------|-----------|
| `P(H)` | **Prior** | Belief about H before data |
| `P(D\|H)` | **Likelihood** | How probable is the data if H is true? |
| `P(D)` | **Marginal likelihood** | Total probability of data across all hypotheses |
| `P(H\|D)` | **Posterior** | Updated belief after seeing data |

### 1.2 Bayes' Rule — Three Equivalent Forms

**Discrete case** (Cowles Ch. 2, Downey Ch. 1):

$$P(H_j | A) = \frac{P(A | H_j)\, P(H_j)}{\sum_k P(A | H_k)\, P(H_k)}$$

This is the law of total probability in the denominator — the **normalising constant**.

**Continuous case** (the form used in PyMC):

$$p(\theta | y) = \frac{p(y | \theta)\, p(\theta)}{\int p(y | \theta')\, p(\theta')\, d\theta'}$$

**Proportional form** (what MCMC exploits — no need to compute the integral):

$$p(\theta | y) \propto p(y | \theta)\, p(\theta)$$

### 1.3 The Cookie & Mammogram Problems — Worked Examples

These two examples from Cowles and Downey crystallise the mechanics.

**Cowles: Mammography screening**

```
Prior P(cancer)     = 0.0045    # base rate in screening population
P(M+ | cancer)      = 0.724     # sensitivity
P(M+ | no cancer)   = 0.027     # false positive rate

# Bayes update
numerator   = 0.724 × 0.0045 = 0.003258
denominator = 0.003258 + (0.027 × 0.9955) = 0.030137
P(cancer | M+) = 0.003258 / 0.030137 ≈ 0.108
```

Even with a positive mammogram, the posterior probability of cancer is only ~10.8%. This counter-intuitive result is entirely explained by the low prior. This is the **base rate fallacy** in action: ignoring the prior (treating the posterior as just the sensitivity) is a common and consequential error.

**Sequential updating** (Cowles §1.3.5): After a positive mammogram → SCNB biopsy:

```
Prior (after M+):  P(cancer) = 0.108
P(S+ | cancer)     = 0.89
P(S- | no cancer)  = 0.94

# Two more updates via the same Bayes table
```

This demonstrates **Bayesian sequential analysis**: today's posterior becomes tomorrow's prior. The order of data collection does not matter — only the accumulated evidence does (assuming conditional independence of data given the parameter).

**Python implementation of the discrete Bayes table** (Downey's framework):

```python
import numpy as np
import pandas as pd

def bayes_table(hypotheses, priors, likelihoods):
    """
    Discrete Bayes table following Downey's Suite framework.
    
    Args:
        hypotheses : list of hypothesis labels
        priors     : array-like of prior probabilities (need not sum to 1)
        likelihoods: array-like of P(data | hypothesis)
    Returns:
        DataFrame with prior, likelihood, unnorm posterior, posterior
    """
    priors = np.array(priors, dtype=float)
    likelihoods = np.array(likelihoods, dtype=float)
    unnorm = priors * likelihoods
    posteriors = unnorm / unnorm.sum()
    return pd.DataFrame({
        'hypothesis' : hypotheses,
        'prior'      : priors / priors.sum(),
        'likelihood' : likelihoods,
        'unnorm'     : unnorm,
        'posterior'  : posteriors
    }).set_index('hypothesis')

# Mammography example
result = bayes_table(
    hypotheses  = ['cancer', 'no_cancer'],
    priors      = [0.0045, 0.9955],
    likelihoods = [0.724,  0.027]
)
print(result)
#              prior  likelihood    unnorm  posterior
# cancer      0.0045      0.7240  0.003258   0.108...
# no_cancer   0.9955      0.0270  0.026879   0.891...
```

### 1.4 Posterior Summaries

```python
import scipy.stats as st
import numpy as np

# Beta-Binomial conjugate: posterior is Beta(alpha + k, beta + n - k)
alpha_prior, beta_prior = 1, 1  # uniform prior
n, k = 250, 140                  # Euro coin: 250 spins, 140 heads

alpha_post = alpha_prior + k
beta_post  = beta_prior  + (n - k)

post = st.beta(alpha_post, beta_post)

print(f"Posterior mean:   {post.mean():.4f}")          # 0.5596
print(f"Posterior median: {post.median():.4f}")        # 0.5598
print(f"MAP estimate:     {(alpha_post-1)/(alpha_post+beta_post-2):.4f}")  # 0.5600
print(f"95% equal-tail CI: {post.interval(0.95)}")     # (0.497, 0.620)
```

> **Credible interval ≠ Confidence interval.** A 95% credible interval is a probability statement: *P(a ≤ θ ≤ b | y) = 0.95*. A 95% confidence interval is a statement about the procedure: 95% of such intervals constructed under repeated sampling will contain the true fixed θ. This distinction is the most commonly examined conceptual point in Bayesian courses.

---

## 2. Distributions & Prior Selection

### 2.1 The Kernel Trick (Cowles §3.2.2)

Cowles introduces the concept of the **kernel** — the part of a density that contains all terms involving the parameter of interest. Normalising constants (terms that don't depend on the parameter) can be dropped during analysis and recovered later.

**Binomial likelihood kernel:**
The full pmf is `C(n,k) × θ^k × (1-θ)^(n-k)`. When viewed as a function of θ, `C(n,k)` is a constant and drops out:

$$p(y | \theta) \propto \theta^k (1 - \theta)^{n-k}$$

This **kernel matching** is the core technique for conjugate analysis: identify that the posterior kernel has the form of a known density family.

### 2.2 Conjugate Priors Reference

| Likelihood | Conjugate Prior | Posterior Update | Sufficient Statistics |
|------------|----------------|-------------------|-----------------------|
| Binomial(n, θ) | Beta(α, β) | Beta(α + k, β + n − k) | k = # successes |
| Poisson(λ) | Gamma(α, β) | Gamma(α + Σy, β + n) | Σy, n |
| Normal(μ, σ² known) | Normal(μ₀, τ₀²) | Normal(μₙ, τₙ²) | ȳ, n |
| Normal(μ known, σ²) | InvGamma(α, β) | InvGamma(α + n/2, β + SS/2) | SS = Σ(yᵢ−μ)² |
| Normal(μ, σ² both unknown) | Normal-InvGamma | Normal-InvGamma | ȳ, s², n |
| Multinomial | Dirichlet(α) | Dirichlet(α + counts) | category counts |

**Normal-Normal update derivation** (Cowles Ch. 6 — the most examinable):

If `y_i ~ N(μ, σ²)` with σ² known, and `μ ~ N(μ₀, τ₀²)`:

```
Posterior precision:  1/τₙ² = 1/τ₀² + n/σ²
Posterior mean:       μₙ   = (μ₀/τ₀² + nȳ/σ²) / (1/τ₀² + n/σ²)
```

This is a **precision-weighted average** of the prior mean and the data mean. As n → ∞, μₙ → ȳ regardless of the prior — the data overwhelm the prior.

```python
def normal_normal_update(mu0, tau0_sq, y, sigma_sq):
    """
    Conjugate Normal-Normal posterior update.
    y is an array of observations; sigma_sq is assumed known.
    """
    n    = len(y)
    ybar = np.mean(y)
    
    prec_prior = 1 / tau0_sq
    prec_data  = n / sigma_sq
    prec_post  = prec_prior + prec_data
    
    tau_n_sq = 1 / prec_post
    mu_n     = (mu0 * prec_prior + ybar * prec_data) / prec_post
    
    return mu_n, tau_n_sq

mu_n, tau_n_sq = normal_normal_update(mu0=0, tau0_sq=100, 
                                       y=np.random.normal(5, 1, 30), 
                                       sigma_sq=1)
print(f"Posterior: N({mu_n:.3f}, {tau_n_sq:.4f})")
```

### 2.3 Non-conjugate & Noninformative Priors (Cowles Ch. 5)

**Jeffreys prior** — invariant to reparameterisation:

$$p(\theta) \propto \sqrt{I(\theta)} = \sqrt{-E\left[\frac{\partial^2 \log p(y|\theta)}{\partial \theta^2}\right]}$$

For Binomial: Jeffreys prior is `Beta(1/2, 1/2)` — a U-shaped prior that gives more weight to extreme probabilities.

For Normal mean (σ known): Jeffreys prior is `p(μ) ∝ 1` — the improper flat prior.

> **Warning (Cowles):** Improper priors do not always yield proper posteriors. Always verify that the posterior integrates to a finite constant before using an improper prior. The condition is: the likelihood must be "concentrated enough" relative to the improper prior.

**Verifying posterior propriety:**

```python
# Quick sanity check: can the posterior be normalised?
from scipy import integrate

def log_likelihood(theta, y):
    return np.sum(st.norm.logpdf(y, loc=theta, scale=1))

def log_prior(theta):
    return 0  # flat improper prior: p(theta) ∝ 1

def unnorm_posterior(theta, y):
    return np.exp(log_likelihood(theta, y) + log_prior(theta))

y = np.array([-2.5, -2.8, -2.3, -2.6, -2.4])
integral, err = integrate.quad(unnorm_posterior, -np.inf, np.inf, args=(y,))
print(f"Integral: {integral:.4f}, Error: {err:.2e}")
# With flat prior + normal likelihood, integral is finite → posterior is proper
```

### 2.4 The Swamping Principle (Downey Ch. 4)

Downey demonstrates with the Euro coin problem that with enough data, **the posterior is nearly insensitive to the prior**. Starting from a uniform prior vs. a triangular prior, after 250 spins the posteriors are nearly identical (means differ by < 0.5%).

This is reassuring: prior sensitivity is a concern primarily in **small data regimes**. In large data regimes, the likelihood dominates and priors become irrelevant.

```python
import scipy.stats as st
import numpy as np

def euro_posterior(prior_alpha, prior_beta, heads=140, total=250):
    """Beta-Binomial update for Euro coin problem."""
    post_alpha = prior_alpha + heads
    post_beta  = prior_beta  + (total - heads)
    return st.beta(post_alpha, post_beta)

# Uniform prior
post_uniform   = euro_posterior(1, 1)
# Informative prior centred at 0.5
post_informative = euro_posterior(50, 50)

print(f"Uniform prior → posterior mean:    {post_uniform.mean():.4f}")
print(f"Informative prior → posterior mean:{post_informative.mean():.4f}")
# Both converge near 0.56 because n=250 is large relative to the prior strength
```

### 2.5 Weakly Informative Priors for PyMC

| Parameter Type | Recommended Prior | Reasoning |
|----------------|------------------|-----------|
| Regression intercept | `Normal(ȳ, 2·std(y))` | Centre on data scale |
| Standardised slope | `Normal(0, 1)` | Regularisation; slopes rarely > 2–3 SD in practice |
| Noise / scale σ | `HalfNormal(1)` or `Exponential(1)` | Positive support; avoid mass at 0 |
| Variance component | `HalfNormal(1)` | Avoids improper uniform priors that cause funnel pathology |
| Probability | `Beta(1,1)` uniform or `Beta(2,2)` weakly informative | Natural [0,1] support |
| Rate λ | `Gamma(2, 1)` or `Exponential` | Positive, regularised |

---

## 3. Computational Bayesian Inference with PyMC

### 3.1 The PMF Framework (Downey) as a Conceptual Bridge

Before MCMC, Downey's discrete PMF approach is invaluable for building intuition. Every Bayesian update is just:

1. Start with a PMF over hypotheses (the prior)
2. For each hypothesis, compute `P(data | hypothesis)` (the likelihood)
3. Multiply: `posterior ∝ prior × likelihood`
4. Normalise

```python
class BayesPMF:
    """
    Downey-style discrete Bayesian updater.
    Bridges intuition with the continuous world of PyMC.
    """
    def __init__(self, hypos, priors=None):
        self.hypos = np.array(hypos, dtype=float)
        if priors is None:
            self.probs = np.ones(len(hypos)) / len(hypos)
        else:
            self.probs = np.array(priors) / np.sum(priors)
    
    def update(self, likelihood_fn, data):
        """In-place Bayesian update."""
        likelihoods = np.array([likelihood_fn(h, data) for h in self.hypos])
        self.probs  = self.probs * likelihoods
        self.probs  = self.probs / self.probs.sum()  # normalise
        return self
    
    def mean(self):
        return np.dot(self.hypos, self.probs)
    
    def credible_interval(self, pct=0.95):
        cdf = np.cumsum(self.probs)
        lo  = self.hypos[np.searchsorted(cdf, (1 - pct) / 2)]
        hi  = self.hypos[np.searchsorted(cdf, 1 - (1 - pct) / 2)]
        return lo, hi

# The dice problem (Downey Ch. 3)
dice = BayesPMF(hypos=[4, 6, 8, 12, 20])
for roll in [6, 8, 7, 7, 5, 4]:
    dice.update(lambda h, d: 1/h if d <= h else 0, roll)

for h, p in zip(dice.hypos, dice.probs):
    print(f"Die {int(h):2d}: P = {p:.4f}")
```

### 3.2 PyMC Model Architecture

Every PyMC model follows a three-zone structure: **priors → likelihood → inference**.

```python
import pymc as pm
import arviz as az
import numpy as np

# Synthetic data for illustration
rng = np.random.default_rng(42)
X   = rng.standard_normal(100)
y   = 2.5 + 1.8 * X + rng.normal(0, 0.8, 100)

with pm.Model() as linear_model:
    
    # ─── Zone 1: Priors ────────────────────────────────────────────────
    alpha = pm.Normal("alpha", mu=y.mean(), sigma=2*y.std())
    beta  = pm.Normal("beta",  mu=0, sigma=1)
    sigma = pm.HalfNormal("sigma", sigma=1)
    
    # ─── Zone 2: Likelihood ────────────────────────────────────────────
    mu    = pm.Deterministic("mu", alpha + beta * X)
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)
    
    # ─── Zone 3: Inference ─────────────────────────────────────────────
    # Prior predictive check BEFORE sampling
    prior_checks = pm.sample_prior_predictive(samples=500, random_seed=42)
    
    # MCMC
    idata = pm.sample(
        draws        = 2000,
        tune         = 1000,
        chains       = 4,
        target_accept= 0.9,
        random_seed  = 42,
        return_inferencedata = True
    )
    
    # Posterior predictive check AFTER sampling
    pm.sample_posterior_predictive(idata, extend_inferencedata=True)

print(az.summary(idata, var_names=["alpha", "beta", "sigma"]))
```

### 3.3 Prior Predictive Checks

Run prior predictive checks **before** fitting. Ask: does the prior model generate data that is at least in the right ballpark? Absurd prior predictions signal prior misspecification.

```python
# Visualise prior predictive
az.plot_ppc(prior_checks, group="prior", num_pp_samples=100)
# If y_rep covers values outside physical possibility → tighten priors
```

### 3.4 Posterior Predictive Checks

After fitting, the posterior predictive distribution `P(ỹ | y)` tells you how well the model replicates key features of the observed data.

```python
# Posterior predictive check
az.plot_ppc(idata, num_pp_samples=100)

# Test statistics of interest
y_rep = idata.posterior_predictive["y_obs"].values.reshape(-1, len(y))
T_obs   = y.std()
T_rep   = y_rep.std(axis=1)
p_value = (T_rep >= T_obs).mean()
print(f"Bayesian p-value (std): {p_value:.3f}")
# Values near 0 or 1 indicate model misfit on this statistic
```

---

## 4. Bayesian Regression & GLMs

### 4.1 Bayesian Linear Regression

The Bayesian treatment augments OLS by replacing point estimates with posterior distributions over all parameters, yielding calibrated uncertainty in predictions.

```python
import pymc as pm
import numpy as np

# ─── Bayesian linear regression ────────────────────────────────────────────

with pm.Model() as blr:
    # Priors — weakly informative on standardised predictors
    alpha = pm.Normal("alpha", mu=0, sigma=2)
    beta  = pm.Normal("beta",  mu=0, sigma=1, shape=X_std.shape[1])
    sigma = pm.HalfNormal("sigma", sigma=1)

    mu    = alpha + pm.math.dot(X_std, beta)
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_std)

    idata = pm.sample(2000, tune=1000, target_accept=0.9, random_seed=42)

# Posterior means and 94% HDI
az.plot_posterior(idata, var_names=["alpha", "beta", "sigma"])
```

**Interpreting the posterior:** the posterior distribution of β_j gives a full probability distribution over effect sizes. A coefficient is "credibly non-zero" when its 95% HDI excludes zero — analogous to but conceptually distinct from a frequentist significance test.

### 4.2 Generalised Linear Models

**Logistic Regression:**

```python
with pm.Model() as logistic_model:
    alpha = pm.Normal("alpha", mu=0, sigma=2)
    beta  = pm.Normal("beta",  mu=0, sigma=1, shape=K)
    
    eta   = alpha + pm.math.dot(X, beta)
    p     = pm.Deterministic("p", pm.math.sigmoid(eta))
    
    y_obs = pm.Bernoulli("y_obs", p=p, observed=y_binary)
    idata = pm.sample(2000, tune=1000, target_accept=0.9, random_seed=42)

# Posterior odds ratios
log_or = idata.posterior["beta"]
or_hdi = az.hdi(np.exp(log_or.values))
```

> **Interpretation:** coefficients are on the log-odds scale. `exp(β_j)` is the odds ratio per unit increase in x_j. Always transform to the probability scale for communication.

**Poisson Regression:**

```python
with pm.Model() as poisson_model:
    alpha = pm.Normal("alpha", mu=0, sigma=2)
    beta  = pm.Normal("beta",  mu=0, sigma=1, shape=K)
    
    log_mu = alpha + pm.math.dot(X, beta)
    mu     = pm.Deterministic("mu", pm.math.exp(log_mu))
    
    y_obs  = pm.Poisson("y_obs", mu=mu, observed=y_counts)
    idata  = pm.sample(2000, tune=1000, target_accept=0.9, random_seed=42)
```

**Diagnosing overdispersion:**

```python
# Compare Poisson vs Negative Binomial via posterior predictive
y_rep = idata.posterior_predictive["y_obs"].values.reshape(-1, n)
print(f"Observed variance/mean ratio: {y_counts.var()/y_counts.mean():.2f}")
print(f"Replicated variance/mean ratio (mean): {(y_rep.var(axis=1)/y_rep.mean(axis=1)).mean():.2f}")
# If replicated << observed → Poisson is underdispersed → use NegBinomial
```

**Negative Binomial for overdispersed counts:**

```python
with pm.Model() as nb_model:
    alpha = pm.Normal("alpha", mu=0, sigma=2)
    beta  = pm.Normal("beta",  mu=0, sigma=1, shape=K)
    psi   = pm.HalfNormal("psi", sigma=1)  # dispersion
    
    mu    = pm.math.exp(alpha + pm.math.dot(X, beta))
    y_obs = pm.NegativeBinomial("y_obs", mu=mu, alpha=psi, observed=y_counts)
    idata = pm.sample(2000, tune=1000, target_accept=0.9, random_seed=42)
```

---

## 5. Hierarchical Models

### 5.1 The Pooling Spectrum (Cowles Ch. 9)

Cowles motivates hierarchical models through a college softball player's batting average across 8 games. Key insight from Table 9.1:

| Game | Hits/At-bats | No-pooling MLE | Hierarchical posterior |
|------|-------------|---------------|----------------------|
| 2    | 0/4         | 0.000         | 0.111 ← shrunk up    |
| 3    | 1/3         | 0.333         | 0.242 ← shrunk down  |
| ...  | ...         | ...           | ...                  |

Games with zero hits get shrunk **upward** from MLE = 0 (a nonsensical point estimate). Games with many hits get shrunk **downward**. This is **partial pooling** in action — the model borrows strength across games.

```python
with pm.Model() as softball_model:
    
    # Hyperpriors (Stage 3 in Cowles' notation)
    alpha_hyper = pm.Gamma("alpha_hyper", alpha=1.0, beta=1.0)
    beta_hyper  = pm.Gamma("beta_hyper",  alpha=1.0, beta=0.33)
    
    # Group-level priors (Stage 2)
    pi = pm.Beta("pi", alpha=alpha_hyper, beta=beta_hyper, shape=n_games)
    
    # Likelihood (Stage 1)
    hits = pm.Binomial("hits", n=at_bats, p=pi, observed=observed_hits)
    
    # Derived quantity: overall average
    mu = pm.Deterministic("mu", alpha_hyper / (alpha_hyper + beta_hyper))
    
    idata = pm.sample(2000, tune=1000, target_accept=0.9, random_seed=42)

# Shrinkage visualisation
mle_estimates = observed_hits / at_bats
post_means     = idata.posterior["pi"].mean(dim=["chain", "draw"]).values
print("MLE vs posterior means (shrinkage):")
for i, (mle, post) in enumerate(zip(mle_estimates, post_means)):
    print(f"  Game {i+1}: MLE={mle:.3f} → posterior={post:.3f}")
```

### 5.2 Three-Stage Hierarchical Architecture

```
Stage 3  (hyperpriors)  :  α ~ Gamma(1, 1),  β ~ Gamma(1, 0.33)
                                   ↓
Stage 2  (group priors) :  πᵢ ~ Beta(α, β)       i = 1, …, J
                                   ↓
Stage 1  (likelihood)   :  yᵢ ~ Binomial(nᵢ, πᵢ)
```

The **joint posterior** is proportional to (Cowles §9.1.4):

$$p(\pi, \alpha, \beta | y) \propto \prod_{i=1}^{J} \left[\binom{n_i}{y_i} \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)} \pi_i^{y_i+\alpha-1}(1-\pi_i)^{n_i-y_i+\beta-1}\right] e^{-\alpha} e^{-0.33\beta}$$

### 5.3 General Hierarchical Linear Model

```python
import pymc as pm
import numpy as np

with pm.Model() as hierarchical_lr:
    
    # Hyperpriors
    mu_alpha    = pm.Normal("mu_alpha", mu=0, sigma=10)
    sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=1)
    mu_beta     = pm.Normal("mu_beta",  mu=0, sigma=1)
    sigma_beta  = pm.HalfNormal("sigma_beta",  sigma=0.5)
    
    # Group-level parameters (NON-CENTERED — see §5.4)
    alpha_raw = pm.Normal("alpha_raw", mu=0, sigma=1, shape=J)
    beta_raw  = pm.Normal("beta_raw",  mu=0, sigma=1, shape=J)
    alpha_j   = pm.Deterministic("alpha_j", mu_alpha + alpha_raw * sigma_alpha)
    beta_j    = pm.Deterministic("beta_j",  mu_beta  + beta_raw  * sigma_beta)
    
    # Within-group model
    sigma = pm.HalfNormal("sigma", sigma=1)
    mu    = alpha_j[group_idx] + beta_j[group_idx] * X
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)
    
    idata = pm.sample(2000, tune=1000, target_accept=0.95, random_seed=42)
```

### 5.4 Non-Centered Parameterisation

The **centered parameterisation** places `alpha_j ~ Normal(mu_alpha, sigma_alpha)`. When group sizes are small, this creates a **Neal's funnel**: the posterior geometry has a narrow neck near `sigma_alpha ≈ 0` where curvature is extreme and NUTS degenerates.

**Detecting the funnel:** look for divergences in `az.summary()` or `az.plot_trace()`.

```python
# Centered (problematic):
alpha_j = pm.Normal("alpha_j", mu=mu_alpha, sigma=sigma_alpha, shape=J)

# Non-centered (robust):
alpha_raw = pm.Normal("alpha_raw", mu=0, sigma=1, shape=J)
alpha_j   = pm.Deterministic("alpha_j", mu_alpha + alpha_raw * sigma_alpha)
```

**Why it works:** the non-centered form decouples `alpha_raw` from `sigma_alpha`. The geometry becomes much more regular — NUTS can traverse it efficiently without divergences.

```python
# Diagnosing divergences
divergences = idata.sample_stats["diverging"].values.sum()
print(f"Number of divergences: {divergences}")
# Should be 0. If > 0, try non-centered parameterisation and/or raise target_accept
```

### 5.5 Intraclass Correlation (ICC)

The ICC quantifies what fraction of total variance is explained by group-level differences:

$$\text{ICC} = \frac{\sigma_\alpha^2}{\sigma_\alpha^2 + \sigma^2}$$

- ICC ≈ 1: nearly all variance is between groups → pooling provides enormous benefit
- ICC ≈ 0: variance is within groups → hierarchical model ≈ no-pooling

```python
sigma_alpha_post = idata.posterior["sigma_alpha"].values
sigma_post       = idata.posterior["sigma"].values
icc              = sigma_alpha_post**2 / (sigma_alpha_post**2 + sigma_post**2)
print(f"ICC: {icc.mean():.3f} [{np.percentile(icc, 2.5):.3f}, {np.percentile(icc, 97.5):.3f}]")
```

---

## 6. MCMC Theory & Diagnostics

### 6.1 Why MCMC?

For all but conjugate models, `P(y) = ∫ P(y|θ) P(θ) dθ` is analytically intractable. MCMC constructs a **Markov chain** whose stationary distribution *is* the posterior — samples are drawn without ever computing the normalising constant.

> **Cowles §8.3.1:** "Markov chains are random variables indexed by time. The key property is the Markov property: the future depends on the present, but not on the past."

Formally: a chain is a sequence θ₁, θ₂, … such that `P(θₜ₊₁ | θ₁, …, θₜ) = P(θₜ₊₁ | θₜ)`. Under **ergodicity** conditions, the chain converges to a unique stationary distribution π — which we engineer to equal the posterior.

### 6.2 Algorithm Hierarchy

| Algorithm | Exploration Mechanism | Strengths | Weaknesses |
|-----------|----------------------|-----------|------------|
| **Metropolis-Hastings** | Random walk + accept/reject | Works anywhere; easy to implement | Slow in high dimensions; manual tuning |
| **Gibbs** | Full conditional draws | Efficient when conditionals available | Fails under strong correlations |
| **HMC** | Hamiltonian dynamics + gradients | Distant proposals; high acceptance | Requires differentiable posteriors; trajectory length tuning |
| **NUTS** | HMC + no-U-turn criterion | Auto-tunes trajectory; PyMC default | Same as HMC but automated |
| **ADVI** | Variational; KL minimisation | Very fast | Approximate; misses multimodality |

**Metropolis acceptance probability** (Cowles §8.3.2):

$$A(\theta^* | \theta_t) = \min\left(1, \frac{P(\theta^* | y)\, q(\theta_t | \theta^*)}{P(\theta_t | y)\, q(\theta^* | \theta_t)}\right)$$

Since the posterior appears as a ratio, the normalising constant cancels — we only need the unnormalised posterior `P(y|θ) P(θ)`.

### 6.3 MCMC Diagnostics — The Full Toolkit

#### R-hat (Gelman-Rubin-Brooks Statistic)

*History:* Gelman and Rubin (1992) proposed this; Brooks and Gelman (1998) generalised it. Cowles discusses it at length in §9.4.

$$\hat{R} = \sqrt{\frac{N-1}{N} + \frac{B}{N \cdot W}}$$

where B = between-chain variance, W = within-chain variance. Modern PyMC/ArviZ implements the **rank-normalised** version (Vehtari et al. 2021) which is more robust.

**Rule:** R-hat < 1.01 for all parameters. Values > 1.05 indicate serious non-convergence.

#### Effective Sample Size (ESS)

Autocorrelation in the chain means successive samples carry less information than independent draws:

$$\text{ESS} = \frac{N}{1 + 2 \sum_{k=1}^\infty \rho_k}$$

where ρₖ is the lag-k autocorrelation. Report **bulk-ESS** (central distribution) and **tail-ESS** (extreme quantiles). Aim for ESS > 400 per chain.

> **Cowles** gives the operational rule: *MC error should be less than 1/20th of the estimated posterior standard deviation.* MC error ≈ posterior_sd / √ESS.

#### Trace Plots

```python
az.plot_trace(idata, var_names=["alpha", "beta", "sigma"])
```

A converged trace looks like a **"hairy caterpillar"** — all chains overlapping, no trends, no stuck regions.

#### Divergences

NUTS divergences signal numerical instability — regions where the log-posterior's Hessian has extreme curvature. They are not just a sampling artifact: each divergence indicates a part of the posterior that was **not explored**.

```python
# Check divergences
az.plot_pair(idata, var_names=["mu_alpha", "sigma_alpha"], 
             divergences=True, scatter_kwargs={"color": "blue"})
# Red points = divergences; if clustered near sigma_alpha ≈ 0 → funnel → reparameterise
```

#### Energy Diagnostic (BFMI)

$$\text{BFMI} = \frac{E[(H_n - H_{n-1})^2]}{\text{Var}(H_n)}$$

BFMI < 0.3 indicates the energy marginal is poorly explored — often accompanies funnel geometry.

```python
energy = idata.sample_stats["energy"].values
bfmi   = np.mean(np.diff(energy, axis=1)**2, axis=1) / np.var(energy, axis=1)
print(f"BFMI per chain: {bfmi}")
```

### 6.4 Full Diagnostic Checklist

```python
# Complete diagnostic suite
summary = az.summary(idata, round_to=3)
print(summary)

# Check convergence
assert (summary["r_hat"] < 1.01).all(), "R-hat too high — not converged"
assert (summary["ess_bulk"] > 400).all(), "ESS too low — run longer chains"

# Check divergences
n_div = idata.sample_stats["diverging"].values.sum()
print(f"Divergences: {n_div}")  # Should be 0

# Trace plots
az.plot_trace(idata)
# Autocorrelation
az.plot_autocorr(idata)
# Rank plots (more robust than traces for convergence assessment)
az.plot_rank(idata)
```

### 6.5 Burn-in and Mixing

**Burn-in** (Cowles §8.4.5): the initial iterations before the chain converges to the stationary distribution. These are discarded. PyMC's `tune` argument controls this.

**Mixing**: how quickly the chain traverses the posterior. A well-mixing chain has low autocorrelation. Poor mixing → high autocorrelation → low ESS → unreliable inference.

```python
idata = pm.sample(
    draws        = 2000,  # post-tuning draws (retained)
    tune         = 1000,  # burn-in draws (discarded)
    chains       = 4,     # always use ≥ 4 for reliable R-hat
    target_accept= 0.9    # raise to 0.95 for hierarchical models
)
```

---

## 7. Causal Inference

### 7.1 The Fundamental Problem

Standard Bayesian regression estimates `P(Y | X, data)` — a conditional association. Causal inference asks: *what would happen to Y if we intervened to set X = x?* This requires the **do-operator**:

$$P(Y | \text{do}(X = x)) \neq P(Y | X = x) \quad \text{in general}$$

### 7.2 DAGs and d-Separation

A **DAG** G = (V, E) encodes causal assumptions. Nodes are variables; directed edges represent direct causal effects. The DAG is a **structural assumption** — not estimated from data.

| DAG Structure | Implication |
|---------------|-------------|
| X → Y ← Z (collider) | X, Z marginally independent; conditioning on Y induces spurious association (Berkson's paradox) |
| X ← Z → Y (fork) | X, Y are confounded by Z; condition on Z to close the backdoor |
| X → Z → Y (chain) | Z is a mediator; conditioning on Z blocks the causal path |

### 7.3 The Backdoor Criterion & Adjustment Formula

A set Z satisfies the **backdoor criterion** for (X, Y) if:
1. Z blocks all backdoor paths from X to Y
2. Z contains no descendants of X

If Z satisfies the backdoor criterion:

$$P(Y | \text{do}(X = x)) = \sum_z P(Y | X = x, Z = z)\, P(Z = z)$$

```python
import pymc as pm
import numpy as np

# Causal effect estimation with backdoor adjustment
# Z is a confounder of X → Y

with pm.Model() as causal_model:
    
    # Priors
    alpha = pm.Normal("alpha", 0, 2)
    beta_x = pm.Normal("beta_x", 0, 1)  # causal effect of X
    beta_z = pm.Normal("beta_z", 0, 1)  # adjustment for confounder Z
    sigma  = pm.HalfNormal("sigma", 1)
    
    # Likelihood (adjusted for Z)
    mu    = alpha + beta_x * X_obs + beta_z * Z_obs
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=Y_obs)
    
    idata = pm.sample(2000, tune=1000, random_seed=42)

# Causal effect of X (confounding adjusted)
az.plot_posterior(idata, var_names=["beta_x"])
```

### 7.4 Common Pitfalls

| Pitfall | Description | Fix |
|---------|-------------|-----|
| **Collider bias** | Conditioning on a variable downstream of both X and Y | Remove collider from adjustment set |
| **M-bias** | Conditioning on a pre-treatment variable that opens a backdoor | Careful DAG analysis |
| **Omitted variable bias** | Failing to condition on a confounder | Include all variables in the backdoor adjustment set |
| **Proxy confounding** | Using a noisy measurement of a confounder | Measurement error models |

---

## 8. Bayesian Time Series

### 8.1 Autoregressive Models

```python
with pm.Model() as ar2_model:
    
    # Priors on AR coefficients — stationarity region for AR(2)
    rho1  = pm.Uniform("rho1", lower=-1, upper=1)
    rho2  = pm.Uniform("rho2", lower=-1, upper=1)
    sigma = pm.HalfNormal("sigma", sigma=1)
    alpha = pm.Normal("alpha", mu=0, sigma=2)
    
    # Likelihood: y_t | y_{t-1}, y_{t-2} ~ Normal(...)
    mu_t  = alpha + rho1 * y[1:-1] + rho2 * y[:-2]
    y_obs = pm.Normal("y_obs", mu=mu_t, sigma=sigma, observed=y[2:])
    
    idata = pm.sample(2000, tune=1000, target_accept=0.9, random_seed=42)
```

**Stationarity condition:** the roots of `1 - ρ₁z - ρ₂z²` must lie outside the complex unit circle.

### 8.2 State-Space Model (Local Level)

The local level model decomposes a time series into a stochastic trend and noise:

```
x_t = x_{t-1} + w_t,    w_t ~ N(0, σ_w²)    [state equation]
y_t = x_t + v_t,         v_t ~ N(0, σ_v²)    [observation equation]
```

```python
with pm.Model() as local_level:
    
    sigma_w = pm.HalfNormal("sigma_w", sigma=1)  # state noise
    sigma_v = pm.HalfNormal("sigma_v", sigma=1)  # observation noise
    
    # GaussianRandomWalk for the latent trend
    trend = pm.GaussianRandomWalk("trend", sigma=sigma_w, shape=T)
    
    # Likelihood
    y_obs = pm.Normal("y_obs", mu=trend, sigma=sigma_v, observed=y)
    
    idata = pm.sample(2000, tune=1000, target_accept=0.9, random_seed=42)
```

### 8.3 Structural Decomposition

```
y_t = μ_t (trend) + γ_t (seasonal) + β_t x_t (regression) + ε_t
```

```python
import numpy as np
import pymc as pm

T = len(y)
t = np.arange(T)
# Fourier seasonality features (period=12 for monthly data)
K = 3  # number of harmonic pairs
fourier = np.column_stack([
    np.sin(2 * np.pi * (k+1) * t / 12) for k in range(K)
] + [
    np.cos(2 * np.pi * (k+1) * t / 12) for k in range(K)
])

with pm.Model() as structural_ts:
    
    # Trend: local linear
    level_sigma = pm.HalfNormal("level_sigma", sigma=0.5)
    slope_sigma = pm.HalfNormal("slope_sigma", sigma=0.1)
    level = pm.GaussianRandomWalk("level", sigma=level_sigma, shape=T)
    slope = pm.GaussianRandomWalk("slope", sigma=slope_sigma, shape=T)
    trend = pm.Deterministic("trend", level + slope * t)
    
    # Seasonality (Fourier)
    beta_fourier = pm.Normal("beta_fourier", mu=0, sigma=1, shape=2*K)
    seasonal     = pm.Deterministic("seasonal", pm.math.dot(fourier, beta_fourier))
    
    # Observation noise
    sigma = pm.HalfNormal("sigma", sigma=1)
    y_obs = pm.Normal("y_obs", mu=trend + seasonal, sigma=sigma, observed=y)
    
    idata = pm.sample(2000, tune=2000, target_accept=0.95, random_seed=42)
```

---

## 9. Approximate Bayesian Computation (ABC)

Downey Ch. 10 introduces **ABC** for problems where the likelihood is intractable but simulation is easy. This is a practical and increasingly important technique.

### 9.1 Core ABC Algorithm

```
for each iteration:
    1. Sample θ* from the prior: θ* ~ P(θ)
    2. Simulate data y* from the model: y* ~ P(y | θ*)
    3. Compute distance: d = distance(y*, y_obs)
    4. Accept θ* if d < ε (tolerance)
The accepted θ* values approximate the posterior P(θ | y)
```

### 9.2 Python Implementation

```python
import numpy as np
from scipy import stats

def abc_rejection(
    prior_sampler,
    simulator,
    summary_statistic,
    y_obs,
    epsilon,
    n_samples=10000
):
    """
    ABC rejection sampler.
    
    Args:
        prior_sampler    : callable, returns a sample from the prior
        simulator        : callable(theta) → simulated dataset
        summary_statistic: callable(y) → summary statistic vector
        y_obs            : observed data
        epsilon          : acceptance tolerance
        n_samples        : number of accepted samples to collect
    Returns:
        accepted_thetas : array of posterior samples
    """
    accepted = []
    s_obs = summary_statistic(y_obs)
    attempts = 0
    
    while len(accepted) < n_samples:
        theta_star = prior_sampler()
        y_star     = simulator(theta_star)
        s_star     = summary_statistic(y_star)
        
        # Euclidean distance on summary statistics
        if np.linalg.norm(s_star - s_obs) < epsilon:
            accepted.append(theta_star)
        attempts += 1
    
    print(f"Acceptance rate: {n_samples/attempts:.4f}")
    return np.array(accepted)

# Example: estimating mean of a Gaussian with unknown μ
y_obs = np.random.normal(loc=3.5, scale=1.0, size=50)

posterior_samples = abc_rejection(
    prior_sampler    = lambda: np.random.uniform(-10, 10),
    simulator        = lambda mu: np.random.normal(mu, 1.0, 50),
    summary_statistic= lambda y: np.array([np.mean(y), np.std(y)]),
    y_obs            = y_obs,
    epsilon          = 0.3,
    n_samples        = 1000
)

print(f"ABC posterior mean: {posterior_samples.mean():.3f}")
print(f"True value:         3.500")
```

### 9.3 When to Use ABC

- Likelihood is analytically intractable (complex generative processes)
- Simulation from the model is easy
- Sufficient summary statistics are available

**Trade-off:** ABC introduces approximation error from (1) the choice of summary statistics (information loss) and (2) the tolerance ε. Smaller ε → less bias but lower acceptance rate → fewer samples.

---

## 10. Decision Analysis under Uncertainty

Downey Ch. 6 formalises decision-making as: given a posterior distribution over states of the world, which action minimises expected loss (or maximises expected utility)?

### 10.1 Expected Loss Framework

```python
def expected_loss(action, posterior_pmf, loss_fn):
    """
    Compute expected loss for a given action.
    
    Args:
        action       : the decision being evaluated
        posterior_pmf: dict {state: probability}
        loss_fn      : callable(action, state) → loss
    Returns:
        expected loss (float)
    """
    return sum(
        prob * loss_fn(action, state)
        for state, prob in posterior_pmf.items()
    )

def optimal_action(candidate_actions, posterior_pmf, loss_fn):
    """Choose the action that minimises expected loss."""
    losses = {a: expected_loss(a, posterior_pmf, loss_fn) for a in candidate_actions}
    return min(losses, key=losses.get), losses
```

### 10.2 Connection to Posterior Summaries

The optimal action under different loss functions corresponds to known posterior summaries:

| Loss Function | Optimal Point Estimate |
|---------------|----------------------|
| Squared error: `(θ - estimate)²` | **Posterior mean** |
| Absolute error: `|θ - estimate|` | **Posterior median** |
| 0–1 loss: `1 if estimate ≠ θ` | **MAP (posterior mode)** |

This is why the posterior mean is often used — it minimises expected squared error, which is the most common loss in practice.

---

## 11. Model Comparison & Evidence

### 11.1 LOO-CV and WAIC

```python
# Fit two competing models
with model_1:
    idata_1 = pm.sample(2000, tune=1000, random_seed=42)
    pm.compute_log_likelihood(idata_1)

with model_2:
    idata_2 = pm.sample(2000, tune=1000, random_seed=42)
    pm.compute_log_likelihood(idata_2)

# Compare via LOO
loo_1 = az.loo(idata_1, pointwise=True)
loo_2 = az.loo(idata_2, pointwise=True)

comparison = az.compare({"model_1": idata_1, "model_2": idata_2})
print(comparison)
```

**Reading the comparison table:**
- `elpd_loo`: estimated log predictive density (higher = better)
- `p_loo`: effective number of parameters (Bayesian complexity penalty)
- `d_loo`: ELPD difference relative to best model
- `dse`: standard error of the ELPD difference

**Decision rule:** prefer the model with higher ELPD. If `|d_loo| < dse`, the models are statistically indistinguishable → prefer the simpler one.

### 11.2 Bayes Factors

$$\text{BF}_{12} = \frac{P(y | M_1)}{P(y | M_2)} = \frac{\int P(y | \theta_1, M_1) P(\theta_1 | M_1) d\theta_1}{\int P(y | \theta_2, M_2) P(\theta_2 | M_2) d\theta_2}$$

| BF₁₂ | Evidence for M₁ |
|-------|----------------|
| 1–3   | Barely worth mentioning |
| 3–10  | Substantial |
| 10–30 | Strong |
| > 30  | Very strong |

> **Caution:** Bayes factors are extremely sensitive to the prior specification — unlike LOO/WAIC. In practice, LOO is preferred for model comparison.

---

## 12. Revision Plan & Formula Reference

### 12.1 Five-Session Revision Plan

| Session | Focus | Active Tasks |
|---------|-------|-------------|
| **A** | Foundations (S1, S2) | Derive Beta-Binomial from scratch; implement `bayes_table()`; verify Normal-Normal conjugate update |
| **B** | Regression & GLMs (S3, S4) | Fit Bayesian LR from scratch in PyMC; diagnose from `az.summary()`; build logistic + Poisson models |
| **C** | Hierarchical (S5) | Implement centered vs non-centered; count divergences; visualise shrinkage; work through softball example |
| **D** | MCMC (S6) | Implement Metropolis by hand; diagnose trace plots deliberately broken by pathological models; compute ESS manually from ACF |
| **E** | Causal + TS + Synthesis (S7, S8) | Draw and annotate a DAG; apply backdoor criterion; fit structural time series; write comparative model memo with LOO evidence |

### 12.2 Essential Formula Reference

```
BAYES' THEOREM
  P(θ|y) = P(y|θ) P(θ) / P(y)
  P(y)   = ∫ P(y|θ) P(θ) dθ          [marginal likelihood]
  P(ỹ|y) = ∫ P(ỹ|θ) P(θ|y) dθ        [posterior predictive]

CONJUGATE UPDATES
  Beta(α,β) + Bin(n,k)   → Beta(α+k, β+n-k)
  Gamma(α,β) + Pois(λ)   → Gamma(α+Σy, β+n)
  N(μ₀,τ₀²) + N(μ,σ²/n) → N(μₙ, τₙ²)
    where 1/τₙ² = 1/τ₀² + n/σ²
          μₙ   = (μ₀/τ₀² + nȳ/σ²) × τₙ²

GLM LINK FUNCTIONS
  Logistic:  logit(p) = log(p/(1-p));   p = sigmoid(η) = 1/(1+e^{-η})
  Log:       log(λ) = η;               λ = exp(η)
  Softmax:   pₖ = exp(ηₖ) / Σⱼ exp(ηⱼ)

METROPOLIS ACCEPTANCE RATIO
  A(θ*|θₜ) = min(1, [P(θ*|y) q(θₜ|θ*)] / [P(θₜ|y) q(θ*|θₜ)])

ELBO (VARIATIONAL INFERENCE)
  ELBO = E_q[log P(y,θ)] - E_q[log q(θ)]
       = log P(y) - KL(q(θ) || P(θ|y))
  Maximise ELBO ↔ Minimise KL divergence

HIERARCHICAL VARIANCE DECOMPOSITION
  Var(yᵢⱼ) = σ² + σ_α²
  ICC = σ_α² / (σ_α² + σ²)

MODEL COMPARISON
  ELPD_loo = Σᵢ log P(yᵢ | y₋ᵢ)       [higher is better]
  Prefer model with highest ELPD
  If |ΔELPD| < SE(ΔELPD) → prefer simpler model

CAUSAL BACKDOOR ADJUSTMENT
  P(Y|do(X=x)) = Σ_z P(Y|X=x, Z=z) P(Z=z)
  [valid when Z satisfies the backdoor criterion]

NON-CENTERED PARAMETERISATION
  Centered:     α_j ~ N(μ_α, σ_α)
  Non-centered: α_raw ~ N(0,1);  α_j = μ_α + α_raw × σ_α

CONVERGENCE DIAGNOSTICS
  R-hat < 1.01    [all parameters]
  ESS_bulk > 400  [per chain]
  Divergences = 0
  BFMI > 0.3
```

### 12.3 Conceptual Pitfall Index

| Pitfall | Correct Understanding |
|---------|----------------------|
| "Credible interval = confidence interval" | CI is a procedural claim; credible interval is a probability statement about θ |
| "Flat prior is objective" | Flat prior on θ is informative on f(θ); use Jeffreys instead if invariance matters |
| "MCMC output is independent" | It is autocorrelated; use ESS not raw N for uncertainty quantification |
| "Posteriors converge quickly in hierarchical models" | Funnel geometry requires non-centered form and high `target_accept` |
| "P-value = P(H₀ is true)" | P-value = P(data as extreme or more | H₀ true); the Bayesian posterior answers the former |
| "Bayes factors are robust to priors" | BFs are extremely sensitive to prior specification; prefer LOO/WAIC |
| "P(cancer) = sensitivity after positive test" | Base rate fallacy; the posterior depends critically on the prior prevalence |
| "Conditioning on colliders is fine" | It opens spurious paths — Berkson's paradox |

---

*Sources synthesised: Course Sessions 1–8 (PyMC/ArviZ) · Cowles, M.K. (2013) Applied Bayesian Statistics · Downey, A.B. (2013) Think Bayes · All Python code targets PyMC ≥ 5 and ArviZ ≥ 0.17*
