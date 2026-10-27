"""Costed, label-blind acquisition in a controlled stochastic-response replay.

One query returns a response vector from one fixed ensemble member. The bank has
no reference labels. Its noise is synthetic and independent of targets. Credits
simulate marginal acquisition cost; they are not measured GPU time or Alpamayo
calls. Uncertainty widths below are heuristics, not coverage guarantees.
"""
from dataclasses import dataclass

import numpy as np


METHODS = ("random", "diversity", "uncertainty", "corrected_fixed",
           "response_magnitude", "adaptive_corrected")


class ResponseBank:
    def __init__(self, means, features, known_features, noise_std, seed, budget, label_cost):
        means = np.asarray(means, dtype=float)
        if means.ndim != 3 or not np.isfinite(means).all():
            raise ValueError("Expected finite [members, candidates, response_dimensions]")
        if budget < 0 or label_cost < 1:
            raise ValueError("Invalid budget/cost")
        self._means = means.copy()
        self._features = np.asarray(features, dtype=float).copy()
        self.known_features = np.asarray(known_features, dtype=float).copy()
        self._noise = np.broadcast_to(np.asarray(noise_std, dtype=float), (means.shape[1],)).copy()
        if np.any(self._noise < 0) or not np.isfinite(self._noise).all():
            raise ValueError("Invalid noise")
        if self._features.shape[0] != means.shape[1]:
            raise ValueError("Feature/candidate mismatch")
        self.seed = int(seed)
        self.budget, self.label_cost = int(budget), int(label_cost)
        self.spent = 0
        self.counts = np.zeros(means.shape[:2], dtype=int)
        self.selected = []
        self.trace = []
        self._feature_cache = {}
        self.n_members, self.n_candidates, self.dimension = means.shape

    @property
    def remaining(self):
        return self.budget-self.spent

    def _charge(self, cost):
        if self.spent+cost > self.budget:
            raise ValueError("Acquisition exceeds budget")
        self.spent += cost

    def query(self, candidate, member):
        if not 0 <= candidate < self.n_candidates or not 0 <= member < self.n_members:
            raise IndexError("Invalid candidate/member")
        self._charge(1)
        draw = int(self.counts[member, candidate])
        # Same indexed random stream for all strategies; query order cannot
        # change which draw is returned for a given candidate/member/count.
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, candidate, member, draw]))
        response = self._means[member, candidate]+self._noise[candidate]*rng.normal(size=self.dimension)
        self.counts[member, candidate] += 1
        self.trace.append({"operation": "response", "candidate": int(candidate),
                           "member": int(member), "draw": draw, "spent": self.spent})
        return response

    def feature(self, candidate):
        if not 0 <= candidate < self.n_candidates:
            raise IndexError("Invalid candidate")
        if candidate not in self._feature_cache:
            self._charge(1)
            self._feature_cache[candidate] = self._features[candidate].copy()
            self.trace.append({"operation": "feature", "candidate": int(candidate), "spent": self.spent})
        return self._feature_cache[candidate].copy()

    def label(self, candidate):
        if not 0 <= candidate < self.n_candidates:
            raise IndexError("Invalid candidate")
        if candidate in self.selected:
            raise ValueError("Duplicate label request")
        self._charge(self.label_cost)
        self.selected.append(int(candidate))
        # Label values are deliberately absent from this class and selector.
        # The training runner retrieves references only after selection ends.
        self.trace.append({"operation": "label", "candidate": int(candidate), "spent": self.spent})


@dataclass
class Moments:
    count: np.ndarray
    total: np.ndarray
    squared_total: np.ndarray

    @classmethod
    def empty(cls, members, candidates, dimension):
        shape = (members, candidates, dimension)
        return cls(np.zeros(shape[:2], dtype=int), np.zeros(shape), np.zeros(shape))

    def add(self, candidate, member, value):
        self.count[member, candidate] += 1
        self.total[member, candidate] += value
        self.squared_total[member, candidate] += value*value

    def estimates(self):
        if np.any(self.count < 2):
            raise ValueError("At least two draws per member/candidate are required")
        n = self.count[..., None]
        means = self.total/n
        variance = np.maximum((self.squared_total-self.total**2/n)/(n-1), 0.)
        k, _, dimension = means.shape
        centered = means-means.mean(axis=0, keepdims=True)
        raw = (centered**2).mean(axis=(0, 2))
        mean_variance = variance/n
        correction = ((k-1)/k**2)*mean_variance.sum(axis=0).mean(axis=-1)
        corrected = raw-correction  # do not clip unbiased score estimates
        gradient = 2*centered/(k*dimension)
        linear_variance = (gradient**2*mean_variance).sum(axis=(0, 2))
        second_order = 2*(mean_variance**2).sum(axis=(0, 2))/(k*k*dimension*dimension)
        width = 2*np.sqrt(linear_variance+second_order)
        resample_priority = (variance/(n*(n+1))).mean(axis=-1)
        magnitude = (means.mean(axis=0)**2).mean(axis=-1)
        return {"raw": raw, "corrected": corrected, "width": width,
                "magnitude": magnitude, "resample_priority": resample_priority}


def _rank(scores, eligible, tie_break):
    return eligible[np.argsort(-(scores[eligible]+tie_break[eligible]), kind="stable")]


def acquire(bank, method):
    if method not in METHODS:
        raise ValueError(method)
    rng = np.random.default_rng(bank.seed+91000)
    tie_break = rng.uniform(0, 1e-12, bank.n_candidates)
    if method == "random":
        for index in rng.permutation(bank.n_candidates):
            if bank.remaining < bank.label_cost:
                break
            bank.label(int(index))
        return _result(bank, method, 0)
    if method == "diversity":
        if bank.remaining < bank.n_candidates+bank.label_cost:
            raise ValueError("Budget cannot pay for diversity features and one label")
        features = np.stack([bank.feature(i) for i in range(bank.n_candidates)])
        known = bank.known_features
        center, scale = known.mean(0), np.maximum(known.std(0), .01)
        features, known = (features-center)/scale, (known-center)/scale
        distance = ((features[:, None]-known[None, :])**2).mean(-1).min(1)
        while bank.remaining >= bank.label_cost and len(bank.selected) < bank.n_candidates:
            eligible = np.setdiff1d(np.arange(bank.n_candidates), bank.selected)
            index = int(_rank(distance, eligible, tie_break)[0])
            bank.label(index)
            distance = np.minimum(distance, ((features-features[index])**2).mean(-1))
        return _result(bank, method, 0)
    pilot_cost = 2*bank.n_members*bank.n_candidates
    if bank.remaining < pilot_cost+bank.label_cost:
        raise ValueError("Budget cannot pay for initial response probes and one label")
    moments = Moments.empty(bank.n_members, bank.n_candidates, bank.dimension)
    for i in range(bank.n_candidates):
        for member in range(bank.n_members):
            for _ in range(2):
                moments.add(i, member, bank.query(i, member))
    extra_cap = int(.25*bank.remaining)
    extra_queries = 0
    while bank.remaining >= bank.label_cost and len(bank.selected) < bank.n_candidates:
        estimate = moments.estimates()
        key = {"uncertainty": "raw", "corrected_fixed": "corrected",
               "response_magnitude": "magnitude", "adaptive_corrected": "corrected"}[method]
        eligible = np.setdiff1d(np.arange(bank.n_candidates), bank.selected)
        ranked = _rank(estimate[key], eligible, tie_break)
        best = int(ranked[0])
        if method == "adaptive_corrected" and len(ranked) > 1:
            second = int(ranked[1])
            overlaps = (estimate["corrected"][best]-estimate["width"][best] <=
                        estimate["corrected"][second]+estimate["width"][second])
            imprecise = estimate["width"][best] > .5*max(abs(estimate["corrected"][best]), 1e-6)
            can_repeat = extra_queries < extra_cap and bank.remaining >= bank.label_cost+1
            if overlaps and imprecise and can_repeat:
                frontier = ranked[:min(8, len(ranked))]
                index = int(frontier[np.argmax(estimate["width"][frontier])])
                member = int(np.argmax(estimate["resample_priority"][:, index]))
                moments.add(index, member, bank.query(index, member))
                extra_queries += 1
                continue
        bank.label(best)
    return _result(bank, method, extra_queries)


def _result(bank, method, extra_queries):
    return {"method": method, "selected": bank.selected.copy(), "spent": bank.spent,
            "budget": bank.budget, "label_cost": bank.label_cost,
            "n_labels": len(bank.selected), "response_queries": int(bank.counts.sum()),
            "feature_queries": len(bank._feature_cache), "extra_response_queries": extra_queries,
            "trace": bank.trace.copy()}
