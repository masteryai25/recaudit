# Recommender audit — one page

**Project:** Measuring what recommender accuracy scores hide
**Repo:** https://github.com/masteryai25/recaudit
**Status:** Experiments complete; project report drafted for review.

---

## The question

Recommender systems are evaluated on accuracy: can the model predict what you
would rate a film, or put films you like near the top of your list. Every common
metric is computed per user and then averaged.

That averaging hides something. A model can score well while recommending
roughly the same items to everybody. Nothing in a standard evaluation would tell
you.

So: can you tell from a recommender's accuracy whether it actually personalises?

## What was done

Five models built and compared on MovieLens latest-small (~100K ratings), replicated on MovieLens 1M
(6,040 users, 1 million ratings):

- a bias-only baseline
- user-based collaborative filtering
- rating-optimised matrix factorisation (the textbook approach)
- the same model with its bias terms removed
- ALS, implemented from Hu, Koren & Volinsky (2008) and verified against the
  `implicit` library

Each was evaluated on accuracy, catalogue coverage, popularity bias, and how
much two random users' recommendation lists overlap — with a reference point
that makes the last one interpretable: what overlap do you get from ranking by
global popularity, without using any user-preference information?

Evaluation used a temporal split (hold out each user's *most recent* ratings)
rather than the usual random split, since recommending is a prediction about the
future.

## What was found

**1. The evaluation protocol inflates published results.** Temporal splits make
every model 3.6–5.1% worse than random splits on the same data, and change the
gaps between models.

**2. Accuracy barely predicts recommendation quality.** Matrix factorisation is
6% better than a bias-only baseline on RMSE and 21× better on precision@10.

**3. The textbook model is less differentiated than popularity ranking.**

| | shared items between two random users (of 10) |
|---|---|
| random recommendations | 0.01 |
| **rating-optimised matrix factorisation** | **5.72** |
| popularity-only ranking (no preference info) | 4.65 |
| ALS (ranking-optimised) | 0.35 |

Seven films fill half of every recommendation slot. One film reaches 94% of
users. Coverage: 2.3% of the catalogue. Replicates on MovieLens 1M, where the
gap is wider (6.49 vs 4.47).

**4. The cause is the bias terms, and removing them improves both properties.**

| | precision@10 | shared items | coverage |
|---|---|---|---|
| full model | 0.0150 | 5.72 | 2.3% |
| factors only, no biases | 0.0410 | 1.07 | 6.2% |

Accuracy nearly triples *and* personalisation improves by 81%. Not a trade-off.

**5. Not a trade-off generally either.** ALS has the best accuracy of all five
models and is also among the most personalised. If accuracy forced uniformity,
it couldn't.

## What was also tried, and didn't work

Reported as negative results rather than dropped:

- A feedback-loop simulation (does the system narrow what people see over time
  when retrained on its own output?). Concentration is high immediately and
  does **not** compound over 20 rounds. Three controls. The simulated users are
  the weak point.
- A popularity-aware modification to ALS. +0.0026 precision over 5 seeds
  (spread 0.0017) and no meaningful coverage gain.
- Diversity re-ranking improves variety *within* a list (+44% genres for <4%
  accuracy cost) but leaves catalogue coverage unchanged.

## The artifact

`recaudit` — point it at any recommender, get back coverage, popularity bias and
inter-user overlap alongside accuracy, with the popularity baseline for
comparison. Five lines to use. MIT licensed.

## Where it is weak, honestly

- Two datasets, both MovieLens, both films.
- The mechanism (item bias drives popularity bias) is **already known** and
  stated in prior work. What's new here is the measured consequence and the
  reference-point framing, not the mechanism.
- The metrics are all existing metrics. The packaging and the baseline are the
  contribution.
- The feedback simulation rests on invented user behaviour.

## What I'd like feedback on

1. Is the central claim stated too strongly, given the mechanism is known?
2. Is a popularity-ranking baseline for inter-user overlap a reasonable
   reference, or is there a standard one I should be using instead?
3. Is a third dataset outside films worth the time, or is two enough?
4. Anything obviously missing that an examiner would ask about?
