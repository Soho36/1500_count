Short answer: this says “almost all leftover profit happens only after MAE ≈ 1R”. That’s a strong result.

What your numbers mean
1️⃣ count = 415

Only 415 trades (out of ~8k) both:

reached ≥1R (MFE ≥ R)

and left profit on the table

➡️ This is already a small, selective subset.

2️⃣ median_MAE_R = 1.0

This is the key.

Interpretation:
For trades where you could have made more, price first went almost fully to your stop.

That means:

Market threatened your trade before giving more upside

Your exit is happening after significant adverse pressure

Counterpoint:
If MAE had been 0.2–0.4R, you’d have room to improve exits.
You don’t.

3️⃣ median_Left_R ≈ 1.26, p75 ≈ 2.5, mean ≈ 2.95

Yes, there is upside in theory.

But:

It comes after full-risk heat

It is fat-tailed (mean >> median)

➡️ This is exactly where fixed TPs look good in hindsight and fail forward.

4️⃣ MAE bins result
(0.0, 0.25]    NaN
(0.25, 0.5]    NaN
(0.5, 0.75]    NaN
(0.75, 1.0]    0.0


This is decisive.