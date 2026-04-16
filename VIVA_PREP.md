# Viva Prep - FYP

Use this as a speaking aid, not a script. The aim is simple: answer clearly, keep the numbers straight, and stay inside what the report actually shows. A good viva answer usually has three parts: the claim, the reason, and one piece of evidence.

---

## 1. Short opening answer

If you get asked to describe the project in your own words:

> I built a three-layer predictive-maintenance system for turbofan engines using the NASA C-MAPSS benchmark. The first layer is a five-model LSTM ensemble that predicts remaining useful life and gives an uncertainty signal from model disagreement. The second layer is a DQN agent that uses both the prediction and the uncertainty to decide whether to keep running or schedule maintenance. The third layer is a deterministic safety supervisor that can override unsafe decisions in low-RUL conditions. The main result is that once the policy can see uncertainty, it stays much more robust under sensor noise and needs far less rescue from the supervisor.

That is enough for the first 20 to 30 seconds. Then expand only if asked.

If they want a one-sentence version:

> The project asks a simple question: if an RL maintenance policy can see when its prediction is unreliable, does it make better decisions under noisy conditions, and the answer in my experiments was yes.

---

## 2. Two-minute walkthrough

1. Problem:
   Predictive maintenance is really a timing problem. If you act too late, you risk failure. If you act too early, you waste useful life. The hard part is that the prediction itself becomes less reliable when the sensors are noisy.

2. Data:
   NASA C-MAPSS, FD001 and FD002 combined. That gives 360 engines, 74,390 rows, and 63,590 sliding windows after preprocessing. Each cycle has 24 input features.

3. Layer 1:
   Five two-layer LSTMs, hidden size 100, dropout 0.2. Each model trains on a different random 80% subset without replacement. At inference, I use the ensemble mean as the RUL prediction and the ensemble standard deviation as the uncertainty signal.

4. Layer 2:
   A custom Gymnasium environment with a DQN agent. The observation is `[mu_RUL, sigma_now, sigma_rolling, sensor_health]`. The action space is binary: `WAIT` or `MAINTAIN`, which keeps the decision problem clean and easy to analyse.

5. Layer 3:
   A deterministic safety supervisor. Tier 1 forces maintenance below roughly 5 cycles remaining. Tier 2 forces maintenance below roughly 15 cycles remaining, but only when rolling uncertainty is low enough that the low-RUL prediction is trusted.

6. Training:
   The DQN trains for 1.5 million timesteps. Noise is injected into 70% of training episodes, with sigma sampled from 0.02 to 0.20. Also, 60% of episodes start near end-of-life so the agent sees enough real decision points.

7. Evaluation:
   Two primary experiments were run across eight noise levels with 500 episodes per condition for each agent. That is 8,000 episodes per primary experiment. A further safety-contribution analysis was derived from the full-system results.

8. Main finding:
   Without the supervisor, the uncertainty-aware agent stays positive at every tested noise level, while the blind baseline collapses into negative reward under heavy noise. With the full three-layer system active, both agents are kept near zero failure, but the uncertainty-aware agent does much more of the work itself.

---

## 3. Numbers to remember

These are the numbers worth keeping in your head.

| Quantity | Value |
|---|---:|
| Engines | 360 |
| Dataset rows | 74,390 |
| Sliding windows | 63,590 |
| Features per cycle | 24 |
| Window size | 30 |
| RUL cap | 125 |
| Ensemble size | 5 |
| LSTM hidden size | 100 |
| DQN network | [256, 256, 128] |
| DQN training | 1.5M timesteps |
| Replay buffer | 150,000 |
| Discount factor | 0.99 |
| Noise probability | 70% of episodes |
| Noise range | sigma in [0.02, 0.20] |
| End-of-life bias | 60% of episodes start in last 50 cycles |
| Crash penalty | -500 |
| Jackpot reward | +500 |
| Tier 1 supervisor | mean RUL < 0.04 |
| Tier 2 supervisor | mean RUL < 0.12 and rolling sigma < 0.55 |
| Clean-data autonomy | UA 97.4%, Blind 59.8% |
| Clean autonomous jackpots | UA 353/500, Blind 214/500 |
| Average annual saving | GBP 556,950 for 50 engines |
| Calibration | PICP 0.61 at nominal 0.95 |

One strong headline from Experiment 1:

- At the heaviest tested noise level (`sigma = 0.175`), the uncertainty-aware agent still scores `135.6`, while the blind agent falls to `-112.8`.

One strong headline from the full system:

- On clean data, the uncertainty-aware agent needs only `13` supervisor overrides, while the blind agent needs `201`.

---

## 4. Questions you are likely to get

### Why LSTM and not Transformer?

Short answer:

> Because the sequences are short, the compute budget is limited, and LSTM ensembles are a well-established baseline on C-MAPSS. A Transformer would have cost more to train and ensemble, without clearly strengthening the main project question.

### Why DQN and not PPO?

> The action space is binary and discrete, so DQN is a natural fit. It is also off-policy and sample-efficient, which matters because every environment step runs the ensemble. My appendix PPO comparison showed PPO could chase jackpots, but it did so with a materially higher failure rate.

### What is actually novel here?

> It is not a new algorithm. The contribution is the way the pieces are combined: ensemble disagreement is treated as a state variable for the RL policy, the policy is trained under noisy conditions, and the safety supervisor also uses uncertainty when deciding whether to intervene.

### Is the supervisor doing all the work?

> No, and that is exactly why I report autonomy and autonomous jackpots. On clean data the uncertainty-aware agent achieves 97.4% autonomy and 353 autonomous jackpots out of 500. The blind agent reaches 59.8% autonomy and 214 autonomous jackpots, so much more of its apparent performance comes from rescue by the supervisor.

### Are the uncertainty estimates calibrated?

> Only partially. The ensemble is overconfident as an interval estimator: PICP is 0.61 at nominal 0.95. That is a limitation. But the DQN uses sigma as a relative reliability signal rather than as a calibrated probability, so the policy result still stands. For deployment I would add a recalibration step.

### Why binary actions?

> Because the project question was whether uncertainty improves timing decisions. A binary action space keeps that comparison clean and makes the reward structure easier to interpret. It does limit realism, and I say that directly in the report.

### How does this generalise beyond C-MAPSS?

> The integration pattern is the part that transfers. Layer 1 would need retraining on a new asset class, but Layers 2 and 3 operate on abstract health and uncertainty signals rather than on turbofan-specific raw channels.

### Why not just use a fixed threshold policy?

> I did compare against threshold-style logic, and the issue is that fixed thresholds cannot adapt to changing confidence. The whole point of the uncertainty-aware agent is that the same predicted RUL should not always produce the same action if the reliability of that prediction has changed.

### What is the biggest limitation?

> The biggest limitation is external validity. C-MAPSS is a strong benchmark, but it is still a simulation. So the architecture is well supported in a controlled setting, but deployment claims would need real operational data, calibration work, and certification-grade assurance.

### What would you do first if you had more time?

> I would validate the framework on a harder setting before adding more complexity: either FD003 and FD004 for multi-fault conditions, or a real industrial dataset. That would tell me more than adding another model variant in isolation.

### What was the hardest technical problem?

> Getting the RL policy to learn the right timing behaviour rather than a trivial policy. Early on it would either wait too long or exploit the reward badly. The combination of end-of-life episode biasing, stronger crash penalties, and noise-aware training is what made the policy behave properly.

---

## 5. Good concise phrases

These are safe phrases to reuse in discussion:

- "The contribution is the integration, not a new learning algorithm."
- "The uncertainty signal is used as a relative reliability cue."
- "The fair comparison is the no-supervisor experiment."
- "Autonomy matters because a safety layer can otherwise hide a weak policy."
- "The system is a research prototype, not a deployment-ready product."
- "I would separate what the experiments show from what deployment would require."
- "The strongest evidence is the controlled ablation under increasing noise."

---

## 6. Claims to avoid

Do not say:

- "It guarantees safety."
- "It is fully calibrated."
- "It is ready for deployment."
- "It is the first system ever to do this."
- "The supervisor proves the policy is safe."

Use instead:

- "It reduced failures to near zero in the tested conditions."
- "The ensemble is overconfident as an interval estimator."
- "Deployment would require real-world validation and certification work."
- "The literature gap is in the way these components are combined."

---

## 7. Sound natural under pressure

- Start with the answer, not the backstory.
- Keep most answers to one claim, one reason, one number.
- If you do not remember an exact number, give the pattern first and then the approximate scale.
- Do not say "basically", "kind of", or "sort of" when you already know the point.
- If they challenge a limitation, agree directly and show that you already accounted for it in the report.
- Stop once the point is made. Let them ask the follow-up.

---

## 8. Recommended demo sequence

If you are asked to show the prototype, keep it short.

1. Engine 134, no added noise:
   Show the clean run. This is the easiest way to explain the three layers without distraction.

2. Engine 200, moderate noise:
   Turn on sensor noise and show uncertainty rising. Explain that the policy is now conditioning on a less reliable prediction.

3. Side-by-side mode:
   Compare uncertainty-aware vs blind. The point is not that the curves look different; the point is that the blind policy cannot respond to uncertainty because it never sees it.

4. Safety intervention case:
   Use Engine 50 or another case where the override fires, and explain that the supervisor is a backstop, not the main decision-maker.

---

## 9. Last checks before the viva

- Read the abstract, Chapter 5, Chapter 7, and Appendix H again.
- Be able to explain the four observation features without hesitating.
- Be able to quote the calibration limitation without sounding defensive.
- Run the dashboard once before the viva and step through at least one clean and one noisy case.
- Keep your answer lengths under control. Most questions can be answered in 30 to 60 seconds.
- Be ready to explain one technical decision, one limitation, and one future-work choice without looking down.

---

## 10. Final reminder

The safest posture is direct and evidence-based. You do not need to oversell the work. The report already has enough: a complete pipeline, controlled ablations, statistics, a live demo, and a clear limitations section. Sound like the person who built it, not the person who memorised a brochure.
