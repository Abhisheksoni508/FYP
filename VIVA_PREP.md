# Viva Prep — FYP

> Goal: walk in able to answer any reasonable question in 60 seconds, in your own words, without hedging.

---

## 1. The 30-second elevator pitch (memorise verbatim)

> Most predictive maintenance systems treat their RUL predictions as if they were ground truth, even when the sensor data feeding the model is degraded. My project builds a three-layer system where the model knows when it does not know: a deep ensemble produces both a prediction and an uncertainty estimate, a reinforcement learning agent uses that uncertainty as part of its observation when deciding whether to wait or maintain, and a deterministic safety supervisor sits on top as a hard backstop. The uncertainty-aware agent stays positive at every noise level I tested, while a baseline blind to uncertainty collapses to negative reward and crashes half its engines.

---

## 2. The 2-minute walkthrough

1. **Problem.** Predictive maintenance on aircraft engines. Wait too long → crash. Maintain too early → wasted life. Real sensors degrade, so the prediction itself is sometimes unreliable, but standard systems do not know this.
2. **Data.** NASA C-MAPSS turbofan dataset, FD001+FD002 combined — 360 engines, ~46k sequences, 24 features per cycle, RUL capped at 125.
3. **Layer 1 — LSTM Deep Ensemble.** Five 2-layer LSTMs (hidden 100, dropout 0.2), each trained on a different bootstrap 80% subset with different seeds. At inference, mean of the 5 predictions = $\mu_{RUL}$, standard deviation = $\sigma$ = epistemic uncertainty.
4. **Layer 2 — DQN Agent.** Custom Gymnasium environment. Observation = $[\mu_{RUL},\ \sigma_{now},\ \sigma_{rolling},\ \text{trend}]$. Actions = WAIT or MAINTAIN. Trained for 1.5M timesteps with noise injected on 70% of episodes (uniform $\sigma \in [0.02, 0.20]$) and end-of-life biasing on 60% of episodes.
5. **Layer 3 — Safety Supervisor.** Two-tier deterministic override. Tier 1 (hard): RUL < 5 cycles → force MAINTAIN regardless of $\sigma$. Tier 2 (confidence-gated): RUL < 15 cycles AND rolling $\sigma$ < 0.55 → force MAINTAIN. The $\sigma$ gate is what makes the supervisor stay out of the way when the prediction is genuinely uncertain.
6. **Reward.** Symmetric ±500. Jackpot +500 for maintaining within last 20 cycles, crash −500 for waiting until RUL=0, smaller shaping in between. Includes uncertainty-urgency penalty for waiting under high $\sigma$ near failure.
7. **Ablation.** 3 experiments × 2 agents × 8 noise levels × 500 episodes = 24,000 episodes. Welch's $t$-test, Cohen's $d$.
8. **Key result.** UA agent wins the fair comparison (no supervisor) at every noise level. Under the full system both agents reach ≈0% failure, but UA does it autonomously (97% clean autonomy, 70% autonomous jackpots) while blind needs the supervisor to rescue it 43% of the time.
9. **Cost.** UA saves £556k/year on a 50-engine fleet under the calibrated cost model.
10. **Honest limitations.** Ensemble is mildly overconfident (PICP 0.60 at nominal 0.95), simulated data only, binary action space, single primary algorithm.

---

## 3. Numbers you MUST be able to quote

| Quantity | Value | Where it comes from |
|---|---|---|
| Engines | 360 (FD001 100 + FD002 260) | Preprocessing |
| Sequences | ~46,000 | After windowing |
| Window size | 30 cycles | `WINDOW_SIZE` |
| Features per cycle | 24 (21 sensors + 3 op settings) | C-MAPSS spec |
| RUL cap | 125 cycles | Heimes 2008 convention |
| Ensemble size | 5 | Compute-bounded |
| LSTM hidden | 100, 2 layers, dropout 0.2 | Pilot tuning |
| DQN architecture | [256, 256, 128] MLP | Pilot tuning |
| Total DQN timesteps | 1.5M | `RL_TIMESTEPS` |
| Replay buffer | 150,000 | `RL_BUFFER` |
| $\gamma$ (discount) | 0.99 | RL standard |
| $\epsilon$-greedy | 1.0 → 0.03 over 40% of training | `RL_EXPLORE_FRAC` |
| Noise probability | 70% of episodes | `NOISE_PROB` |
| Noise range | $\sigma \in [0.02, 0.20]$ | Domain randomisation |
| EOL bias | 60% of episodes start in last 50 cycles | `EOL_EPISODE_PROB` |
| Crash penalty | −500 | `CRASH_PENALTY` |
| Jackpot reward | +500 | Reward table |
| Tier 1 supervisor | RUL < 0.04 (~5 cycles), unconditional | `HARD_CRITICAL_RUL_NORM` |
| Tier 2 supervisor | RUL < 0.12 (~15 cycles) AND $\sigma$ < 0.55 | `CRITICAL_RUL_NORM` |
| **Headline result Exp1** | UA +136, Blind −113 at $\sigma$=0.15 | Table 5.1 |
| **Peak Cohen's d** | 0.65 (medium) at $\sigma$=0.175 | Table 5.1 |
| **UA clean autonomy** | 97.4% | Exp2 |
| **Blind clean autonomy** | 59.8% | Exp2 |
| **UA autonomous jackpots** | 70.6% (353/500) | Exp3 |
| **Blind autonomous jackpots** | 42.8% (214/500) | Exp3 |
| **Annual saving** | £556,950 (50 engines) | Cost analysis |
| **PICP at nominal 0.95** | 0.60 (overconfident) | Appendix H.1 |
| Validation RMSE | 12–15 cycles | Comparable to published baselines |

If you remember nothing else from this list: **+136 vs −113**, **70% vs 43% autonomous jackpots**, **£556k/year**, **PICP 0.60 at 0.95**.

---

## 4. The "Why?" questions Yasmine WILL ask

These map directly onto the points in her feedback. Have an answer for each.

### Q1. Why LSTM and not Transformer or CNN?
> Three reasons. First, fit to the data: 30-cycle windows of 24 features are short multivariate sequences, which is exactly what LSTMs were designed for. Transformers shine on longer sequences and CNNs are weaker on asymmetric degradation curves. Second, ensemble cost: my LSTM ensemble trains in about 25 minutes on a single consumer GPU, a Transformer ensemble would have been two to three hours from pilot runs and would have eaten my RL experiment budget. Third, uncertainty quantification maturity: deep ensembles are well characterised for LSTMs but Transformer ensemble calibration on small datasets is less well understood. So LSTM was the pragmatic choice, not a claim that it's intrinsically better. Transformer is in my future work list.

### Q2. Why DQN and not PPO or SAC?
> Four reasons. First, action-space fit: my action is binary discrete, which is the natural home of value-based methods. PPO and SAC target continuous or high-dimensional actions and offer no structural benefit here. Second, sample efficiency: every environment step runs five LSTM forward passes, which is expensive, so experience replay matters. PPO would discard data after each update and slow training under the same hardware budget. Third, stability: I deliberately used vanilla DQN rather than Double or Duelling because I needed a clean reference against which I could isolate the uncertainty contribution. Swapping in a stronger variant and the new observation together would have muddied the ablation. Fourth, my supplementary PPO run in Appendix B.4 actually shows DQN winning on risk-adjusted reward — PPO hit 18% failure rate on clean data versus DQN's 0%.

### Q3. Where did the idea of feeding $\sigma$ into the observation come from?
> Two threads. First, reading the C-MAPSS RL literature I noticed that almost no one propagated uncertainty into the agent — either they used point predictions or they used uncertainty only for post-hoc filtering. Second, the deep ensembles paper by Lakshminarayanan in 2017 showed that ensemble disagreement is a meaningful signal but most downstream uses are passive. The combination of those two observations gave me the design: treat $\sigma$ as a first-class state variable rather than a confidence score on the side. The novelty is in the integration, not in the building blocks.

### Q4. How is your work novel? (Be careful — frame it cautiously)
> I'm not claiming a new algorithm. Each building block — LSTM prediction, deep ensembles, DQN, shielding — is well established. My contribution is integration: taking ensemble disagreement as a first-class state variable in the RL observation, training under noise so the agent learns what high $\sigma$ means operationally, and combining that with a $\sigma$-gated safety supervisor that uses uncertainty to decide when to defer to the agent. To my knowledge that combination isn't widely reported in the predictive maintenance literature, but I frame it as a simplified, integrated implementation of state-of-the-art ideas, not as something fundamentally new.

### Q5. Why binary actions? Doesn't that limit real-world applicability?
> Yes, and I treat it as a deliberate trade-off, not an oversight. Three reasons I went binary. First, it keeps the reward signal clean — every episode ends in a clearly labelled outcome (jackpot, safe, wasteful, or crash), so I can attribute reward gaps directly to the policy. Second, it isolates the uncertainty contribution: any difference between UA and blind is attributable to $\sigma$, not to action-space handling. Third, it cuts state-action space dramatically, which mattered for training under my compute budget. The cost is real — graduated responses (inspect, partial repair, full overhaul) and fleet-level coordination are out of scope. I list this as a top-line limitation in Chapter 5 and as future work in Chapter 7.

### Q6. The safety supervisor — isn't it doing all the work?
> That's the question I confronted explicitly in Experiment 3. For the UA agent, the supervisor fires on about 7 out of 500 episodes on clean data — that's 1.4% — and the agent produces 353 autonomous jackpots before the supervisor has any chance to intervene. The RL policy is doing the work; the supervisor is a backstop. For the blind agent the pattern inverts: the supervisor fires on about 215 episodes — 43% — and the agent only produces 214 autonomous jackpots. The blind agent is essentially passive and most of its apparent safety comes from the shield. So under UA, system success is driven by the policy. Under blind, it's driven by the safety net. That asymmetry is the cleanest empirical evidence I have that the uncertainty-aware training actually changes what the agent learns.

### Q7. Are your uncertainty estimates actually calibrated? (THIS WILL COME UP)
> Honestly, only partially. Appendix H.1 shows the PICP analysis: at nominal 95% coverage the empirical coverage is only 60%, so the ensemble's predictive intervals are systematically too narrow — it's mildly overconfident. This is a known limitation of small deep ensembles. Now, the important thing is that my DQN doesn't consume $\sigma$ as a calibrated probability — it consumes it as a relative signal. The decision boundary analysis in Appendix H.2 confirms the agent learned the relative pattern: high $\sigma$ → maintain earlier. So the trained policy is not invalidated by the calibration result. But for a production deployment you'd want to recalibrate using temperature scaling or isotonic regression before exposing $\sigma$ as a probability to anything downstream. I added that to the limitations section.

### Q8. Why 5 ensemble members and not 10 or 20?
> Compute. Each member adds an LSTM forward pass to every environment step during RL training, and I'm running 1.5M timesteps. The marginal cost of going from 5 to 10 was significant for training time and would have squeezed the noise-augmentation experiments. Lakshminarayanan reports that calibration improves with ensemble size, so a larger ensemble would probably narrow the PICP gap I just mentioned. I list this as a limitation.

### Q9. Why noise on 70% of episodes and not 100%?
> Two reasons. First, I wanted the agent to see clean operating conditions during training because clean conditions are still part of the deployment distribution — if I trained only on noise the agent would be over-conservative on clean data and lose practical utility. Second, 70% gives enough exposure to high-$\sigma$ states that the agent actually learns to condition on them — I tried lower fractions in pilot runs and the policy did not separate from blind, which was the empirical signal that 70% was the working point.

### Q10. Why end-of-life biasing?
> Without it, only about 13% of random episode starts land in the critical decision zone (last 50 cycles). The agent would spend 87% of training in the healthy phase where no decision matters, and the gradient signal would be dominated by trivial WAIT rewards. With 60% EOL bias, the agent encounters the consequential states often enough to learn the timing structure. It's a curriculum learning trick.

### Q11. Why the symmetric ±500 reward?
> The +500 jackpot has to dominate the discounted sum of WAIT rewards or the agent learns to wait forever and accept the small but consistent reward. The −500 crash penalty has to be symmetric so that failing to act is exactly as costly as acting optimally is beneficial — otherwise the agent develops asymmetric risk preferences. I tried smaller penalties in pilot runs and the agent learned to accept occasional crashes because the discounted WAIT stream outweighed the rare large penalty.

### Q12. What did you actually code yourself?
> The Gymnasium environment in `src/gym_env.py` (276 lines) is mine — observation construction, reward shaping, the noise injection scheme, the safety supervisor logic, end-of-life initialisation. The training scripts and the experiment harness are mine. The dashboard (`dashboard.py`, 883 lines) is mine. I used PyTorch for the LSTM building blocks and Stable-Baselines3 for the DQN implementation rather than reimplementing those from scratch.

### Q13. What would you do differently if you started again?
> Three things. First, I'd use a larger ensemble (10–20 members) for better calibration and add a temperature scaling step. Second, I'd run a proper PPO and SAC comparison from day one rather than as a late appendix addition. Third, I'd explore a graduated action space — even three actions (wait / inspect / maintain) — to test whether the uncertainty advantage scales with action richness.

### Q14. How does this generalise beyond C-MAPSS?
> The architecture is dataset-agnostic. Layer 1 needs retraining per asset class because LSTM weights are domain-specific. Layers 2 and 3 are reusable: they operate on abstract health and uncertainty signals, not on raw sensor channels. So the integration pattern transfers; the prediction model does not. I'd want to test it on at least one real operational dataset before claiming generalisation, which is why "no real-world validation" is a top-line limitation.

### Q15a. Why C-MAPSS only? Doesn't that limit your IoT generality claim?
> The framework is the contribution, the dataset is the validation case. Layer~2 (the DQN observation space) and Layer~3 (the safety supervisor) contain no asset-specific assumptions — they operate on abstract health and uncertainty signals, so they're reusable across IoT domains. Only Layer~1 needs retraining for a new asset class. C-MAPSS was chosen for four specific reasons: (1) it's the most thoroughly benchmarked run-to-failure dataset in prognostics, so my prediction quality is directly comparable to hundreds of published baselines; (2) turbofans are the hardest IoT case — high-dimensional, multivariate, noisy, safety-critical — so an approach that works under C-MAPSS noise has earned a defensible claim it'll work on simpler analogues like wind turbines, CNC spindles, HVAC compressors, or pump fleets; (3) real industrial IoT data almost never has true RUL labels because operators replace components long before failure, but C-MAPSS is run-to-failure so every cycle has ground truth — that's essential for honest training and evaluation; (4) it's public, so the experiments are reproducible. Real proprietary IoT data would have made the project unverifiable. The cross-domain transfer test — retraining Layer~1 on a different IoT sensor dataset while keeping Layers~2 and~3 — is a top-line item in my future work.

### Q15b. So you're calling this an IoT framework but you only validated on jet engines?
> The framing is intentional. The title says "IoT Systems" because the architecture is designed to be domain-agnostic, and the report says explicitly in Section~1.2 that turbofan engines are the validation case study, not the target domain. Think of it the same way a paper on a new optimiser might validate on ImageNet — ImageNet is the test bed, the optimiser is what's being claimed. Here, C-MAPSS is the test bed, the three-layer uncertainty-aware architecture is what's being claimed. I'm careful in the report not to claim deployment readiness for any specific domain — both turbofans and other IoT assets would need real-world validation before deployment, and that's listed as a top-line limitation.

### Q15c. Could the architecture really transfer to, say, a wind turbine?
> Architecturally yes, with the caveat that Layer~1 needs retraining on wind turbine data. The reason it transfers is that Layers~2 and~3 don't see raw sensor channels — they see four abstract signals: normalised RUL prediction, instantaneous uncertainty, rolling uncertainty, and a sensor trend. Those four signals exist in any prognostics problem regardless of the underlying sensor modality. The reward structure (jackpot / safe / wasteful / crash) and the supervisor's threshold logic are also asset-agnostic — they encode "act early enough but not too early," which is the universal PdM problem. What \emph{wouldn't} transfer cleanly is the specific RUL cap of 125 and the noise distribution range, both of which would need tuning per asset class. But the core mechanism — uncertainty-aware decision-making with a deterministic backstop — is portable.

### Q15. Walk me through one cycle of the system.
> At cycle $t$, sensor data comes in — 24 features. The 30-cycle window is fed into all 5 LSTMs in parallel, producing 5 RUL predictions. I take the mean ($\mu$) and standard deviation ($\sigma$). $\sigma$ is scaled by 15 to map the raw range into something useful. The DQN observation is built: $[\mu, \sigma_{now}, \sigma_{rolling\_3}, \text{sensor trend}]$. The DQN forward pass produces Q-values for WAIT and MAINTAIN, picks the argmax. That proposed action goes to Layer 3. The supervisor checks Tier 1 first — if $\mu < 0.04$, override to MAINTAIN unconditionally. If not, check Tier 2 — if $\mu < 0.12$ AND $\sigma_{rolling} < 0.55$, override to MAINTAIN. Otherwise, pass the agent's action through. Action executes in the environment, episode either continues or terminates.

---

## 5. Likely "trap" questions from the report

These are sentences in the report that, if read carefully, invite a follow-up. Be ready.

### Trap 1. "On clean data the blind agent achieved a higher raw reward (432.1 vs 396.6)."
**Likely follow-up:** *So your agent loses on clean data?*
> Only on raw reward, and only because the supervisor rescues blind heavily. UA scores 396 with 13 supervisor interventions; blind scores 432 with 201 interventions. If you measure autonomous performance — which is what matters for deployment — UA wins 70% to 43% on jackpots. The clean-data raw-reward gap is a robustness-optimality trade-off: I trained UA to respond to mild uncertainty even on clean data, which makes it slightly more conservative on the easy condition but far more robust everywhere else.

### Trap 2. "PPO achieved higher jackpot rates but suffered unacceptable failure rates."
**Likely follow-up:** *So PPO is actually better at finding optimal timing?*
> PPO is more aggressive at finding the +500 jackpot, yes, but it does so by being willing to crash. Up to 18.3% of clean-data episodes ended in catastrophic failure under PPO versus 0% for DQN. In a safety-critical setting, jackpot rate is the wrong metric — you have to look at risk-adjusted reward, where DQN dominates. That's exactly why I kept DQN.

### Trap 3. "Effect sizes were moderate (peak Cohen's d 0.65)."
**Likely follow-up:** *Medium isn't large. How much should I trust this?*
> Three things compress the achievable effect size. First, the binary action space: there are only two policies the agent can express, so there's a ceiling on how different two agents can look. Second, the symmetric ±500 shaping: any agent that gets timing right occasionally produces a high mean reward, which compresses the gap. Third, the supervisor in Experiment 2 actively closes the gap because it rescues weaker policies. The fair comparison — Experiment 1, no supervisor — is where the effect size is most meaningful, and there it grows monotonically with noise from negligible at clean data to 0.65 at heavy noise. The monotonic trend matters more than the peak value.

### Trap 4. "PICP at nominal 0.95 is 0.60."
**Likely follow-up:** *That's pretty bad calibration. Doesn't this break the whole architecture?*
> No, and the reason is in how the DQN consumes $\sigma$. It doesn't consume it as a calibrated probability — it consumes it as a relative signal alongside three other features. What the agent learns is "when $\sigma$ is high relative to the training distribution, I should be more cautious." That's a learned monotone function of $\sigma$, not a function that depends on $\sigma$ being a true 95% interval width. Appendix H.2 shows the decision boundary bends with $\sigma$ exactly the way you'd predict from the training signal. So the trained policy is not invalidated. But you're right that for a production system I would add a recalibration step before exposing $\sigma$ to anything downstream, and I list that in the limitations.

### Trap 5. "24,000 episodes is a lot of evidence."
**Likely follow-up:** *Are these statistically independent?*
> Each episode draws a fresh random engine and a fresh noise level from the seeded distribution, and the seeds are fixed for reproducibility. Within a single condition, the 500 episodes are independent draws from the engine distribution. Across noise levels they're not independent draws from a single distribution — they're samples from distinct conditions, which is exactly what I want for the noise-vs-performance comparison. The Welch's $t$-test compares pairs of conditions and is appropriate for unequal variances, which is why I picked it over a standard $t$-test.

### Trap 6. "The financial analysis shows £556k saved per year."
**Likely follow-up:** *Where do the cost figures come from?*
> They're industry-representative but not validated against operator records — I list this as a limitation. The £18k jackpot, £35k safe, £52k wasteful, £275k crash figures are derived from public sources on aerospace maintenance economics. The analysis is best read as a sensitivity demonstration: for any plausible cost ratio where a crash is much more expensive than a wasteful maintenance, the UA system wins. The exact pound figure depends on the inputs.

---

## 6. Things you should NOT say

- ❌ "My system is novel" → say "the integration is the contribution"
- ❌ "It's better than published baselines" → say "in the right ballpark on prediction accuracy"
- ❌ "It always wins" → say "it wins on the fair comparison and on autonomy metrics; on raw clean-data reward with the supervisor active, blind is slightly higher because the supervisor rescues it"
- ❌ "The supervisor guarantees safety" → say "the supervisor reduced failure rates to near zero in the experimental conditions"
- ❌ "It's ready for deployment" → say "it's a research prototype; deployment would require certification under DO-178C and real-world validation"
- ❌ "The ensemble is well calibrated" → say "the ensemble is mildly overconfident at high coverage levels but the relative $\sigma$ signal is what the agent uses"
- ❌ "I invented X" → say "I implemented and integrated X"
- ❌ Anything beginning with "obviously" or "trivially"
- ❌ Speculation about why an examiner is asking — just answer the question

---

## 7. The two questions you must rehearse out loud

These are the most likely opening questions. Practice answering each in under 90 seconds, out loud, three times before the viva.

**Opening Q1: "Tell me about your project in your own words."**
> Use the 2-minute walkthrough above. Don't read from notes.

**Opening Q2: "What's the most important finding?"**
> The asymmetry in autonomous jackpot rates. UA achieves 70% optimal-timing decisions without supervisor intervention; blind only achieves 43%. Both reach near-zero failure under the full system, but the UA agent achieves safety through its own learned policy while the blind agent is essentially rescued by the supervisor. That's the strongest evidence that uncertainty awareness changes what the agent actually learns, not just how it scores.

---

## 8. If you get a question you don't know the answer to

Three legitimate moves:
1. **"That's a good question — let me think for a second."** Then think. Silence is fine.
2. **"I didn't test that directly, but my expectation would be X because Y."** Show reasoning even if you don't have data.
3. **"I don't know — it would be a good follow-up experiment."** This is acceptable once or twice. Do not bluff.

Do not invent numbers. Do not bluff a citation. Do not say "I think it's around X" — either you know or you don't.

---

## 9. Last 24 hours before the viva

- Read [main.tex](main.tex) cover to cover. **Every page.** Highlight any sentence you couldn't defend.
- Re-run [dashboard.py](dashboard.py) and play with it for 20 minutes. They may ask you to demo.
- Re-read this file the morning of, then close it.
- Bring a printed copy of the report. Tab the key tables.
- Sleep.

---

## 10. The mindset

You built a working three-layer system that addresses a real gap in the literature, you ran 24,000 evaluation episodes, you applied proper statistics, you took your supervisor's feedback seriously and rewrote sections to address it, and you're honest about limitations including the unflattering calibration result. That's a solid first-class FYP. Walk in confident, answer in your own words, and let the work speak.
