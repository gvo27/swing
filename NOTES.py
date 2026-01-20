I'm now writing notes digitally as of v2_bo (old)
{branches
[checkpoints (procedural branches, you can come back to it)
""inside chat

{so when we say...}: continue here for any other paper trading questions

31b_portfolio_backtest_slots_cash3 to v1_1_31b_slots_cash3_FREEZE, changing the end date to 12/22/2025 didn't return the same numbers

IMPORTANT: FREEZE the dataset for future testing, and have separate pathing for the live data set
I will ensure the "same" version gets shipped to the live version 

["Reply with: ablation and I’ll paste it."
Pre The Great Schism (it's not even that "far" back from where we are now because we haven't gotten anything better since, I would try sending it both of the entire scripts and see why test.py (or what we're making in this branch assuming it does the same gate) and FREEZE produce such different numbers)
31b_portfolio_backtest_slots_cash.py
32_portfolio_compare_ablation.py: tracing all the back to just before the gate that got us our "current" numbers, 32 reads off of 31b and needs the EB split to get different numbers,need to fix the date, *this is no longer in effect cause EB allo made much worse numbers 
2: fixed date
3:gate, close_col, date_col 
*31b_portfolio_backtest_slots_cash4.py is the EXACT same as FREEZE


PAPER TRADING:
I started over because we got too bogged on on debugging. We are now setting up paper trading then moving into v2 in the same chat. 
run_daily.py, works 
v1_1_tp_off_riskoff_live.py: simplified script for getting signals 
	2: fixed "tomorrow" error
01_build_dataset_sp1002: fixed column error
Signal fetching is working, we now just need a procedure set up:
	we set up the google sheet, we have the signal ready for tomorrow
now I want to know what I do if the signal goes through
then incorporate earning dates and news DONE
update 15_build_qqq_dd52w.py and 01_build_dataset_sp1002.py everyday before running v1_1_tp_off_riskoff_live2.py

LIVE TRADING: limit order, 1/2 size floor round down


v2:
"v2.0 design principles (write these down)"
domains: reduce 2022 with improved filtering, exit archetypes beyond just time, dynamic rules for TP and MR
we may explore cross-strategy interaction

WE'RE HERE --> reduce 2022 with improved TP filtering (same branch as "my idea"):
tp_v3_isolation.py: it runs but slowly with only 32 configs and the original gpt code wanted to do 2000+... 
tp_v3_isolation.py2: updated script that's apparently faster
tp_v3_isolation_fast: stable, but not solving the problem at all 
tp_v3_isolation_fast2: just added some prints, concluded the regime filter isn't working, I ditched this because I kept making new errors {DF date range:...
tp_v3_gate_sweep.py: new code, need to fix some stuff in the code
tp_v3_gate_sweep2.py: new script again, test also failed, ditching TP_v3 to fix 2022, going back to a different MR idea
["Given your stated goal...
panic_mr_sweep.py
panic_mr_p1_years.py
v2_0_31b_slots_cash3_TP_v2__MR_stress_replace_MR_P1_FREEZE.py: first major 2022 win, but we lost CAGR
2: better, integration
3.py: just a gate tweak
4: budget tweak, figurin it out, better overall numbers but 2022 slid
5: currently exploded but we gotta make sure we're not cheating
6: same thing w sanities, it's legit!
7: another sanity, and then ask about your help 


GPT lead v2: 
"Option A"
tp_v3_sma200_sweep.py: sma200 is boolean, not percentage or distance
I'm opting to continue in the "my idea" branch just cause we're on the same page in terms of code
tp_v3_sma200_sweep.py2: just gonna try the changes here
01_build_dataset_sp1003: including the sma200 distance feature, new parquet 2  
tp_v3_sma200_sweep.py2(still, forgot to make a new copy): sma200 distance didn't help
tp_v3_sma200_sweep.py3: new regime filter for TP_v3
	inspect_qqq_regime.py, gave em the columns
tp_v3_sma200_sweep.py4: only seeing .05 above, block fix GPT is doing, numbers are overall better but still bad for 2022
tp_v3_sma200_sweep.py5: filtered out futurewarning alerts (thank you), I gave em the column names for qqq_dd52w (different filter for the 2022 bear market), that's all I've done
tp sizing over "naive hard gating", and let MR do the "heavy lifting"
v1_1_31b_slots_cash3_FREEZE2: dd-scale, outdated date indexing
3: 2 with the same index fix we did before 
4: dd-scale with tp 100%
5: original with tp 100%
6: 3 with print and True
7. 6 with TP gate neutralized 
8. 3 with dd-scale only with risk_on=True, last note where you are
9: .25 another TP change, reduced
10: ran it but didn't paste results, more budget sanity lines
11: MR budget w/ TP budget print replaced
	"MR sizing was the drawdown driver"
12: doesn't work with sanity print
I accidently deleted the chat


My idea: we're going to first try an updated mean reversion idea based on my old model, if it doesn't work, go back to "excellent I feel good now...", GPT *did* seem to capture what happened in 2022 for me great because all but four signals had P4 bucket activated
*I humbly have a suspicion greater than 10 may be too much*
16_build_spy_stress.py - it uses QQQ actually, which I'm fine with
17_backtest_mrp_isolation.py - the actual test 
we're onto editing the "panic" conditions for the next test
	2: think we're good so far
	3: sweep parameter, went through several edits of this, no constraints fit as of now
	4: filter "killer" diagnostic, code worked and close_pos was the silent killer
	5: found a positive result and we'll now integrate back in to portfolio
...v1_1_31b_slots_cash3_FREEZE2.py: first MR-P test didn't work, made improvements but we had a huge 2020 drop and the stress day count is low
v1_1_31b_slots_cash3_FREEZE3.py: implememnting
I ditched this and am returning to GPT's first idea




Stat line for v1_1_31b_slots_cash3_FREEZE.py:
Total return: 1023.90%
CAGR: 26.81%
Max Drawdown: -39.80%
Avg daily ret: 0.00110   Daily vol: 0.01763

Yearly returns:
date
2016    0.485918 
2017    0.158714
2018    0.077527
2019    0.324318
2020    0.444970
2021    0.288171
2022   -0.239692
2023    0.252233
2024    0.366291
2025    0.552920

first big 2022 win: 
v2_0_31b_slots_cash3_TP_v2__MR_stress_replace_MR_P1_FREEZE.py
Total return: 
713.68% CAGR: 22.86% 
Max Drawdown: -39.54% 
2016 0.500141 ~
2017 0.210263 +
2018 -0.087208 --
2019 0.232554 -
2020 0.397654 -
2021 0.244099 -
2022 -0.024261 +++
2023 0.259048 ~
2024 0.168957 ---
2025 0.311285 ---


python 31b_portfolio_backtest_slots_cash.py, False
False: Total return: 801.03% 
CAGR: 24.09% 
Max Drawdown: -37.37% 
2016 0.684076 
2017 0.092108 
2018 -0.141287 
2019 0.339545 
2020 0.391833 
2021 0.246825 
2022 -0.078640 
2023 0.267423 
2024 0.337792 
2025 0.291324


5.py/6.py (sanities)
Total return: 1311.76%
CAGR: 29.69%
Max Drawdown: -39.17%
2016    0.523920
2017    0.102859
2018    0.043142
2019    0.269673
2020    0.672208
2021    0.397222
2022   -0.125993
2023    0.248218
2024    0.316176
2025    0.540467



 








'v2':
*OLD* I accidently deleted the chat, this stuff is now in backtests_v2.0(old)_archived
04_backtest_bo_sweep.py: first run for v2
strong_gate:
04_backtest_bo_sweep2.py: 
step 1: replace QQQ map builder
step 2: replace a line
step 3: another line replace
step 4: add print lines, inexplicit code  
step 5: more print lines, inexplicit code
"Lock the BO signal"

09_streams_bo.py: concurrent run with mr and tp
09_streams_bo3.py: it's currently the same thing as bo2(deleted), I will include the updated fixes for the columns
09_streams_bo4.py same as bo3 with ret3d and sma10 calculations

"Do you want v2 to primarily reduce drawdowns, or increase CAGR, even if drawdowns stay similar?"

{what's the difference between...} explanation between CAGR and total return, CAGR calculates compounding, so essentially it's your necessary return assuming compounding to reach total return 

[I chose "reducing drawdowns", and we'll see where that takes us]
	Four blocks ie "Block 1"
	["exact regime rules"]
		"regime code" 
		08_build_market_regime_qqq.py
		09_streams_bo4.py, strong neutral, risk_off stuff
                v2_portfolio_regime.py
			worked, but much worse results (first backtest for v2)
		v2_portfolio_regime2.py
			adding two new changes, involving exposure caps 
		v2_portfolio_regime3.py
			another sizing logic change, we will change caps next as well  
		v2_portfolio_regime4.py
			tuning to try and increase CAGR, got drawdown down in 3
		v2_portfolio_regime5.py
			we've reverted BACK to 3 this one after 4 failed, changed some exposure stuff again and now are going to include code to get average open positions 
		v2_portfolio_regime6.py: conceptual fix, emphasis on strong (just changing exposure)




