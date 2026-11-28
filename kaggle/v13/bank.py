"""Bank of test rankings with a known public score (own submissions, pinned public files)."""
S = '/tmp/claude-0/-home-user-predicting-credit-card-fraud/a976eae0-0cac-5b1b-b4bb-31f43e21d186/scratchpad'
P = S + '/pin/'
G = S + '/pub/goodpjw2008_s6e11-board-model-realmlp-lb-0-95687/'
BANK = {
    # own submissions (file submitted, public score)
    S + '/pub/d28_tarunsingh931/submission.csv': 0.95762,
    S + '/pub/d28_mzishran/submission.csv': 0.95760,
    S + '/pub/d28_daoviet/submission.csv': 0.95758,
    S + '/pub/d28_rishabhbafnaiiitd/submission.csv': 0.95753,  # author's stated score
    S + '/v12/dao20_v8.csv': 0.95475,
    S + '/v12/local_step1.csv': 0.95765,  # v13 local board step 1
    S + '/v12/local_step2.csv': 0.95765,  # v13 local board step 2
    S + '/v12/priv10_step1_v8.csv': 0.95424,  # v13 private-first 10/90
    S + '/v12/probeB_b0.csv': 0.95770,  # v13 probe B block 0
    S + '/v12/probeB_b1.csv': 0.95783,  # v13 probe B block 1 (on b0)
    S + '/v12/probeB_b3.csv': 0.95790,  # v13 probe B block 3 (on b1)
    S + '/pub/new_daoviet2/submission.csv': 0.95740,
    S + '/v12/dao_mz_mean.csv': 0.95736,
    S + '/v12/board_gain0.001.csv': 0.95719,  # v13 board step 1 (predicted +0.001, real -0.00021)
    S + '/v12/dao_plus_half.csv': 0.95733,
    S + '/v12/extra_mz_ag_0.5.csv': 0.95728,
    S + '/v12/extra_mz_ag_1.0.csv': 0.95716,
    S + '/pub/new_mzishran/submission.csv': 0.95727,
    S + '/pub/new_rishabhbafnaiiitd/submission.csv': 0.95726,
    S + '/pub/monya1_s6e11-zero-fraud-zones-and-rank-extrapolation/submission.csv': 0.95700,
    S + '/pub/dennis/submission.csv': 0.95688,
    G + 'submission.csv': 0.95687,
    S + '/v12/monya75_v8.csv': 0.95677,
    S + '/v12/ag75_v8.csv': 0.95646,
    S + '/v12/ag50_v8.csv': 0.95576,
    S + '/v12/mean_stepab_v8.csv': 0.95511,
    S + '/v12/cv_led.csv': 0.95445,
    S + '/v12/cv_led_zones.csv': 0.95447,
    S + '/v12/cvled50_v8.csv': 0.95427,
    S + '/v12/cvled50_v8_zones.csv': 0.95427,
    '/home/user/v8m/stack_mean.csv': 0.95347,
    '/home/user/v7_stack.csv': 0.95307,
    '/home/user/v5out/blend.csv': 0.95559,
    '/home/user/ext/stepab/step_ab.csv': 0.95569,
    '/home/user/ext/stepab/clipped.csv': 0.95568,
    # goodpjw2008 steps with a published public score
    G + 'submission_step_ad.csv': 0.95582,
    G + 'submission_step_ae.csv': 0.95597,
    # pinned public files (score in the file name)
    P + 'areebbb_0p95552.csv': 0.95552, P + 'hosen_PM8r8b_0p95510.csv': 0.95510,
    P + 'malyshevdanil_forecast_0p95537.csv': 0.95537, P + 'malyshevdanil_round2_0p95539.csv': 0.95539,
    P + 'malyshevdanil_round3_0p95540.csv': 0.95540, P + 'malyshevdanil_round4_0p95541.csv': 0.95541,
    P + 'malyshevdanil_round5_0p95551.csv': 0.95551, P + 'malyshevdanil_round6_0p95557.csv': 0.95557,
    P + 'malyshevdanil_step_l_moved_0p95527.csv': 0.95527, P + 'malyshevdanil_v42_0p95565.csv': 0.95565,
    P + 'malyshevdanil_v43_0p95567.csv': 0.95567, P + 'sriram_orthogonal_0p95568.csv': 0.95568,
    P + 'scored_step_j_0p95508.csv': 0.95508, P + 'scored_step_k_0p95519.csv': 0.95519,
    P + 'scored_step_l_0p95527.csv': 0.95527, P + 'scored_step_m_0p95531.csv': 0.95531,
    P + 'scored_step_n_0p95533.csv': 0.95533, P + 'scored_step_o_0p95536.csv': 0.95536,
    P + 'scored_step_p_0p95539.csv': 0.95539, P + 'scored_step_q_0p95540.csv': 0.95540,
    P + 'scored_step_r_0p95546.csv': 0.95546,
}
