# テストスイート解説

## 実行方法

```bash
julia --project -e 'using Pkg; Pkg.test()'
```

## 全体構成

```
test/
  runtests.jl                      # エントリポイント
  test_estimators.jl               # 推定器の update! テスト
  test_system.jl                   # System.run! 統合テスト
  test_selection_probabilities.jl  # 行動選択確率の整合性テスト
  test_evaluation.jl               # 評価関数テスト
```

`runtests.jl` は `using MultiBandits` でパッケージを読み込み、各テストファイルを順に実行する。

---

## 1. test_estimators.jl (41 tests)

**目的**: 各推定器 (Estimator) の `update!` 関数が、状態ベクトル (Q, W, N 等) を
正しく更新するかを手計算の値と照合して検証する。

### テスト内容

| testset | 推定器 | 検証内容 |
|---------|--------|----------|
| **EmpiricalReward** | `EmpiricalReward` | 初期値が Q=0, N=0 であること。報酬 1.0 → Q=1.0, 報酬 0.0 → Q=0.5（報酬平均）。W が Q と一致すること。未選択腕が変化しないこと |
| **ThompsonSampling** | `ThompsonSampling` | 初期値 alpha=1, beta=1 (Beta(1,1)=一様分布)。成功で alpha+1、失敗で beta+1。update 後に W が有限値であること（サンプルが正常） |
| **UCB1** | `UCB1` | 初期 Q=Inf（未選択腕を優先的に探索）。全腕を1回ずつ選択後、Q が報酬平均になること。ボーナス項 B が正の有限値であること |
| **STEP** | `STEP` | 成功を3回与えた腕の閾値超過確率 W が 0.5 より大きいこと。未選択腕は Beta(1,1) の CDF から W=0.5 であること |
| **RS** | `RS` | RS値 = N * (Q - aleph)。報酬1.0 を3回で W = 3*(1.0-0.5) = 1.5。未選択腕は W=0 |
| **CognitiveEstimator (default)** | `CognitiveEstimator` | デフォルト設定 (alpha=-1.0) で報酬平均になること。Q = (1.0+0.0)/2 = 0.5 |
| **CognitiveEstimator (fixed LR)** | `CognitiveEstimator(alpha=0.3)` | 固定学習率での Q 更新。Q = 0 + 0.3*(1.0-0) = 0.3、次に Q = 0.3 + 0.3*(0-0.3) = 0.21 |
| **CognitiveEstimator (forgetting)** | `CognitiveEstimator(alphaf=0.1, mu=0.5)` | 未選択腕の Q が忘却デフォルト値 mu=0.5 に向かってドリフトすること |
| **CognitiveEstimator (stickiness)** | `CognitiveEstimator(tau=0.5, phi=1.0)` | 選択した腕の固執性 C が増加すること。未選択腕の C が 0 のままであること。W = beta*V + phi*C により W > V となること |
| **CognitiveEstimator (BetaCDF)** | `CognitiveEstimator(eta=0.3, nu=5.0, alpha=0.5)` | Beta CDF による効用変換で V != Q となること。V が [0, 1] の範囲内であること |
| **EmptyEstimator** | `EmptyEstimator` | update! が何もせず nothing を返すこと（RandomResponding 用） |

---

## 2. test_system.jl (12 tests, 1 broken)

**目的**: Agent (= Policy + Estimator) と Environment を組み合わせた System が、
`run!` で指定 trial 数を**エラーなく完走**するかを検証する統合テスト。

全テストで `MersenneTwister(42)` による固定シードを使い、再現性を確保している。
環境は 4本腕の Bernoulli バンディット `[0.2, 0.4, 0.6, 0.8]`。

### テスト内容

| testset | Policy + Estimator | 検証内容 |
|---------|-------------------|----------|
| **Greedy + EmpiricalReward** | `Greedy` + `EmpiricalReward` | 100 trial 完走。全 trial で行動が記録されていること。報酬が 0.0 か 1.0（Bernoulli）であること |
| **Greedy + ThompsonSampling** | `Greedy` + `ThompsonSampling` | 100 trial 完走。行動が 1 ~ n_arms の範囲内であること |
| **Greedy + UCB1** | `Greedy` + `UCB1` | **`@test_broken`**: UCB1 の `update!` で未選択腕のボーナス項 `sqrt(log(t)/0)` が NaN になり、`argmax` が空リストを返して例外が発生するバグ。修正されるまで broken として記録 |
| **Greedy + STEP** | `Greedy` + `STEP(aleph=0.5)` | 100 trial 完走 |
| **Greedy + RS** | `Greedy` + `RS(aleph=0.5)` | 100 trial 完走 |
| **RandomResponding + EmptyEstimator** | `RandomResponding` + `EmptyEstimator` | 100 trial 完走（推定器なしのランダム選択） |
| **SimpleSat + EmpiricalReward** | `SimpleSat(0.5)` + `EmpiricalReward` | `SimpleSatAgent` コンストラクタ経由。100 trial 完走 |
| **SoftmaxPolicy + CognitiveEstimator** | `SoftmaxPolicy(1.0)` + `CognitiveEstimator(beta=3.0, alpha=0.1)` | CognitiveEstimator 内部の beta=3.0 が実質的な逆温度として機能。100 trial 完走 |
| **NonStationaryEnvironment** | `Greedy` + `ThompsonSampling`, `NonStationaryEnvironment` | change point を trial 数より後に設定し、History の定常性 assert に引っかからないようにして 100 trial 完走 |

### 既知の問題

- **UCB1 の NaN バグ**: `update!` 内で `e.N[a] == 0` の腕に対して `sqrt(log(t)/e.N[a])` を計算し NaN が発生する。
  初期値 `Q0=Inf` により最初は全腕が選ばれるが、`update!` 時点で他の腕の N が 0 のままボーナスを計算してしまう。
- **NonStationaryEnvironment + History**: `History` の `record!` は定常環境を前提とした assert (`expectations` が全 trial 同一) を持つため、
  change point をまたぐシミュレーションでは `EstimatorHistory` か、assert を緩和した History を使う必要がある。

---

## 3. test_selection_probabilities.jl (19 tests)

**目的**: 各 Policy の `selection_probabilities` 関数の出力が**有効な確率分布**
（全要素が非負、合計が 1.0）であることを検証する。

ヘルパー関数 `is_probability_distribution(probs)` で共通チェックを行う:
```julia
all(probs .>= 0.0) && isapprox(sum(probs), 1.0; atol=1e-10)
```

### テスト内容

| testset | 検証内容 |
|---------|----------|
| **Greedy + EmpiricalReward** | 1腕だけ報酬を与えた後、その腕の選択確率が 1.0 であること |
| **Greedy: tied arms** | 全腕の価値が同じとき（初期状態）、均等確率 1/n_arms になること |
| **SoftmaxPolicy** | 通常 (beta=1.0): 確率分布であること。高温 (beta=100.0): 最良腕の確率が 0.99 超。低温 (beta=0.0): 均等分布 |
| **RandomResponding** | デフォルト: 均等確率 1/n_arms。カスタム確率: 指定した確率ベクトルがそのまま返ること |
| **SimpleSat** | 満足化方策での確率が有効な確率分布であること |
| **CognitiveEstimator (W vector)** | W ベクトルから直接 softmax で計算した確率が確率分布であること |
| **available_arms** | 利用可能な腕のみに確率が割り当てられ、他の腕が 0.0 であること。全体として確率分布であること |

---

## 4. test_evaluation.jl (17 tests)

**目的**: シミュレーション結果の評価関数が正しい値を返すかを検証する。

### テスト内容

| testset | 関数 | 検証内容 |
|---------|------|----------|
| **cumulative_regret (stationary)** | `cumulative_regret(actions, expectations::Vector{Float64})` | 2本腕 [0.3, 0.7] で腕1を2回選択後に腕2を3回選択。累積リグレット = 2*0.4 = 0.8 で停滞。単調非減少であること |
| **cumulative_regret (non-stationary)** | `cumulative_regret(actions, expectations::Vector{Vector{Float64}})` | trial ごとに期待値が変わる場合の計算。各 trial の最適腕との差を正しく累積すること |
| **moving_average_rewards** | `moving_average_rewards(rewards, window_size)` | ウィンドウサイズ 3 での移動平均が手計算と一致すること |
| **average_rewards** | `average_rewards(rewards)` | 累積平均報酬。最初の trial で 1.0、4 trial 後に 0.5 であること |
| **action_moving_averages** | `action_moving_averages(actions, n_arms, window_size)` | 出力行列のサイズが (n_trials, n_arms) であること。各行（trial）の合計が 1.0 であること（各 trial の行動選択割合の合計は常に 1） |

---

## テスト結果サマリ（2026-03-06 時点）

```
Test Summary:     | Pass  Broken  Total
GoalDirectedTS.jl |   88       1     89
  Estimators      |   41              41
  System run!     |   11       1      12
  selection_prob.. |   19              19
  Evaluation      |   17              17
```

- **88 passed**: 全主要機能が期待通り動作
- **1 broken**: UCB1 の NaN バグ（既知、パッケージ化時に修正予定）
