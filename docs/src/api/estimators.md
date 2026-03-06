# 推定器 (Estimators)

## 型

```@docs
AbstractEstimator
EmptyEstimator
EmpiricalReward
ThompsonSampling
ThompsonSamplingN
ThompsonSamplingNU
UCB1
STEP
RS
CognitiveEstimator
```

## 主要関数

```@docs
update!
selection_probabilities
MultiBandits.calculate_action_values!
```

## CognitiveEstimator 関連

```@docs
MultiBandits.parameters
MultiBandits.algo_params_default_vals
MultiBandits.count_params
MultiBandits.beta_a
MultiBandits.beta_b
MultiBandits.normalized_rs_values
```

## 表示

```@docs
MultiBandits.toString(::EmpiricalReward)
MultiBandits.toString(::UCB1)
Base.show(::IO, ::MIME"text/plain", ::EmpiricalReward)
Base.show(::IO, ::MIME"text/plain", ::UCB1)
Base.show(::IO, ::MIME"text/plain", ::CognitiveEstimator)
```
