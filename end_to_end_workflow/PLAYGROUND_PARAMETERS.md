# Playground Parameters Guide

This document describes all the adjustable parameters in the workflow application. All values can be modified in the sidebar to test different configurations of the adaptive interview algorithm.

## Time Allocation Formula

### Importance Exponent (α)
- **Default**: 2.0
- **Range**: 0.0 - 10.0
- **Formula**: `Tcap = (I^α / ΣI^α) × T_total`
- **Effect**: Controls how time is distributed across objectives based on importance
  - Higher α: More time concentrated on high-importance objectives
  - Lower α: More even distribution of time
  - α = 0: Equal time for all objectives
  - α = 1: Linear proportional to importance
  - α = 2 (default): Quadratic emphasis on importance

## Priority Formula

### Depth Weight (D-E gap multiplier)
- **Default**: 0.1
- **Range**: 0.0 - 1.0
- **Formula**: `P = (I/(1+C)) × (1 + weight×|D_target - E|) + B`
- **Effect**: Controls how much the difficulty-evidence gap affects priority
  - Higher weight: Bigger priority boost for objectives where D_target differs from E
  - Lower weight: Less impact from difficulty-evidence mismatch
  - 0.0: Gap has no effect on priority
  - 0.1 (default): 10% boost per unit of gap

### Probing Bonus (β)
- **Default**: 50.0
- **Range**: 0.0 - 100.0
- **Effect**: Bonus added to priority when objective enters probing mode
  - Higher β: Stronger tendency to continue probing the same objective
  - Lower β: More likely to switch to other objectives
  - Used when consecutive low scores trigger deeper investigation

## Completion Thresholds

### Confidence Threshold
- **Default**: 0.9
- **Range**: 0.0 - 1.0
- **Effect**: Objective completes when `C >= threshold`
  - Higher threshold: More questions required per objective
  - Lower threshold: Earlier completion, more objectives covered
  - 0.9 (default): High confidence required (90%)
  - 1.0: Perfect confidence required (unrealistic)
  - 0.7-0.8: Moderate confidence acceptable

### Consecutive Failures for Stuck
- **Default**: 2
- **Range**: 1 - 5
- **Effect**: Number of consecutive low scores before moving to next objective
  - Higher value: More persistent, will ask more questions before giving up
  - Lower value: Moves on quickly from struggling areas
  - 1: Give up after single low score
  - 2 (default): Two strikes and you're out
  - 3+: Very persistent probing

## Score Thresholds

### Low Score Threshold
- **Default**: -3
- **Range**: -10 to 0
- **Effect**: Score below this triggers:
  - Difficulty decrease in probing mode
  - Stuck detection counter increment
  - Higher magnitude (e.g., -5): Only very poor answers trigger effects
  - Lower magnitude (e.g., -2): More sensitive to weak answers

### High Score Threshold
- **Default**: 8
- **Range**: 0 to 10
- **Effect**: Score above this triggers difficulty increase in probing mode
  - Higher threshold (e.g., 9): Only exceptional answers increase difficulty
  - Lower threshold (e.g., 6): Good answers increase difficulty
  - Used to ramp up challenge for strong candidates

## Testing Scenarios

### Conservative Configuration (maximize coverage)
- Confidence Threshold: 0.7
- Consecutive Failures: 1
- Depth Weight: 0.05
- α: 1.5

### Aggressive Configuration (deep probing)
- Confidence Threshold: 0.95
- Consecutive Failures: 3
- Depth Weight: 0.2
- β: 75.0
- α: 2.5

### Balanced Configuration (default)
- Confidence Threshold: 0.9
- Consecutive Failures: 2
- Depth Weight: 0.1
- β: 50.0
- α: 2.0

### Time-Focused Configuration
- α: 3.0 (heavy emphasis on important objectives)
- Confidence Threshold: 0.85
- Consecutive Failures: 2

## Observable Effects

When you change parameters, watch for:

1. **Interview Duration**: How long until all objectives complete or time runs out
2. **Objective Coverage**: How many objectives get questioned vs. how many complete
3. **Question Distribution**: How many questions per objective
4. **Switching Behavior**: How often the interview switches between objectives
5. **Completion Patterns**: Which objectives complete and which time out

## Recommended Experiments

1. **Test α sensitivity**: Try α = 0, 1, 2, 3 to see time distribution changes
2. **Test confidence impact**: Run with 0.7, 0.85, 0.95 to see coverage vs. depth tradeoff
3. **Test stuck detection**: Set to 1 vs. 3 to see persistence differences
4. **Test depth weight**: Try 0.0, 0.1, 0.3 to see priority selection changes
5. **Test score thresholds**: Adjust to see difficulty adaptation behavior
