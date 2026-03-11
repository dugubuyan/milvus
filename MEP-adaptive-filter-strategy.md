# MEP: Adaptive Filter Strategy Selection via Cost-Based Estimation

- **Status**: Proposed
- **Author**: <!-- your name -->
- **Created**: 2026-03-06
- **Related Issue**: <!-- link to GitHub issue -->

## Summary

Add a cost-based mechanism to the QueryNode that automatically selects the optimal filter execution strategy (pre-filter vs. iterative filter) per segment at query time, using existing per-segment field statistics to estimate filter selectivity.

## Motivation

Milvus supports two filter execution strategies in segcore:

**Pre-filter** (default):
```
FilterBitsNode → MvccNode → VectorSearchNode
```
Evaluates the scalar filter over the entire segment first, producing a bitset, then runs ANN search on surviving candidates. Optimal when filter selectivity is **high** (few rows pass the filter).

**Iterative filter**:
```
MvccNode → VectorSearchNode → IterativeFilterNode
```
Runs ANN search first to get top-K candidates, then validates each against the scalar filter, iterating until enough valid results are collected. Optimal when filter selectivity is **low** (most rows pass the filter).

The strategy is currently selected statically via a user-supplied `hints` parameter:

```python
collection.search(..., param={"hints": "iterative_filter"})
```

This creates two problems:

1. **Wrong default for common queries.** A filter like `age > 18` (where 99% of rows pass) defaults to pre-filter, which scans the entire segment to produce a nearly-full bitset — significant wasted work before ANN search even begins.

2. **Burden on users.** Users must understand their data distribution and manually tune `hints` per query. This is impractical in production, especially as data distributions shift over time.

Traditional relational databases (PostgreSQL, MySQL) solve this with a Cost-Based Optimizer: estimate cardinality after filtering, model the cost of each execution path, and automatically pick the cheapest plan. Milvus lacks this layer entirely for hybrid (vector + scalar) queries.

## Design

### Core Idea

Estimate filter selectivity per segment using existing `FieldStats`, then automatically set `iterative_filter_execution` in the search plan for each segment accordingly. No user intervention required.

### Selectivity Estimation

`FieldStats` (`internal/storage/field_stats.go`) already stores `Min`, `Max`, and a `BloomFilter` per field per segment, persisted in `PartitionStatsSnapshot`. This is sufficient for lightweight selectivity estimation:

| Expression type | Estimation method |
|---|---|
| `field > val`, `field < val` | `(Max - val) / (Max - Min)` |
| `a < field < b` | `(b - a) / (Max - Min)` |
| `field = val` | BloomFilter existence check; if present, `1 / estimated_ndv` |
| `field IN [v1, v2, ...]` | sum of per-value estimates, capped at 1.0 |
| `expr1 AND expr2` | `sel(expr1) * sel(expr2)` (independence assumption) |
| `expr1 OR expr2` | `sel(expr1) + sel(expr2) - sel(expr1) * sel(expr2)` |
| `NOT expr` | `1 - sel(expr)` |
| unknown / unsupported | fall back to a configurable default (e.g. 0.5) |

If `FieldStats` is unavailable for a segment (e.g. growing segment, stats not yet flushed), fall back to the current default behavior (pre-filter).

### Decision Rule

```
if estimatedSelectivity(filterExpr, segment.FieldStats) > threshold:
    iterative_filter_execution = true
else:
    iterative_filter_execution = false
```

The threshold represents the crossover point where iterative filter becomes cheaper than pre-filter. A reasonable starting default is **0.5** (50%), configurable via `paramtable`.

### Where the Change Lives

The decision is made in the **QueryNode delegator layer**, before `searchSegments()` is called, following the same pattern as `PruneByScalarField` in `scalar_pruner.go`.

```
internal/querynodev2/delegator/
  scalar_pruner.go          ← existing: prune segments using FieldStats
  filter_strategy_advisor.go  ← new: estimate selectivity, return hints per segment
```

The per-segment hints are then injected into the `SearchRequest` before being dispatched to each segment's `Search()` call.

### Affected Components

| Component | Change |
|---|---|
| `internal/querynodev2/delegator/filter_strategy_advisor.go` | New file: selectivity estimator + strategy decision |
| `internal/querynodev2/tasks/search_task.go` | Pass per-segment hints into search requests |
| `internal/querynodev2/segments/search.go` | Accept per-segment hints override |
| `pkg/util/paramtable/component_param.go` | New config: `queryNode.adaptiveFilterThreshold` |

No changes to the C++ segcore layer. The existing `hints` mechanism in `PlanProto.cpp` is reused as-is.

### Configuration

```yaml
queryNode:
  adaptiveFilterStrategy:
    enabled: true          # default: true; set false to restore legacy behavior
    threshold: 0.5         # selectivity above this → iterative filter
```

Both parameters are `refreshable: true` to allow runtime tuning without restart.

## Alternatives Considered

**1. Proxy-layer heuristic (expression structure only)**

Estimate selectivity purely from the filter expression type (equality → high selectivity, range → low selectivity), without actual statistics. Simpler, but inaccurate — a range filter on a clustered field can still be highly selective. Rejected in favor of statistics-based estimation.

**2. Runtime adaptive execution**

Collect actual filter hit rates during execution and dynamically switch strategies mid-query (similar to PostgreSQL's adaptive query execution). Most accurate, but requires changes to the C++ segcore execution engine and is significantly more complex. Could be pursued as a follow-up.

**3. Collection-level or query-level threshold override**

Expose the selectivity threshold as a per-collection property or per-query parameter, letting advanced users tune the decision boundary. This is complementary and can be layered on top of this proposal.

## Rollout Plan

- The feature is **opt-out** via `queryNode.adaptiveFilterStrategy.enabled`.
- Existing behavior is fully preserved when disabled, or when `FieldStats` are unavailable for a segment.
- No API changes. No proto changes. No C++ changes.

## Test Plan

- Unit tests for the selectivity estimator covering all expression types, including edge cases (empty range, `Min == Max`, missing stats).
- Unit tests for the strategy decision logic at various threshold values.
- Benchmark comparing query latency with adaptive strategy vs. fixed pre-filter on synthetic datasets with varying selectivity (1%, 10%, 50%, 90%).

## References

- Existing scalar pruner: `internal/querynodev2/delegator/scalar_pruner.go`
- FieldStats definition: `internal/storage/field_stats.go`
- Filter strategy flag: `internal/core/src/common/QueryInfo.h`, `internal/core/src/query/PlanProto.cpp`
- Hints plumbing (Go): `internal/proxy/search_util.go`, `pkg/proto/plan.proto` (`QueryInfo.hints`)
- Execution path comments: `internal/core/src/exec/operator/VectorSearchNode.cpp`
