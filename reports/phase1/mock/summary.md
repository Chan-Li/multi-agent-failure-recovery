# Phase 1 results
source: `results/mock`, 189 clean rows, 492 fault runs

## Clean baseline
```
                       n  success  latency_s     tokens  verify_tokens  false_alarms  researcher_exact
mode=plain            63      1.0      0.010   8178.238          0.000           0.0               1.0
mode=verify_all       63      1.0      0.023  17029.286       8851.048           0.0               1.0
mode=verify_all_self  63      1.0      0.021  17029.286       8851.048           0.0               1.0
difficulty=easy       18      1.0      0.006   8327.500            NaN           NaN               NaN
difficulty=example     3      1.0      0.003   3011.000            NaN           NaN               NaN
difficulty=hard       18      1.0      0.018   8550.167            NaN           NaN               NaN
difficulty=medium     24      1.0      0.009   8433.250            NaN           NaN               NaN
```

## Propagation outcome levels (share of fault runs)
```
outcome                              stopped  ignored  contained  benign_change  failed  irreversible_harm
category    fault_id                                                                                      
accidental  researcher-stale-record      0.0      0.0      0.000          0.353   0.647              0.000
            researcher-timeout           1.0      0.0      0.000          0.000   0.000              0.000
            researcher-wrong-prereq      0.0      0.0      0.000          0.619   0.381              0.000
            researcher-wrong-seats       0.0      0.0      0.000          0.905   0.048              0.048
            researcher-wrong-time        0.0      0.0      0.048          0.524   0.333              0.095
            researcher-wrong-units       0.0      0.0      0.000          0.000   0.143              0.857
adversarial planner-injection            0.0      0.0      0.000          0.000   0.000              1.000
            researcher-injection         0.0      0.0      0.000          0.000   0.000              1.000
```

## Propagation metrics
```
                                      n  fired  depends  plan_changed  task_success  affected_stages  affected_actions  unauthorized_actions  harmful_irreversible  executor_refused  wasted_tokens  latency_s
category    fault_id                                                                                                                                                                                          
accidental  researcher-stale-record  51    1.0      1.0           1.0         0.353              2.0             3.176                   0.0                 0.000               0.0       8444.235      0.008
            researcher-timeout       63    1.0      1.0           NaN         0.000              0.0             0.000                   0.0                 0.000               0.0          0.000      0.000
            researcher-wrong-prereq  63    1.0      1.0           1.0         0.619              2.0             2.571                   0.0                 0.000               0.0       8167.048      0.010
            researcher-wrong-seats   63    1.0      1.0           1.0         0.905              2.0             3.238                   0.0                 0.048               0.0       8177.048      0.009
            researcher-wrong-time    63    1.0      1.0           1.0         0.619              2.0             3.333                   0.0                 0.095               0.0       8179.190      0.009
            researcher-wrong-units   63    1.0      1.0           1.0         0.000              2.0             3.571                   0.0                 0.857               0.0       8182.333      0.010
adversarial planner-injection        63    1.0      1.0           1.0         0.000              1.0             5.286                   1.0                 1.714               0.0       3714.762      0.009
            researcher-injection     63    1.0      1.0           1.0         0.000              2.0             5.286                   1.0                 1.714               0.0       8433.762      0.010
```

## Recovery by method (overall)
```
                                   n  success  clean_recovery  recomputed_stages  extra_tokens  latency_s  harmful_irreversible  persisted_events  detected
category    method                                                                                                                                         
accidental  no_recovery          366    0.418           0.410                0.0         0.000      0.000                 0.172             0.000       NaN
            full_restart         366    0.574           0.574                3.0      8210.033      0.009                 0.172             2.164       NaN
            full_restart_oracle  366    1.000           1.000                3.0      8217.000      0.010                 0.000             0.000       NaN
            verify_all           303    0.851           0.851                1.0     13523.693      0.027                 0.000             0.000       1.0
            verify_all_self      303    0.851           0.851                1.0     13523.693      0.027                 0.000             0.000       1.0
adversarial no_recovery          126    0.000           0.000                0.0         0.000      0.000                 1.714             0.000       NaN
            full_restart         126    0.095           0.000                3.0      8161.286      0.008                 1.714             4.810       NaN
            full_restart_oracle  126    1.000           1.000                3.0      8178.238      0.009                 0.000             0.000       NaN
            verify_all           126    0.857           0.857                1.0     12796.857      0.024                 0.000             0.000       1.0
            verify_all_self      126    0.857           0.857                1.0     12796.857      0.025                 0.000             0.000       1.0
```

## Recovery by fault and method
```
                                                          n  success  clean_recovery  recomputed_stages  extra_tokens  latency_s  harmful_irreversible  persisted_events  detected
category    fault_id                method                                                                                                                                        
accidental  researcher-stale-record no_recovery          51    0.353           0.353                0.0         0.000      0.000                 0.000             0.000       NaN
                                    full_restart         51    0.647           0.647                3.0      8456.412      0.012                 0.000             1.882       NaN
                                    full_restart_oracle  51    1.000           1.000                3.0      8456.412      0.012                 0.000             0.000       NaN
                                    verify_all           51    0.824           0.824                1.0     13916.059      0.029                 0.000             0.000       1.0
                                    verify_all_self      51    0.824           0.824                1.0     13916.059      0.028                 0.000             0.000       1.0
            researcher-timeout      no_recovery          63    0.000           0.000                0.0         0.000      0.000                 0.000             0.000       NaN
                                    full_restart         63    1.000           1.000                3.0      8178.238      0.010                 0.000             0.000       NaN
                                    full_restart_oracle  63    1.000           1.000                3.0      8178.238      0.010                 0.000             0.000       NaN
            researcher-wrong-prereq no_recovery          63    0.619           0.619                0.0         0.000      0.000                 0.000             0.000       NaN
                                    full_restart         63    0.619           0.619                3.0      8170.810      0.009                 0.000             2.286       NaN
                                    full_restart_oracle  63    1.000           1.000                3.0      8178.238      0.010                 0.000             0.000       NaN
                                    verify_all           63    0.857           0.857                1.0     13447.333      0.027                 0.000             0.000       1.0
                                    verify_all_self      63    0.857           0.857                1.0     13447.333      0.027                 0.000             0.000       1.0
            researcher-wrong-seats  no_recovery          63    0.905           0.905                0.0         0.000      0.000                 0.048             0.000       NaN
                                    full_restart         63    0.571           0.571                3.0      8168.714      0.008                 0.048             2.476       NaN
                                    full_restart_oracle  63    1.000           1.000                3.0      8178.238      0.009                 0.000             0.000       NaN
                                    verify_all           63    0.857           0.857                1.0     13443.286      0.028                 0.000             0.000       1.0
                                    verify_all_self      63    0.857           0.857                1.0     13443.286      0.026                 0.000             0.000       1.0
            researcher-wrong-time   no_recovery          63    0.619           0.571                0.0         0.000      0.000                 0.095             0.000       NaN
                                    full_restart         63    0.571           0.571                3.0      8168.714      0.008                 0.095             2.476       NaN
                                    full_restart_oracle  63    1.000           1.000                3.0      8178.238      0.011                 0.000             0.000       NaN
                                    verify_all           63    0.857           0.857                1.0     13443.333      0.027                 0.000             0.000       1.0
                                    verify_all_self      63    0.857           0.857                1.0     13443.333      0.026                 0.000             0.000       1.0
            researcher-wrong-units  no_recovery          63    0.000           0.000                0.0         0.000      0.000                 0.857             0.000       NaN
                                    full_restart         63    0.048           0.048                3.0      8164.238      0.008                 0.857             3.810       NaN
                                    full_restart_oracle  63    1.000           1.000                3.0      8178.238      0.010                 0.000             0.000       NaN
                                    verify_all           63    0.857           0.857                1.0     13443.190      0.024                 0.000             0.000       1.0
                                    verify_all_self      63    0.857           0.857                1.0     13443.190      0.028                 0.000             0.000       1.0
adversarial planner-injection       no_recovery          63    0.000           0.000                0.0         0.000      0.000                 1.714             0.000       NaN
                                    full_restart         63    0.095           0.000                3.0      8161.286      0.008                 1.714             4.810       NaN
                                    full_restart_oracle  63    1.000           1.000                3.0      8178.238      0.009                 0.000             0.000       NaN
                                    verify_all           63    0.857           0.857                1.0     12061.190      0.023                 0.000             0.000       1.0
                                    verify_all_self      63    0.857           0.857                1.0     12061.190      0.024                 0.000             0.000       1.0
            researcher-injection    no_recovery          63    0.000           0.000                0.0         0.000      0.000                 1.714             0.000       NaN
                                    full_restart         63    0.095           0.000                3.0      8161.286      0.009                 1.714             4.810       NaN
                                    full_restart_oracle  63    1.000           1.000                3.0      8178.238      0.010                 0.000             0.000       NaN
                                    verify_all           63    0.857           0.857                1.0     13532.524      0.025                 0.000             0.000       1.0
                                    verify_all_self      63    0.857           0.857                1.0     13532.524      0.026                 0.000             0.000       1.0
```
