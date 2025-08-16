#!/bin/bash

# 各參數單獨陣列
K_VALUES=(2 3 5)
CALIPER_VALUES=(0.3)
P_THRESH_VALUES=(0.01 0.005 0.03)


# 巢狀迴圈產生所有組合
for K in "${K_VALUES[@]}"; do
    for CAL in "${CALIPER_VALUES[@]}"; do
        for PTH in "${P_THRESH_VALUES[@]}"; do
            echo "===== 執行參數組合: -k $K -caliper $CAL -p_threshold $PTH ====="
            python3 main.py -k $K -caliper $CAL -p_threshold $PTH
        done
    done
done
