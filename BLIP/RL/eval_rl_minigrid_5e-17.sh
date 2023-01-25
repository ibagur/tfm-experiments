python evaluate_rl_minigrid.py --algo ppo --approach blip --date 2023-01-25 \
    --experiment minigrid_5e5_redbluedoor-lavagap-doorkey-wallgap \
    --num-eval-episodes 30 --tasks-sequence 1 --F-prior 5e-17 && \
python evaluate_rl_minigrid.py --algo ppo --approach ewc --date 2023-01-21 \
    --experiment minigrid_5e5_redbluedoor-lavagap-doorkey-wallgap \
    --num-eval-episodes 30 --tasks-sequence 1 && \
python evaluate_rl_minigrid.py --algo ppo --approach blip --date 2023-01-25 \
    --experiment minigrid_5e5_lavagap-doorkey-emptyrand-redbluedoor-wallgap \
    --num-eval-episodes 30 --tasks-sequence 3 --F-prior 5e-17 && \
python evaluate_rl_minigrid.py --algo ppo --approach ewc --date 2023-01-22 \
    --experiment minigrid_5e5_lavagap-doorkey-emptyrand-redbluedoor-wallgap \
    --num-eval-episodes 30 --tasks-sequence 3

