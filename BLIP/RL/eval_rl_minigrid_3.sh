python evaluate_rl_minigrid.py --algo ppo --approach blip --date 2023-01-22 \
    --experiment minigrid_5e5_lavagap-doorkey-emptyrand-redbluedoor-wallgap \
    --num-eval-episodes 30 --tasks-sequence 3 && \
python evaluate_rl_minigrid.py --algo ppo --approach ewc --date 2023-01-22 \
    --experiment minigrid_5e5_lavagap-doorkey-emptyrand-redbluedoor-wallgap \
    --num-eval-episodes 30 --tasks-sequence 3 && \
python evaluate_rl_minigrid.py --algo ppo --approach fine-tuning --date 2023-01-22 \
    --experiment minigrid_5e5_lavagap-doorkey-emptyrand-redbluedoor-wallgap \
    --num-eval-episodes 30 --tasks-sequence 3
