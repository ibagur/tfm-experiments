python evaluate_rl_minigrid.py --algo ppo --approach blip --date 2023-01-21 \
    --experiment minigrid_5e5_doorkey-wallgap-lavagap-redbluedoor-emptyrand \
    --num-eval-episodes 30 --tasks-sequence 2 && \
python evaluate_rl_minigrid.py --algo ppo --approach ewc --date 2023-01-21 \
    --experiment minigrid_5e5_doorkey-wallgap-lavagap-redbluedoor-emptyrand \
    --num-eval-episodes 30 --tasks-sequence 2 && \
python evaluate_rl_minigrid.py --algo ppo --approach fine-tuning --date 2023-01-21 \
    --experiment minigrid_5e5_doorkey-wallgap-lavagap-redbluedoor-emptyrand \
    --num-eval-episodes 30 --tasks-sequence 2
