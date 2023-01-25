python evaluate_rl_minigrid.py --algo ppo --approach blip --date 2023-01-19 \
    --experiment minigrid_5e5_doorkey-wallgap-lavagap-redbluedoor \
    --num-eval-episodes 30 --tasks-sequence 0 && \
python evaluate_rl_minigrid.py --algo ppo --approach ewc --date 2023-01-19 \
    --experiment minigrid_5e5_doorkey-wallgap-lavagap-redbluedoor \
    --num-eval-episodes 30 --tasks-sequence 0 && \
python evaluate_rl_minigrid.py --algo ppo --approach fine-tuning --date 2023-01-19 \
    --experiment minigrid_5e5_doorkey-wallgap-lavagap-redbluedoor \
    --num-eval-episodes 30 --tasks-sequence 0
