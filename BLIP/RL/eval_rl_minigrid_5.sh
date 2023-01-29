python evaluate_rl_minigrid.py --algo ppo --approach blip --date 2023-01-28 \
    --experiment minigrid_5e5_distshift-doorkey-wallgap-lavagap-redbluedoor-emptyrand \
    --num-eval-episodes 30 --F-prior 1e-16 --tasks-sequence 5 && \
python evaluate_rl_minigrid.py --algo ppo --approach ewc --date 2023-01-28 \
    --experiment minigrid_5e5_distshift-doorkey-wallgap-lavagap-redbluedoor-emptyrand \
    --num-eval-episodes 30 --tasks-sequence 5 && \
python evaluate_rl_minigrid.py --algo ppo --approach fine-tuning --date 2023-01-28 \
    --experiment minigrid_5e5_distshift-doorkey-wallgap-lavagap-redbluedoor-emptyrand \
    --num-eval-episodes 30 --tasks-sequence 5
