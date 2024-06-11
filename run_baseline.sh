export CUDA_VISIBLE_DEVICES=2
export WANDB_API_KEY=247e23f9da34555c8f9d172474c4d49ad150e88d
python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=3s_vs_5z