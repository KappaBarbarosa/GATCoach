export CUDA_VISIBLE_DEVICES=1
export WANDB_API_KEY=56e66d4501a9a1e27100a7e2495c23eada3a5755

python3 src/main.py --config=qmix_coach_embedding_gnn --env-config=sc2 with env_args.map_name=3s_vs_5z

# python3 src/main.py --config=qmix_coach_embedding --env-config=sc2 with env_args.map_name=2s3z
# python3 src/main.py --config=qmix_coach_embedding --env-config=sc2 with env_args.map_name=2s3z
# python3 src/main.py --config=qmix_coach_embedding --env-config=sc2 with env_args.map_name=2s3z