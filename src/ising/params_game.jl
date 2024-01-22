using AlphaZero
using AlphaZero: CyclicSchedule
using Flux
using AlphaZero: Adam

# TODO: look into SamplesWeighingPolicy in AlphaZero.jl? 
# TODO:  

self_play = SelfPlayParams(
  sim=SimParams(
    num_games=32,
    num_workers=16,
    batch_size=16,
    use_gpu=false,
    # use_gpu=true,
    reset_every=1,# empty MCTS tree every 1 game # TODO???
    flip_probability=0.0,
    alternate_colors=false), mcts=MctsParams(
    num_iters_per_turn=15,
    cpuct=1.0,  # UCT hyperparameter, controls the exploration
    temperature=ConstSchedule(0.0),
    dirichlet_noise_ϵ=0.0,
    dirichlet_noise_α=1.0))

arena = ArenaParams(
  sim=SimParams(
    num_games=32,
    num_workers=16,
    batch_size=16,
    use_gpu=false,
    # use_gpu=true,
    reset_every=1,
    flip_probability=0.0,
    alternate_colors=false),
  mcts=self_play.mcts,
  update_threshold=0.00)

learning = LearningParams(
  use_gpu=false,
  # use_gpu=true,
  use_position_averaging=false, # TODO: dlaczego false???
  samples_weighing_policy=CONSTANT_WEIGHT,
  rewards_renormalization=1,
  l2_regularization=1e-4,
  optimiser=Adam(lr=1e-4),
  batch_size=16,
  loss_computation_batch_size=17,
  nonvalidity_penalty=0.3,
  min_checkpoints_per_epoch=1,
  max_batches_per_checkpoint=5_000,
  num_checkpoints=2)

params = Params(
  arena=arena,
  self_play=self_play,
  learning=learning,
  num_iters=1000,
  memory_analysis=nothing,
  #   ternary_outcome=false,
  use_symmetries=false,
  use_ranked_reward=true,
  ranked_reward_alpha=0.75,
  mem_buffer_size=PLSchedule(80_000))

benchmark_sim = SimParams(
  arena.sim;
  num_games=64*3,
  num_workers=32,
  batch_size=32)


# # Explanation (single-player games)
# + The two competing networks play `sim.num_games` games each.
# + The evaluated network replaces the current best one if its average collected rewards
#   exceeds the average collected reward of the old one by `update_threshold` at least. 

benchmark = [
  Benchmark.Single(
    Benchmark.Full(self_play.mcts),
    benchmark_sim),
  Benchmark.Single(
    Benchmark.NetworkOnly(),
    benchmark_sim)]
