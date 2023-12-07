using AlphaZero
using AlphaZero: CyclicSchedule
Network = NetLib.SimpleNet

netparams = NetLib.SimpleNetHP(
  width=64,
  depth_common=6,
  use_batch_norm=false)

self_play = SelfPlayParams(
  sim=SimParams(
    num_games=256,
    num_workers=1,
    batch_size=1,
    use_gpu=false,
    reset_every=2,# empty MCTS tree every 1 game
    flip_probability=0.,
    alternate_colors=false),

  mcts=MctsParams(
    num_iters_per_turn=15,
    cpuct=1.0,  # UCT hyperparameter, controls the exploration
    temperature=ConstSchedule(0.),
    dirichlet_noise_ϵ=0.,
    dirichlet_noise_α=1.))

arena = ArenaParams(
  sim=SimParams(
    num_games=256,
    num_workers=1,
    batch_size=1,
    use_gpu=false,
    reset_every=1,
    flip_probability=0.,
    alternate_colors=false),
  mcts = self_play.mcts,
  update_threshold=0.00)

learning = LearningParams(
  use_gpu=false,
  use_position_averaging=false,
  samples_weighing_policy=CONSTANT_WEIGHT,
  rewards_renormalization=1,
  l2_regularization=1e-4,
  optimiser=Adam(lr=1e-3),
  batch_size=1,
  loss_computation_batch_size=1,
  nonvalidity_penalty=1.,
  min_checkpoints_per_epoch=1,
  max_batches_per_checkpoint=5_000,
  num_checkpoints=2)

  params = Params(
  arena=arena,
  self_play=self_play,
  learning=learning,
  num_iters=40,
  memory_analysis=nothing,
#   ternary_outcome=false,
  use_symmetries=false,
  mem_buffer_size=PLSchedule(80_000))

benchmark_sim = SimParams(
  arena.sim;
  num_games=224,
  num_workers=1,
  batch_size=1)


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
