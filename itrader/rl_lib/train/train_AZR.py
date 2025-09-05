# Hyperparameters (to be evolved via PBT)
alpha_actor      = 1e-3      # actor learning rate
alpha_critic     = 5e-4      # critic learning rate (usually lower)
beta_entropy     = 1e-2      # entropy weight
gamma_discount   = 0.99      # future reward discount
lambda_GAE       = 0.95      # GAE smoothing
lambda_role      = 0.8       # weight of solver vs. proposer reward
batch_size       = 64        # # proposals+solves per update
K_rollouts       = 32        # # solver attempts per proposed task

# Buffers & data
historical_data = load_market_data()   # your 2-year OHLCV
valid_windows   = compute_valid_indices(historical_data, window_size=N)

def sample_window():
    idx = random.choice(valid_windows)
    return historical_data[idx:idx+N]

# Core loop
for iteration in range(num_iterations):
    proposals = []
    # ——— Proposer Phase ———
    for _ in range(batch_size):
        s_prop = sample_window()
        # embed with LLM + proposer head
        task_id, start_idx, start_logp = proposer.sample(s_prop)  
        # define the window as a “task”
        window = historical_data[start_idx : start_idx+N]
        # Collect solver performance over K rollouts
        solve_rewards = []
        for _ in range(K_rollouts):
            # Solver generates action trajectory
            actions, solve_logps = solver.sample(window)
            # Compute r_solve for this rollout
            r_s = compute_r_solve(actions, window)
            solve_rewards.append(r_s)
        # Empirical solver success rate
        p_solve = sum(solve_rewards) / K_rollouts
        # proposer reward: inverted parabola centered at 0.5
        r_prop = 1 - ((p_solve - 0.5)/0.5)**2    # ∈[-3,1] if scaled like AZR
        proposals.append({
            "s_prop": s_prop,
            "task": window,
            "logp_prop": start_logp,
            "p_solve": p_solve,
            "r_prop": r_prop,
            "solver_rollouts": solve_rewards
        })

    # ——— Collect Trajectories & Rewards ———
    trajectories = []
    for item in proposals:
        window = item["task"]
        # One canonical solve rollout to get final action and logp
        a_solve, logp_s = solver.sample(window)
        r_solve = compute_r_solve(a_solve, window)
        trajectories.append({
            **item,
            "action_solve": a_solve,
            "logp_solve": logp_s,
            "r_solve": r_solve
        })

    # ——— Advantage Estimation (Actor‐Critic + GAE) ———
    actor_loss = 0.0
    critic_loss = 0.0
    entropy_loss = 0.0

    # For each trajectory, build pseudo‐episode of length 1 (task as “state”)
    for traj in trajectories:
        # Critic values
        V_s     = critic(traj["task"])
        V_next  = critic(traj["task"])  # same, since single‐step
        # TD error
        delta_solve = traj["r_solve"] + gamma_discount * V_next - V_s
        delta_prop  = traj["r_prop"]  + gamma_discount * V_next - V_s

        # GAE advantages (single‐step here)
        A_solve = delta_solve
        A_prop  = delta_prop

        # Normalize advantages per (task‐family, role)
        A_solve_norm = normalize(A_solve, family="00", role="solve")
        A_prop_norm  = normalize(A_prop,  family="00", role="prop")

        # Actor losses
        actor_loss += -traj["logp_prop"]  * A_prop_norm * (1 - lambda_role)
        actor_loss += -traj["logp_solve"] * A_solve_norm * lambda_role

        # Critic losses (MSE)
        critic_loss += delta_solve.pow(2) * 0.5 + delta_prop.pow(2) * 0.5

        # Entropy regularization
        entropy_loss += -beta_entropy * (
            proposer.entropy(traj["s_prop"]) +
            solver.entropy(traj["task"])
        )

    # ——— Backward & Optimize ———
    optimizer_actor.zero_grad()
    (actor_loss + entropy_loss).backward(retain_graph=True)
    optimizer_actor.step(lr=alpha_actor)

    optimizer_critic.zero_grad()
    critic_loss.backward()
    optimizer_critic.step(lr=alpha_critic)
