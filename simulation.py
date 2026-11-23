import numpy as np

# denom floor
THETA_ERROR_EPSILON = 1e-12


class DistributedRegression:

    def __init__(
        self,
        X_seen_aug,
        y_seen,
        X_global_aug,
        y_global,
        cond_threshold,
        gossip_mode="coordinate",
        stop_on_theta_change=False,
        epsilon=1e-3,
        r=20,
        q=1,
    ):

        self.X_seen_aug = X_seen_aug
        self.y_seen = y_seen

        self.N = len(y_seen)

        self.cond_threshold = cond_threshold

        if gossip_mode not in {"block", "coordinate", "full"}:
            raise ValueError(
                "gossip_mode must be 'block', 'coordinate', or 'full'"
            )

        self.gossip_mode = gossip_mode

        # NODE STOPPING BEHAVIOR

        self.stop_on_theta_change = stop_on_theta_change
        self.epsilon = epsilon
        self.r = int(r)

        if self.epsilon <= 0:
            raise ValueError("epsilon must be > 0")
        if self.r < 1 or self.r != r:
            raise ValueError("r must be a positive integer")

        self.active = np.ones(
            self.N,
            dtype=bool,
        )
        self.convergence_counter = np.zeros(
            self.N,
            dtype=int,
        )

        # GLOBAL LEAST-SQUARES SOLUTION

        self.theta_global = np.linalg.lstsq(
            X_global_aug,
            y_global,
            rcond=None,
        )[0]

        # SEEN-DATA LEAST-SQUARES SOLUTION

        self.theta_seen = np.linalg.lstsq(
            X_seen_aug,
            y_seen,
            rcond=None,
        )[0]

        # LOCAL SUFFICIENT STATISTICS

        self.agents_M = np.array([
            np.outer(z, z)
            for z in X_seen_aug
        ])

        self.agents_B = np.array([
            z * yi
            for z, yi in zip(
                X_seen_aug,
                y_seen,
            )
        ])

        self.p = self.agents_B.shape[1]

        # vech(M) convention
        self.triu_indices = np.array(
            [(a, b) for a in range(self.p) for b in range(a, self.p)],
            dtype=int,
        )
        self.num_M_coords = len(self.triu_indices)   # p * (p + 1) // 2

        self.num_B_coords = self.p

        self.total_coords = self.num_M_coords + self.num_B_coords

        if isinstance(q, (bool, np.bool_)) or not isinstance(q, (int, np.integer)):
            raise ValueError("q must be an integer between 1 and total_coords")
        if not 1 <= q <= self.total_coords:
            raise ValueError(
                f"q must be between 1 and total_coords ({self.total_coords})"
            )

        self.q = self.total_coords if gossip_mode == "full" else int(q)

        self.M_global = np.mean(self.agents_M, axis=0)
        self.B_global = np.mean(self.agents_B, axis=0)

        self.w_initial_sum = np.concatenate([
            self.agents_M.reshape(self.N, -1),
            self.agents_B
        ], axis=1).sum(axis=0)

        initial_w_flat = np.concatenate([
            self.agents_M.reshape(self.N, -1),
            self.agents_B
        ], axis=1)

        self.l1_history = []
        self.l2_history = []
        self.linf_history = []


    def gossip_step(self):

        if self.stop_on_theta_change:

            active_nodes = np.where(
                self.active
            )[0]

            if len(active_nodes) == 0:
                return

        initiator, responder = np.random.choice(
            self.N,
            2,
            replace=False,
        )

        self.interact(
            initiator,
            responder,
        )


    def interact(
        self,
        initiator,
        responder,
    ):

        old_B_initiator = (
            self.agents_B[initiator].copy()
        )

        old_B_responder = (
            self.agents_B[responder].copy()
        )


        old_theta_initiator = (
            self._compute_single_theta(
                initiator
            )
        )

        old_theta_responder = (
            self._compute_single_theta(
                responder
            )
        )

        self._block_protocol(initiator, responder)

        self._record_B_change(
            initiator,
            responder,
            old_B_initiator,
            old_B_responder,
        )

        # CHECK HEURISTIC STOPPING RULE

        if self.stop_on_theta_change:

            new_theta_initiator = self._compute_single_theta(initiator)
            new_theta_responder = self._compute_single_theta(responder)

            self._update_stopping_status(
                initiator, responder, old_theta_initiator, new_theta_initiator
            )
            self._update_stopping_status(
                responder, initiator, old_theta_responder, new_theta_responder
            )

    def all_inactive(self):
        """Return True if every agent has been deactivated by the heuristic rule."""
        return not np.any(self.active)

    def _compute_single_theta(
        self,
        agent,
    ):

        M = self.agents_M[agent]
        B = self.agents_B[agent]

        c = np.linalg.cond(M)

        return self._solve_if_usable(M, B, c)

    def _solve_if_usable(self, M, B, condition):
        if self.cond_threshold and not condition < self.cond_threshold:
            return None
        try:
            theta = np.linalg.solve(M, B)
        except np.linalg.LinAlgError:
            return None
        return theta if np.isfinite(theta).all() else None

    # HEURISTIC STOPPING RULE

    def _normalized_theta_distance(self, theta_i, theta_j):
        if theta_i is None or theta_j is None:
            return np.inf
        denominator = max(
            1.0,
            np.linalg.norm(theta_i, ord=2),
            np.linalg.norm(theta_j, ord=2),
        )
        return np.linalg.norm(theta_i - theta_j, ord=2) / denominator

    def _normalized_theta_change(self, old_theta, new_theta):
        if old_theta is None or new_theta is None:
            return np.inf
        denominator = max(1.0, np.linalg.norm(new_theta, ord=2))
        return np.linalg.norm(new_theta - old_theta, ord=2) / denominator

    def _update_stopping_status(
        self, agent, other_agent, old_theta, new_theta
    ):
        other_theta = self._compute_single_theta(other_agent)
        d_ij = self._normalized_theta_distance(new_theta, other_theta)
        s_i = self._normalized_theta_change(old_theta, new_theta)

        if d_ij < self.epsilon and s_i < self.epsilon:
            self.convergence_counter[agent] += 1
            if self.convergence_counter[agent] >= self.r:
                self.active[agent] = False
        else:
            self.convergence_counter[agent] = 0
            self.active[agent] = True


    def _block_protocol(self, initiator, responder):

        if self.q == self.total_coords:
            coords = np.arange(self.total_coords)
        elif self.q == 1:
            coords = np.array([np.random.randint(self.total_coords)])
        else:
            coords = np.random.choice(self.total_coords, size=self.q,
                                      replace=False)

        matrix_coords = coords[coords < self.num_M_coords]
        if matrix_coords.size:
            a, b = self.triu_indices[matrix_coords].T
            avg = 0.5 * (
                self.agents_M[initiator, a, b]
                + self.agents_M[responder, a, b]
            )
            self.agents_M[initiator, a, b] = avg
            self.agents_M[responder, a, b] = avg
            # Update both symmetric entries of every selected matrix pair.
            self.agents_M[initiator, b, a] = avg
            self.agents_M[responder, b, a] = avg

        vector_coords = coords[coords >= self.num_M_coords] - self.num_M_coords
        if vector_coords.size:
            avg = 0.5 * (
                self.agents_B[initiator, vector_coords]
                + self.agents_B[responder, vector_coords]
            )
            self.agents_B[initiator, vector_coords] = avg
            self.agents_B[responder, vector_coords] = avg

    # METRIC COLLECTION

    def _record_B_change(
        self,
        initiator,
        responder,
        old_B_initiator,
        old_B_responder,
    ):

        diff_initiator = (
            self.agents_B[initiator]
            - old_B_initiator
        )

        diff_responder = (
            self.agents_B[responder]
            - old_B_responder
        )

        self.l1_history.append(
            (
                np.linalg.norm(
                    diff_initiator,
                    ord=1,
                )
                +
                np.linalg.norm(
                    diff_responder,
                    ord=1,
                )
            )
            / self.N
        )

        self.l2_history.append(
            (
                np.linalg.norm(
                    diff_initiator,
                    ord=2,
                )
                +
                np.linalg.norm(
                    diff_responder,
                    ord=2,
                )
            )
            / self.N
        )

        self.linf_history.append(
            (
                np.linalg.norm(
                    diff_initiator,
                    ord=np.inf,
                )
                +
                np.linalg.norm(
                    diff_responder,
                    ord=np.inf,
                )
            )
            / self.N
        )


    def compute_theta_agents(self):

        theta_agents = []
        conds = []
        stable = []

        for M, B in zip(
            self.agents_M,
            self.agents_B,
        ):

            c = np.linalg.cond(M)

            conds.append(c)

            theta = self._solve_if_usable(M, B, c)
            stable.append(theta is not None)
            if theta is None:
                theta = np.full(M.shape[0], np.nan)
            theta_agents.append(theta)

        return (
            np.array(theta_agents),
            np.array(conds),
            np.array(stable),
        )


    def relative_theta_error(
        self,
        theta_agents,
    ):

        valid = theta_agents[
            np.isfinite(theta_agents).all(axis=1)
        ]

        if len(valid) == 0:
            return np.nan

        denom = max(np.linalg.norm(self.theta_seen), THETA_ERROR_EPSILON)

        return np.mean(
            np.linalg.norm(
                valid - self.theta_seen,
                axis=1,
            )
            / denom
        )

    # MATRIX ERROR

    def relative_matrix_error(
        self,
        matrices,
    ):

        if len(matrices) == 0:
            return np.nan

        denom = np.linalg.norm(
            self.M_global,
            ord="fro",
        )

        return np.mean([
            np.linalg.norm(
                M - self.M_global,
                ord="fro",
            )
            / denom
            for M in matrices
        ])