class SolverBase:
    """Abstract base for all solver agents."""
    def act(self, obs): raise NotImplementedError
    def learn(self, *args, **kwargs): raise NotImplementedError
