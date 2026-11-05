"""Execution: a Broker interface, the PaperBroker, guarded live adapters, and the Executor that
turns an approved book into orders. Live adapters refuse to send anything unless four
independent locks are open (settings, env flag, env keys, confirmation file)."""

from .broker import Broker, Fill, LiveDisabled, Order, OrderState, live_guard  # noqa: F401
from .executor import ExecutionReport, Executor  # noqa: F401
from .paper import PaperBroker  # noqa: F401
