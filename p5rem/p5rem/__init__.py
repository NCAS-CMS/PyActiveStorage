"""p5rem package."""

# Import cache optionally - it requires diskcache which may not be available
# on remote servers where only the stub is needed
try:
	from . import cache
except ImportError:
	cache = None  # type: ignore

# Import bootstrap optionally - it requires paramiko which is only needed
# on the client side for remote SSH operations
try:
	from .bootstrap import BootstrappedProcess, BootstrapError, ReconnectingBootstrappedSession, bootstrap_reconnecting_session, bootstrap_server, bootstrap_session, discover_remote_conda_envs
except ImportError:
	BootstrappedProcess = None  # type: ignore
	BootstrapError = None  # type: ignore
	ReconnectingBootstrappedSession = None  # type: ignore
	bootstrap_reconnecting_session = None  # type: ignore
	bootstrap_server = None  # type: ignore
	bootstrap_session = None  # type: ignore
	discover_remote_conda_envs = None  # type: ignore

from .proxy import rDataset, rFile
from .session import Session, p5remSession

__all__ = [
	"BootstrappedProcess",
	"BootstrapError",
	"ReconnectingBootstrappedSession",
	"Session",
	"bootstrap_reconnecting_session",
	"bootstrap_server",
	"bootstrap_session",
	"cache",
	"discover_remote_conda_envs",
	"p5remSession",
	"rDataset",
	"rFile",
]