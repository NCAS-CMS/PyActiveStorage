"""p5rem package."""

from importlib import import_module

# Import cache optionally - it requires diskcache which may not be available
# on remote servers where only the stub is needed
try:
	from . import cache
except ImportError:
	cache = None  # type: ignore

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

_EXPORTS = {
	"BootstrappedProcess": ("activestorage.bootstrap", "BootstrappedProcess"),
	"BootstrapError": ("activestorage.bootstrap", "BootstrapError"),
	"ReconnectingBootstrappedSession": ("activestorage.bootstrap", "ReconnectingBootstrappedSession"),
	"Session": ("activestorage.session", "Session"),
	"bootstrap_reconnecting_session": ("activestorage.bootstrap", "bootstrap_reconnecting_session"),
	"bootstrap_server": ("activestorage.bootstrap", "bootstrap_server"),
	"bootstrap_session": ("activestorage.bootstrap", "bootstrap_session"),
	"discover_remote_conda_envs": ("activestorage.bootstrap", "discover_remote_conda_envs"),
	"p5remSession": ("activestorage.session", "p5remSession"),
	"rDataset": ("activestorage.remote", "rDataset"),
	"rFile": ("activestorage.remote", "rFile"),
}


def __getattr__(name: str):
	if name == "cache":
		return cache
	try:
		module_name, attr_name = _EXPORTS[name]
	except KeyError as exc:
		raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
	module = import_module(module_name)
	value = getattr(module, attr_name)
	globals()[name] = value
	return value
