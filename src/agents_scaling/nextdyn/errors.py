"""Typed failures of the nextdyn package.

Every refusal in nextdyn is a subclass of :class:`NextdynError`, so a caller can tell a
contract problem from an infrastructure one without parsing messages.  The classes exist
because ARD36 raised ``ValueError``/``ProtocolError`` for everything and the lane drivers
could not distinguish "the manifest is wrong" (fix and rerun) from "the dependency is not
here yet" (re-queue).  Plan §1.2 (``errors.py``), §3.4 (``EXIT_DEPS_UNMET`` is a
:class:`DependencyUnmet`, never a failure).
"""

from __future__ import annotations


class NextdynError(Exception):
    """Base class of every nextdyn refusal."""


class ConfigError(NextdynError):
    """A YAML config the strict loader does not admit (unknown key, missing key, bad type)."""


class ContractViolation(NextdynError):
    """A compiled configuration that contradicts the spec (17.4) or the plan.

    Instances are also *returned* by :func:`agents_scaling.nextdyn.config.compile_checks`
    so a caller can list every violation at once instead of fixing them one raise at a time.
    """


class IdentityError(NextdynError):
    """An id that would alias two different observations, or a key with a null component."""


class ManifestError(NextdynError):
    """A cell manifest or item file that the frozen design does not admit."""


class DependencyUnmet(NextdynError):
    """A ``cell:``/``bundle:``/``register:`` dependency that is absent or tampered."""


class LeakageError(NextdynError):
    """Gold, a composition label or a correctness cue reached a generation-side object."""


class MaterializeError(NextdynError):
    """A table projection or key assertion failed at materialize time."""


class LockError(NextdynError):
    """A lock file is missing, null where a value is required, or drifted from disk."""




class LockSourceMissing(LockError):
    """A declared input is simply not on this filesystem yet.

    Distinguished from every other :class:`LockError` because the two need different actions and
    the spec (17.5) makes ``lock-inputs`` report them separately: a *missing* source is a launch
    blocker ("fetch it, then re-run"), while a source that is present but wrong -- a revision
    that drifted, a checksum that does not match, a corrupt file -- is a failure that must be
    understood before anything is fetched.  Collapsing the two is how a run ends up fetching
    over a real integrity problem.
    """

__all__ = [
    "ConfigError", "ContractViolation", "DependencyUnmet", "IdentityError", "LeakageError",
    "LockError", "ManifestError", "MaterializeError", "NextdynError",
]
