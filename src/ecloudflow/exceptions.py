"""Typed exception hierarchy for ECloudFlow contracts and workflows."""


class ECloudFlowError(Exception):
    """Base exception for failures raised by ECloudFlow."""


class ContractValidationError(ECloudFlowError, ValueError):
    """Raise when a typed tensor contract violates a required invariant."""


class CoordinateFrameError(ContractValidationError):
    """Raise when a coordinate frame is malformed or cannot transform points."""


class FragmentInvariantError(ECloudFlowError, ValueError):
    """Raise when a fragment condition cannot be exactly applied to a state."""


class DataValidationError(ECloudFlowError, ValueError):
    """Raise when a source structure cannot form a valid ECloudFlow sample."""
