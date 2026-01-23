"""Exception classes for Interactive Topic Model."""


class ITMError(RuntimeError):
    """Base exception for ITM errors."""
    pass


class IdentityError(ITMError):
    """Error related to doc_id identity issues."""
    pass


class NotFittedError(ITMError):
    """Error when model is not fitted but operation requires fitting."""
    pass
