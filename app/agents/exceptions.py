"""Custom domain exceptions for the Enterprise Multi-Agent Orchestration Engine (`benzene-rag`)."""


class MissionException(Exception):
    """Base exception for all agent mission failures."""


class SecurityException(MissionException):
    """Raised when a prompt injection, jailbreak, or unsafe payload is detected."""


class MissionNotFound(MissionException):
    """Raised when attempting to load or approve a non-existent mission snapshot."""


class MissionAlreadyExecuted(MissionException):
    """Raised when attempting to re-execute or approve a mission that has already been completed."""


class MissionRejected(MissionException):
    """Raised when a Human-in-the-Loop (HITL) approval request is rejected."""


class InvalidMissionState(MissionException):
    """Raised when an operation is attempted on a mission in an invalid state."""
