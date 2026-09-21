"""Data ingestion, synthetic generation and preprocessing."""

from .loader import (
    DatasetUnavailableError,
    UCIDatasetLoader,
    build_reopen_target,
    collapse_event_log,
    filter_terminal_states,
    load_itsm_dataset,
)
from .preprocessor import (
    PRIORITY_ORDER,
    SchemaValidationError,
    TicketPreprocessor,
    TicketRecord,
    encode_priority_triad,
)
from .synthetic import (
    SyntheticTicketGenerator,
    generate_synthetic_tickets,
    write_synthetic_dataset,
)

__all__ = [
    "DatasetUnavailableError",
    "UCIDatasetLoader",
    "build_reopen_target",
    "collapse_event_log",
    "filter_terminal_states",
    "load_itsm_dataset",
    "PRIORITY_ORDER",
    "SchemaValidationError",
    "TicketPreprocessor",
    "TicketRecord",
    "encode_priority_triad",
    "SyntheticTicketGenerator",
    "generate_synthetic_tickets",
    "write_synthetic_dataset",
]
