"""Platform adapters: pluggable readers for player and draft data.

Design constraints this module exists to satisfy:

* **No unauthorised scraping.** Every adapter shipped here reads a file the user
  exported themselves, or generated sample data. No adapter fetches from a
  platform's private endpoints or circumvents its terms.
* **No hard dependency on an external endpoint.** The app is fully usable with
  file-based adapters alone; :class:`HttpJsonAdapter` is an opt-in shell the user
  points at a URL *they* are entitled to use, and it degrades to a clear error
  rather than breaking the app.

Register a new adapter with :func:`register_adapter` and it appears in the UI.
"""

from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import pandas as pd

from core.enums import Platform
from core.validation import ValidationReport

LOGGER = logging.getLogger("fantasy_mock_draft.adapters")


@dataclass(slots=True)
class AdapterResult:
    """A raw table plus provenance and any issues raised while reading it."""

    frame: pd.DataFrame
    source_name: str
    platform: str | None = None
    season: int | None = None
    is_sample_data: bool = False
    report: ValidationReport = field(default_factory=ValidationReport)
    notes: str = ""

    @property
    def ok(self) -> bool:
        return self.report.ok and not self.frame.empty


class PlayerDataAdapter(Protocol):
    """Reads a player pool table from some source."""

    name: str
    description: str
    requires_upload: bool

    def read(self, **kwargs: Any) -> AdapterResult: ...


# ─────────────────────────────────────────────────────────────────────────────
# Spreadsheet reading
# ─────────────────────────────────────────────────────────────────────────────
def read_tabular(
    data: Any,
    *,
    file_name: str = "",
    sheet_name: str | int | None = 0,
    report: ValidationReport | None = None,
) -> tuple[pd.DataFrame, ValidationReport]:
    """Read CSV / TSV / Excel bytes, a path, or a file-like object into a frame.

    Never raises for a bad file: parse failures become validation errors so the
    UI can show the reason next to the upload widget.
    """
    report = report or ValidationReport()
    lowered = (file_name or getattr(data, "name", "") or "").lower()

    try:
        if lowered.endswith((".xlsx", ".xlsm", ".xls")):
            frame = pd.read_excel(data, sheet_name=sheet_name or 0, dtype=object)
            if isinstance(frame, dict):  # sheet_name=None returns every sheet
                first = next(iter(frame))
                report.info(
                    "multi_sheet",
                    f"Workbook has {len(frame)} sheets; read '{first}'.",
                )
                frame = frame[first]
        elif lowered.endswith((".tsv", ".tab")):
            frame = pd.read_csv(data, sep="\t", dtype=object)
        elif lowered.endswith(".json"):
            frame = _frame_from_json(data)
        else:
            frame = pd.read_csv(data, dtype=object, sep=None, engine="python")
    except UnicodeDecodeError:
        report.error(
            "encoding",
            f"Could not decode '{file_name or 'file'}' as text. Re-save it as "
            "UTF-8 CSV or upload the .xlsx directly.",
        )
        return pd.DataFrame(), report
    except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        report.error("parse_failed", f"Could not read '{file_name or 'file'}': {exc}")
        return pd.DataFrame(), report
    except ImportError as exc:
        report.error(
            "missing_reader",
            f"Reading '{file_name}' needs an extra library: {exc}. "
            "Install openpyxl for .xlsx files.",
        )
        return pd.DataFrame(), report

    if frame.empty:
        report.error("empty_file", f"'{file_name or 'file'}' contains no rows.")
    # Drop fully blank rows and unnamed padding columns that Excel adds.
    frame = frame.dropna(how="all")
    keep = [c for c in frame.columns if not str(c).startswith("Unnamed:")]
    dropped = len(frame.columns) - len(keep)
    if dropped:
        report.info("blank_columns", f"Ignored {dropped} unnamed/blank column(s).")
    return frame[keep].reset_index(drop=True), report


def _frame_from_json(data: Any) -> pd.DataFrame:
    """Read a JSON list-of-objects, or an object wrapping such a list."""
    if hasattr(data, "read"):
        raw = data.read()
        payload = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
    elif isinstance(data, (str, bytes)) and str(data).lstrip()[:1] in "[{":
        payload = json.loads(data)
    else:
        with open(data, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

    if isinstance(payload, dict):
        for key in ("players", "picks", "data", "rows", "items", "results"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
        else:
            payload = [payload]
    return pd.DataFrame(payload)


def read_pasted_text(text: str, report: ValidationReport | None = None) -> tuple[pd.DataFrame, ValidationReport]:
    """Parse text pasted into a textarea (CSV, TSV, or whitespace-aligned)."""
    report = report or ValidationReport()
    stripped = (text or "").strip()
    if not stripped:
        report.error("empty_paste", "Nothing was pasted.")
        return pd.DataFrame(), report

    first_line = stripped.splitlines()[0]
    if "\t" in first_line:
        separator: str | None = "\t"
    elif "," in first_line:
        separator = ","
    elif "|" in first_line:
        separator = "|"
    else:
        separator = r"\s{2,}"
    try:
        frame = pd.read_csv(
            io.StringIO(stripped), sep=separator, dtype=object,
            engine="python", skipinitialspace=True,
        )
    except (ValueError, pd.errors.ParserError) as exc:
        report.error("paste_parse_failed", f"Could not parse the pasted text: {exc}")
        return pd.DataFrame(), report
    return frame.dropna(how="all").reset_index(drop=True), report


# ─────────────────────────────────────────────────────────────────────────────
# Concrete adapters
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class FileUploadAdapter:
    """Reads whatever the user uploads. The default, always-available path."""

    name: str = "File upload (CSV / Excel / JSON)"
    description: str = (
        "Upload an export you already have. Column names are matched loosely, "
        "so most platform exports work without editing."
    )
    requires_upload: bool = True
    platform: str | None = None

    def read(
        self,
        data: Any = None,
        *,
        file_name: str = "",
        season: int | None = None,
        platform: str | None = None,
        sheet_name: str | int | None = 0,
        **_: Any,
    ) -> AdapterResult:
        report = ValidationReport()
        if data is None:
            report.error("no_file", "No file was provided.")
            return AdapterResult(pd.DataFrame(), file_name or self.name, report=report)
        frame, report = read_tabular(
            data, file_name=file_name, sheet_name=sheet_name, report=report
        )
        return AdapterResult(
            frame=frame,
            source_name=file_name or "upload",
            platform=platform or self.platform,
            season=season,
            report=report,
        )


@dataclass(slots=True)
class PastedTextAdapter:
    """Reads a block of text pasted into the UI."""

    name: str = "Paste text"
    description: str = "Paste rows copied from a spreadsheet or draft recap page."
    requires_upload: bool = False

    def read(
        self, text: str = "", *, season: int | None = None,
        platform: str | None = None, **_: Any,
    ) -> AdapterResult:
        frame, report = read_pasted_text(text)
        return AdapterResult(
            frame=frame, source_name="pasted text", platform=platform,
            season=season, report=report,
        )


@dataclass(slots=True)
class SampleDataAdapter:
    """Loads the bundled fictional dataset. Always labelled as sample data."""

    name: str = "Bundled sample data"
    description: str = (
        "Fictional players and drafts for trying the app. Clearly labelled — "
        "never presented as real NFL data."
    )
    requires_upload: bool = False
    loader: Callable[[], pd.DataFrame] | None = None
    source_label: str = "sample data"

    def read(self, *, season: int | None = None, **_: Any) -> AdapterResult:
        report = ValidationReport()
        if self.loader is None:
            report.error("no_loader", "This sample adapter has no loader configured.")
            return AdapterResult(pd.DataFrame(), self.source_label, report=report)
        frame = self.loader()
        report.info(
            "sample_data",
            "Loaded bundled SAMPLE DATA — fictional players, not real NFL data.",
        )
        return AdapterResult(
            frame=frame, source_name=self.source_label, season=season,
            is_sample_data=True, report=report,
        )


@dataclass(slots=True)
class HttpJsonAdapter:
    """Optional adapter for a JSON URL the user supplies and is entitled to use.

    Deliberately inert by default: the app never calls out on its own, nothing
    else depends on this class, and a failure produces a validation error rather
    than an exception. The user is responsible for the URL complying with the
    provider's terms.
    """

    name: str = "Custom JSON URL (advanced)"
    description: str = (
        "Point at a JSON endpoint you are permitted to use. Optional — the app "
        "works fully without it, and nothing else depends on it."
    )
    requires_upload: bool = False
    timeout_seconds: float = 10.0

    def read(
        self, url: str = "", *, season: int | None = None,
        platform: str | None = None, **_: Any,
    ) -> AdapterResult:
        report = ValidationReport()
        if not url:
            report.error("no_url", "No URL was provided.")
            return AdapterResult(pd.DataFrame(), "custom url", report=report)
        if not url.lower().startswith("https://"):
            report.error("insecure_url", "Only https:// URLs are accepted.")
            return AdapterResult(pd.DataFrame(), url, report=report)

        try:
            import urllib.request

            request = urllib.request.Request(
                url, headers={"User-Agent": "fantasy-mock-draft/1.0"}
            )
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = response.read()
            frame = _frame_from_json(io.BytesIO(payload))
        except Exception as exc:  # noqa: BLE001 - any network/parse failure is user-facing
            report.error(
                "fetch_failed",
                f"Could not read {url}: {exc}. The rest of the app is unaffected — "
                "upload a file instead.",
            )
            return AdapterResult(pd.DataFrame(), url, report=report)

        report.warn(
            "external_source",
            "Data came from an external URL. Verify you are permitted to use it "
            "and that the values are current.",
        )
        return AdapterResult(
            frame=frame, source_name=url, platform=platform, season=season, report=report
        )


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────
_REGISTRY: dict[str, Any] = {}


def register_adapter(key: str, adapter: Any) -> None:
    """Register an adapter under a stable key (later registration wins)."""
    _REGISTRY[key] = adapter
    LOGGER.debug("Registered adapter '%s' (%s)", key, type(adapter).__name__)


def unregister_adapter(key: str) -> bool:
    """Remove an adapter. Returns whether one was actually removed.

    Exists so a caller that registers an adapter temporarily can undo it —
    :func:`available_adapters` hands back a copy, so popping from that does not
    unregister anything.
    """
    return _REGISTRY.pop(key, None) is not None


def get_adapter(key: str) -> Any | None:
    return _REGISTRY.get(key)


def available_adapters() -> dict[str, Any]:
    """Copy of the registry, so callers cannot mutate it by accident."""
    return dict(_REGISTRY)


def adapter_choices() -> list[tuple[str, str]]:
    """``(key, display name)`` pairs for a selectbox."""
    return [(key, getattr(adapter, "name", key)) for key, adapter in _REGISTRY.items()]


register_adapter("upload", FileUploadAdapter())
register_adapter("paste", PastedTextAdapter())
register_adapter("http_json", HttpJsonAdapter())

# :class:`SampleDataAdapter` is deliberately **not** registered here. The synthetic
# league lives in ``tests/fixtures/sample_league`` and the app has no route to it:
# registering it would put fictional players back into a user-facing dropdown,
# which is precisely what was removed. The class itself remains because the test
# suite registers it with the fixture loader to exercise the sample-data labelling
# path end to end — see ``tests/test_sample_data.py``.


def platform_hint(platform: Platform | str | None) -> str:
    """Short guidance on exporting from a given platform, shown in the UI."""
    resolved = Platform.coerce(platform, Platform.CUSTOM)
    hints = {
        Platform.ESPN: (
            "In ESPN, open your league's draft recap and use the export/print "
            "view, or copy the table and paste it here."
        ),
        Platform.YAHOO: (
            "In Yahoo, the draft results page can be copied directly; paste it "
            "into the text box."
        ),
        Platform.SLEEPER: (
            "Sleeper's draft board can be exported to CSV from the league "
            "history screen."
        ),
        Platform.NFL: "Copy the draft results table from your NFL.com league history.",
        Platform.CBS: "CBS leagues offer a draft results export under League Office.",
        Platform.UNDERDOG: "Underdog best-ball drafts can be exported to CSV.",
    }
    return hints.get(
        resolved,
        "Any CSV or Excel file works. Required columns: season, manager_name, "
        "overall_pick, player_name.",
    )


__all__ = [
    "AdapterResult", "PlayerDataAdapter", "read_tabular", "read_pasted_text",
    "FileUploadAdapter", "PastedTextAdapter", "SampleDataAdapter",
    "HttpJsonAdapter", "register_adapter", "get_adapter", "available_adapters",
    "adapter_choices", "unregister_adapter", "platform_hint",
]
