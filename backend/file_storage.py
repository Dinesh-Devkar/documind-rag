"""Filesystem-only utilities (no Streamlit UI calls)."""

from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .config import ALLOWED_EXTENSIONS


def save_uploaded_files(
    uploaded_files: list[Any], destination_dir: Path
) -> list[Path]:
    """Write selected Streamlit UploadedFile objects to disk; return local paths.

    Called by app.py only when the uploaded-file fingerprint changes.
    Every stored name is unique, and a partial batch is removed on failure.

    Args:
        uploaded_files: Streamlit UploadedFile objects, each with name/getvalue().
        destination_dir: Directory that will hold this browser session's uploads.

    Returns:
        Local Path objects suitable for PyPDFLoader(str(path)), etc.

    Raises:
        ValueError: Extension/name is not allowed.
        OSError: File cannot be written.
    """

    print('Save_File_Path Called')
    if not uploaded_files:
        return []

    # Validate the entire batch before writing anything.
    cleaned_names: list[str] = []
    for uploaded_file in uploaded_files:
        # Handle both POSIX (/) and Windows (\\) separators safely.
        filename = str(uploaded_file.name).replace("\\", "/").split("/")[-1]
        if not filename or filename in {".", ".."}:
            raise ValueError("Invalid uploaded filename.")
        extension = Path(filename).suffix.lower().lstrip(".")
        if extension not in ALLOWED_EXTENSIONS:
            raise ValueError(f"File type '.{extension}' is not allowed.")
        cleaned_names.append(filename)

    destination_dir.mkdir(parents=True, exist_ok=True)
    paths_written: list[Path] = []
    try:
        for uploaded_file, filename in zip(uploaded_files, cleaned_names):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            unique_id = uuid4().hex[:8]
            local_path = destination_dir / f"{timestamp}_{unique_id}_{filename}"

            # Using "xb" prevents accidental overwrites even if IDs collide.
            with local_path.open("xb") as output_file:
                paths_written.append(local_path)
                output_file.write(uploaded_file.getvalue())

        return paths_written
    except Exception:
        # Avoid leaving half-saved upload batches on disk.
        for local_path in paths_written:
            try:
                local_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise
