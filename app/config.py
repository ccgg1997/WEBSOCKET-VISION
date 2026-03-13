from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


SERVICE_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(SERVICE_ROOT / ".env")


def _env(key: str, default: str = "") -> str:
    value = os.getenv(key, "").strip()
    return value if value else default


def _env_int(key: str, default: int) -> int:
    raw = _env(key)
    return int(raw) if raw else default


def _env_float(key: str, default: float) -> float:
    raw = _env(key)
    return float(raw) if raw else default


def _env_bool(key: str, default: bool) -> bool:
    raw = _env(key).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "si"}


def _env_csv(key: str, default: str = "") -> tuple[str, ...]:
    raw = _env(key, default)
    if not raw:
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _env_mapping_int(key: str) -> dict[str, int]:
    raw = _env(key)
    if not raw:
        return {}
    mapping: dict[str, int] = {}
    for item in raw.split(","):
        part = item.strip()
        if not part or ":" not in part:
            continue
        source, value = part.split(":", 1)
        source_key = source.strip()
        value_raw = value.strip()
        if not source_key or not value_raw:
            continue
        mapping[source_key] = int(value_raw)
    return mapping


def _env_mapping_str(key: str) -> dict[str, str]:
    raw = _env(key)
    if not raw:
        return {}
    mapping: dict[str, str] = {}
    for item in raw.split(","):
        part = item.strip()
        if not part or ":" not in part:
            continue
        source, value = part.split(":", 1)
        source_key = source.strip()
        value_raw = value.strip()
        if not source_key or not value_raw:
            continue
        mapping[source_key] = value_raw
    return mapping


def _resolve_relative_path(raw: str) -> Path:
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return candidate
    return (SERVICE_ROOT / candidate).resolve()


def _model_env_token(model_id: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", model_id).strip("_")
    return normalized.upper()


def _fallback_model_path(model_id: str) -> str:
    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("._-") or "model"
    return f"runtime/models/{safe_id}.pt"


def _sqlite_url_for(path: Path) -> str:
    return f"sqlite+pysqlite:///{path.as_posix()}"


def _normalize_database_url(raw: str) -> str:
    lowered = raw.lower()
    if lowered.startswith("postgresql+"):
        return raw
    if lowered.startswith("postgresql://"):
        return f"postgresql+psycopg://{raw[len('postgresql://'):]}"
    if lowered.startswith("postgres://"):
        return f"postgresql+psycopg://{raw[len('postgres://'):]}"
    return raw


def _coerce_bool(value: Any, default: bool | None = None) -> bool | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if not normalized:
        return default
    return normalized in {"1", "true", "yes", "si"}


def _parse_string_items(
    value: Any,
    *,
    field_name: str,
    context: str,
) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    if isinstance(value, (list, tuple)):
        items: list[str] = []
        for raw_item in value:
            item = str(raw_item).strip()
            if item:
                items.append(item)
        return tuple(items)
    raise ValueError(f"{context}: '{field_name}' debe ser string o lista.")


def _parse_string_mapping(
    value: Any,
    *,
    field_name: str,
    context: str,
) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{context}: '{field_name}' debe ser un objeto JSON.")
    mapping: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key).strip()
        mapped_value = str(raw_value).strip()
        if key and mapped_value:
            mapping[key] = mapped_value
    return mapping


@dataclass(frozen=True)
class ModelDefinition:
    model_id: str
    display_name: str
    path_raw: str = ""
    url: str = ""
    sha256: str = ""
    fallback_path_raw: str = "runtime/model.pt"
    selector_raw: str = ""
    selection_aliases: tuple[str, ...] = ()
    fill_events_enabled: bool | None = None
    fill_event_storage_prefix_raw: str = ""

    def resolve_path(self, service_root: Path) -> Path:
        candidate = Path(self.path_raw or self.fallback_path_raw).expanduser()
        if candidate.is_absolute():
            return candidate
        return (service_root / candidate).resolve()

    def resolve_fill_event_storage_prefix(self, default_prefix: str) -> str:
        prefix = self.fill_event_storage_prefix_raw.strip()
        return prefix or default_prefix


@dataclass(frozen=True)
class ModelCatalog:
    definitions: dict[str, ModelDefinition]
    default_model_id: str = ""
    selection_aliases: dict[str, str] = field(default_factory=dict)
    has_explicit_fill_event_models: bool = False


@dataclass(frozen=True)
class ServiceSettings:
    service_root: Path = field(default_factory=lambda: SERVICE_ROOT)

    host: str = field(default_factory=lambda: _env("YOLO_WS_HOST", "0.0.0.0"))
    port: int = field(
        default_factory=lambda: int(os.getenv("PORT", _env("YOLO_WS_PORT", "8000")))
    )
    log_level: str = field(default_factory=lambda: _env("YOLO_WS_LOG_LEVEL", "INFO"))

    auth_username: str = field(default_factory=lambda: _env("YOLO_WS_AUTH_USERNAME", ""))
    auth_password: str = field(default_factory=lambda: _env("YOLO_WS_AUTH_PASSWORD", ""))
    auth_password_hash: str = field(
        default_factory=lambda: _env("YOLO_WS_AUTH_PASSWORD_HASH", "")
    )
    auth_timeout_seconds: int = field(
        default_factory=lambda: _env_int("YOLO_WS_AUTH_TIMEOUT_SECONDS", 10)
    )

    model_ids: tuple[str, ...] = field(default_factory=lambda: _env_csv("YOLO_WS_MODEL_IDS"))
    default_model_id_raw: str = field(
        default_factory=lambda: _env("YOLO_WS_DEFAULT_MODEL_ID", "")
    )
    models_file_raw: str = field(default_factory=lambda: _env("YOLO_WS_MODELS_FILE", ""))
    model_selection_aliases: dict[str, str] = field(
        default_factory=lambda: _env_mapping_str("YOLO_WS_MODEL_SELECTION_ALIASES")
    )
    model_path_raw: str = field(default_factory=lambda: _env("YOLO_WS_MODEL_PATH", ""))
    model_url: str = field(default_factory=lambda: _env("YOLO_WS_MODEL_URL", ""))
    model_sha256: str = field(default_factory=lambda: _env("YOLO_WS_MODEL_SHA256", ""))
    model_download_timeout: int = field(
        default_factory=lambda: _env_int("YOLO_WS_MODEL_DOWNLOAD_TIMEOUT", 120)
    )
    device: str = field(default_factory=lambda: _env("YOLO_WS_DEVICE", "auto"))
    torch_threads: int = field(
        default_factory=lambda: _env_int("YOLO_WS_TORCH_THREADS", max(os.cpu_count() or 1, 1))
    )

    conf_threshold: float = field(
        default_factory=lambda: _env_float("YOLO_WS_CONF_THRESHOLD", 0.35)
    )
    iou_threshold: float = field(
        default_factory=lambda: _env_float("YOLO_WS_IOU_THRESHOLD", 0.45)
    )
    imgsz: int = field(default_factory=lambda: _env_int("YOLO_WS_IMGSZ", 640))
    max_det: int = field(default_factory=lambda: _env_int("YOLO_WS_MAX_DET", 100))
    max_frame_bytes: int = field(
        default_factory=lambda: _env_int("YOLO_WS_MAX_FRAME_BYTES", 4 * 1024 * 1024)
    )
    jpeg_quality: int = field(default_factory=lambda: _env_int("YOLO_WS_JPEG_QUALITY", 85))
    default_return_image: bool = field(
        default_factory=lambda: _env_bool("YOLO_WS_RETURN_IMAGE", False)
    )
    events_enabled: bool = field(
        default_factory=lambda: _env_bool("YOLO_WS_EVENTS_ENABLED", True)
    )
    trigger_labels: tuple[str, ...] = field(
        default_factory=lambda: _env_csv("YOLO_WS_TRIGGER_LABELS", "25")
    )
    label_percent_aliases: dict[str, int] = field(
        default_factory=lambda: _env_mapping_int("YOLO_WS_LABEL_PERCENT_ALIASES")
    )
    clip_seconds: int = field(
        default_factory=lambda: _env_int("YOLO_WS_CLIP_SECONDS", 5)
    )
    event_cooldown_seconds: int = field(
        default_factory=lambda: _env_int("YOLO_WS_EVENT_COOLDOWN_SECONDS", 5)
    )
    event_record_fps: float = field(
        default_factory=lambda: _env_float("YOLO_WS_EVENT_RECORD_FPS", 5.0)
    )
    snapshot_interval_seconds: int = field(
        default_factory=lambda: _env_int("YOLO_WS_SNAPSHOT_INTERVAL_SECONDS", 10)
    )
    snapshot_save_empty: bool = field(
        default_factory=lambda: _env_bool("YOLO_WS_SNAPSHOT_SAVE_EMPTY", True)
    )
    telegram_enabled: bool = field(
        default_factory=lambda: _env_bool("YOLO_WS_TELEGRAM_ENABLED", False)
    )
    telegram_bot_token: str = field(
        default_factory=lambda: _env("YOLO_WS_TELEGRAM_BOT_TOKEN", "")
    )
    telegram_chat_id: str = field(
        default_factory=lambda: _env("YOLO_WS_TELEGRAM_CHAT_ID", "")
    )
    telegram_model_ids: tuple[str, ...] = field(
        default_factory=lambda: _env_csv("YOLO_WS_TELEGRAM_MODEL_IDS", "cana")
    )
    telegram_fill_threshold: int = field(
        default_factory=lambda: _env_int("YOLO_WS_TELEGRAM_FILL_THRESHOLD", 50)
    )
    telegram_timeout_seconds: int = field(
        default_factory=lambda: _env_int("YOLO_WS_TELEGRAM_TIMEOUT_SECONDS", 10)
    )
    fill_event_storage_prefix: str = field(
        default_factory=lambda: _env("YOLO_WS_FILL_EVENT_STORAGE_PREFIX", "cana")
    )
    database_url_raw: str = field(
        default_factory=lambda: _env("YOLO_WS_DATABASE_URL", "")
    )
    db_path_raw: str = field(
        default_factory=lambda: _env("YOLO_WS_DB_PATH", "runtime/service.db")
    )
    temp_dir_raw: str = field(
        default_factory=lambda: _env("YOLO_WS_TEMP_DIR", "runtime/tmp")
    )
    storage_backend: str = field(
        default_factory=lambda: _env("YOLO_WS_STORAGE_BACKEND", "local").lower()
    )
    minio_endpoint: str = field(
        default_factory=lambda: _env("YOLO_WS_MINIO_ENDPOINT", "")
    )
    minio_access_key: str = field(
        default_factory=lambda: _env("YOLO_WS_MINIO_ACCESS_KEY", "")
    )
    minio_secret_key: str = field(
        default_factory=lambda: _env("YOLO_WS_MINIO_SECRET_KEY", "")
    )
    minio_bucket: str = field(
        default_factory=lambda: _env("YOLO_WS_MINIO_BUCKET", "yolo-events")
    )
    minio_public_endpoint: str = field(
        default_factory=lambda: _env("YOLO_WS_MINIO_PUBLIC_ENDPOINT", "")
    )
    minio_secure: bool = field(
        default_factory=lambda: _env_bool("YOLO_WS_MINIO_SECURE", True)
    )
    minio_auto_create_bucket: bool = field(
        default_factory=lambda: _env_bool("YOLO_WS_MINIO_AUTO_CREATE_BUCKET", True)
    )
    minio_prefix: str = field(
        default_factory=lambda: _env("YOLO_WS_MINIO_PREFIX", "events")
    )
    presigned_url_expiry_seconds: int = field(
        default_factory=lambda: _env_int("YOLO_WS_PRESIGNED_URL_EXPIRY_SECONDS", 3600)
    )
    local_storage_root_raw: str = field(
        default_factory=lambda: _env("YOLO_WS_LOCAL_STORAGE_ROOT", "runtime/object_storage")
    )

    def resolve_models_file_path(self) -> Path | None:
        if not self.models_file_raw:
            return None
        return _resolve_relative_path(self.models_file_raw)

    def _load_models_catalog(self) -> ModelCatalog | None:
        config_path = self.resolve_models_file_path()
        if config_path is None:
            return None
        if not config_path.exists():
            raise ValueError(f"YOLO_WS_MODELS_FILE no existe: {config_path}")

        try:
            raw_config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"YOLO_WS_MODELS_FILE no es JSON valido: {config_path}"
            ) from exc

        if isinstance(raw_config, list):
            raw_models = raw_config
            default_model_id = ""
            selection_aliases: dict[str, str] = {}
        elif isinstance(raw_config, dict):
            raw_models = raw_config.get("models")
            if not isinstance(raw_models, list):
                raise ValueError(
                    "YOLO_WS_MODELS_FILE debe tener una lista 'models' o ser una lista directa."
                )
            default_model_id = str(raw_config.get("default_model_id") or "").strip()
            selection_aliases = _parse_string_mapping(
                raw_config.get("selection_aliases"),
                field_name="selection_aliases",
                context=f"YOLO_WS_MODELS_FILE ({config_path})",
            )
        else:
            raise ValueError(
                "YOLO_WS_MODELS_FILE debe ser un objeto JSON o una lista de modelos."
            )

        definitions: dict[str, ModelDefinition] = {}
        has_explicit_fill_event_models = False
        for index, raw_model in enumerate(raw_models, start=1):
            context = f"YOLO_WS_MODELS_FILE ({config_path}) modelo #{index}"
            if not isinstance(raw_model, dict):
                raise ValueError(f"{context}: cada modelo debe ser un objeto JSON.")

            model_id = str(raw_model.get("id") or raw_model.get("model_id") or "").strip()
            if not model_id:
                raise ValueError(f"{context}: falta 'id'.")
            if model_id in definitions:
                raise ValueError(f"{context}: el id '{model_id}' esta repetido.")

            display_name = str(
                raw_model.get("name") or raw_model.get("display_name") or model_id
            ).strip()
            path_raw = str(raw_model.get("path") or raw_model.get("model_path") or "").strip()
            url = str(raw_model.get("url") or raw_model.get("model_url") or "").strip()
            sha256 = str(
                raw_model.get("sha256") or raw_model.get("model_sha256") or ""
            ).strip()
            selector_raw = str(
                raw_model.get("selector")
                or raw_model.get("selection")
                or raw_model.get("selection_value")
                or ""
            ).strip()
            aliases = _parse_string_items(
                raw_model.get("selection_aliases")
                if raw_model.get("selection_aliases") is not None
                else raw_model.get("aliases"),
                field_name="aliases",
                context=context,
            )

            fill_events_enabled: bool | None = None
            if "fill_events_enabled" in raw_model:
                fill_events_enabled = _coerce_bool(raw_model.get("fill_events_enabled"))
                has_explicit_fill_event_models = True
            elif "telegram_enabled" in raw_model:
                fill_events_enabled = _coerce_bool(raw_model.get("telegram_enabled"))
                has_explicit_fill_event_models = True

            fill_event_storage_prefix_raw = str(
                raw_model.get("fill_event_storage_prefix")
                or raw_model.get("storage_prefix")
                or ""
            ).strip()

            definitions[model_id] = ModelDefinition(
                model_id=model_id,
                display_name=display_name,
                path_raw=path_raw,
                url=url,
                sha256=sha256,
                fallback_path_raw=_fallback_model_path(model_id),
                selector_raw=selector_raw,
                selection_aliases=aliases,
                fill_events_enabled=fill_events_enabled,
                fill_event_storage_prefix_raw=fill_event_storage_prefix_raw,
            )

        return ModelCatalog(
            definitions=definitions,
            default_model_id=default_model_id,
            selection_aliases=selection_aliases,
            has_explicit_fill_event_models=has_explicit_fill_event_models,
        )

    def resolve_default_model_id(self) -> str:
        if self.default_model_id_raw:
            return self.default_model_id_raw

        catalog = self._load_models_catalog()
        if catalog is not None:
            if catalog.default_model_id:
                return catalog.default_model_id
            if catalog.definitions:
                return next(iter(catalog.definitions))

        if self.model_ids:
            return self.model_ids[0]
        return "default"

    def resolve_model_definitions(self) -> dict[str, ModelDefinition]:
        catalog = self._load_models_catalog()
        if catalog is not None:
            return catalog.definitions

        if self.model_ids:
            definitions: dict[str, ModelDefinition] = {}
            for model_id in self.model_ids:
                token = _model_env_token(model_id)
                path_raw = _env(f"YOLO_WS_MODEL_{token}_PATH", "")
                url = _env(f"YOLO_WS_MODEL_{token}_URL", "")
                sha256 = _env(f"YOLO_WS_MODEL_{token}_SHA256", "")
                display_name = _env(f"YOLO_WS_MODEL_{token}_NAME", model_id)
                definitions[model_id] = ModelDefinition(
                    model_id=model_id,
                    display_name=display_name,
                    path_raw=path_raw,
                    url=url,
                    sha256=sha256,
                    fallback_path_raw=_fallback_model_path(model_id),
                )
            return definitions

        default_model_id = self.resolve_default_model_id()
        default_path = (
            _resolve_relative_path(self.model_path_raw)
            if self.model_path_raw
            else self.service_root / "runtime" / "model.pt"
        )
        display_name = _env("YOLO_WS_MODEL_NAME", default_path.name)
        return {
            default_model_id: ModelDefinition(
                model_id=default_model_id,
                display_name=display_name,
                path_raw=self.model_path_raw,
                url=self.model_url,
                sha256=self.model_sha256,
                fallback_path_raw="runtime/model.pt",
            )
        }

    def resolve_model_selection_aliases(self) -> dict[str, str]:
        definitions = self.resolve_model_definitions()
        aliases: dict[str, str] = {}
        catalog = self._load_models_catalog()
        if catalog is not None:
            aliases.update(catalog.selection_aliases)
        aliases.update(self.model_selection_aliases)
        for definition in definitions.values():
            if definition.selector_raw:
                aliases.setdefault(definition.selector_raw, definition.model_id)
            for alias in definition.selection_aliases:
                aliases.setdefault(alias, definition.model_id)
        for index, model_id in enumerate(definitions.keys(), start=1):
            aliases.setdefault(str(index), model_id)
        return aliases

    def resolve_model_selection_values(self) -> dict[str, tuple[str, ...]]:
        definitions = self.resolve_model_definitions()
        aliases = self.resolve_model_selection_aliases()
        values: dict[str, list[str]] = {model_id: [] for model_id in definitions}

        for model_id, definition in definitions.items():
            if definition.selector_raw and definition.selector_raw not in values[model_id]:
                values[model_id].append(definition.selector_raw)
            for alias in definition.selection_aliases:
                if alias not in values[model_id]:
                    values[model_id].append(alias)

        for alias, target_model_id in aliases.items():
            model_values = values.get(target_model_id)
            if model_values is None or alias in model_values:
                continue
            model_values.append(alias)

        return {
            model_id: tuple(model_values)
            for model_id, model_values in values.items()
        }

    def resolve_model_selector(self, model_id: str) -> str:
        definitions = self.resolve_model_definitions()
        definition = definitions[model_id]
        if definition.selector_raw:
            return definition.selector_raw

        selection_values = self.resolve_model_selection_values().get(model_id, ())
        numeric_alias = next((value for value in selection_values if value.isdigit()), "")
        if numeric_alias:
            return numeric_alias
        if selection_values:
            return selection_values[0]
        return model_id

    def resolve_requested_model_id(self, requested: str) -> str:
        normalized = str(requested).strip()
        if not normalized:
            return self.resolve_default_model_id()
        aliases = self.resolve_model_selection_aliases()
        return aliases.get(normalized, normalized)

    def resolve_fill_event_model_ids(self) -> tuple[str, ...]:
        definitions = self.resolve_model_definitions()
        catalog = self._load_models_catalog()
        if catalog is not None and catalog.has_explicit_fill_event_models:
            return tuple(
                model_id
                for model_id, definition in definitions.items()
                if definition.fill_events_enabled
            )
        return self.telegram_model_ids

    def resolve_fill_event_storage_prefix(self, model_id: str) -> str:
        definitions = self.resolve_model_definitions()
        definition = definitions.get(model_id)
        if definition is not None:
            return definition.resolve_fill_event_storage_prefix(
                self.fill_event_storage_prefix
            )
        return self.fill_event_storage_prefix

    def resolve_model_path(self) -> Path:
        default_model = self.resolve_model_definitions()[self.resolve_default_model_id()]
        return default_model.resolve_path(self.service_root)

    def resolve_db_path(self) -> Path:
        return _resolve_relative_path(self.db_path_raw)

    def resolve_database_url(self) -> str:
        if self.database_url_raw:
            return _normalize_database_url(self.database_url_raw)
        return _sqlite_url_for(self.resolve_db_path())

    def resolve_temp_dir(self) -> Path:
        return _resolve_relative_path(self.temp_dir_raw)

    def resolve_local_storage_root(self) -> Path:
        return _resolve_relative_path(self.local_storage_root_raw)

    def validate(self) -> "ServiceSettings":
        if not self.auth_username:
            raise ValueError("Define YOLO_WS_AUTH_USERNAME.")
        if not self.auth_password and not self.auth_password_hash:
            raise ValueError(
                "Define YOLO_WS_AUTH_PASSWORD_HASH o YOLO_WS_AUTH_PASSWORD."
            )
        if len(set(self.model_ids)) != len(self.model_ids):
            raise ValueError("YOLO_WS_MODEL_IDS no puede tener ids repetidos.")

        model_definitions = self.resolve_model_definitions()
        if not model_definitions:
            raise ValueError("Debes configurar al menos un modelo.")

        default_model_id = self.resolve_default_model_id()
        if default_model_id not in model_definitions:
            raise ValueError(
                f"YOLO_WS_DEFAULT_MODEL_ID={default_model_id} no existe en la configuracion."
            )

        for model_id, definition in model_definitions.items():
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", model_id):
                raise ValueError(
                    f"Model id invalido: {model_id}. Usa solo letras, numeros, guion o guion bajo."
                )
            if not definition.display_name.strip():
                raise ValueError(f"El modelo {model_id} debe tener un nombre visible.")

        explicit_model_aliases: dict[str, str] = {}
        for model_id, definition in model_definitions.items():
            alias_candidates = []
            if definition.selector_raw:
                alias_candidates.append(definition.selector_raw)
            alias_candidates.extend(definition.selection_aliases)
            for alias in alias_candidates:
                owner = explicit_model_aliases.get(alias)
                if owner is not None and owner != model_id:
                    raise ValueError(
                        f"El selector/alias '{alias}' esta repetido entre {owner} y {model_id}."
                    )
                explicit_model_aliases[alias] = model_id

        unknown_alias_targets = sorted(
            set(self.resolve_model_selection_aliases().values()).difference(model_definitions)
        )
        if unknown_alias_targets:
            raise ValueError(
                "YOLO_WS_MODEL_SELECTION_ALIASES contiene destinos no configurados: "
                + ", ".join(unknown_alias_targets)
            )

        if not 0 < self.conf_threshold <= 1:
            raise ValueError("YOLO_WS_CONF_THRESHOLD debe estar entre 0 y 1.")
        if not 0 < self.iou_threshold <= 1:
            raise ValueError("YOLO_WS_IOU_THRESHOLD debe estar entre 0 y 1.")
        if self.max_frame_bytes <= 0:
            raise ValueError("YOLO_WS_MAX_FRAME_BYTES debe ser mayor que 0.")
        if self.imgsz <= 0:
            raise ValueError("YOLO_WS_IMGSZ debe ser mayor que 0.")
        if self.max_det <= 0:
            raise ValueError("YOLO_WS_MAX_DET debe ser mayor que 0.")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("YOLO_WS_JPEG_QUALITY debe estar entre 1 y 100.")
        if self.auth_timeout_seconds <= 0:
            raise ValueError("YOLO_WS_AUTH_TIMEOUT_SECONDS debe ser mayor que 0.")
        if self.torch_threads <= 0:
            raise ValueError("YOLO_WS_TORCH_THREADS debe ser mayor que 0.")
        if self.events_enabled and not self.trigger_labels:
            raise ValueError("YOLO_WS_TRIGGER_LABELS no puede estar vacio.")
        for label, percent in self.label_percent_aliases.items():
            if not 0 <= percent <= 100:
                raise ValueError(
                    f"YOLO_WS_LABEL_PERCENT_ALIASES invalido para {label}: {percent}."
                )
        if not 0 <= self.telegram_fill_threshold <= 100:
            raise ValueError("YOLO_WS_TELEGRAM_FILL_THRESHOLD debe estar entre 0 y 100.")
        if self.telegram_timeout_seconds <= 0:
            raise ValueError("YOLO_WS_TELEGRAM_TIMEOUT_SECONDS debe ser mayor que 0.")
        if not self.fill_event_storage_prefix.strip():
            raise ValueError("YOLO_WS_FILL_EVENT_STORAGE_PREFIX no puede estar vacio.")

        fill_event_model_ids = self.resolve_fill_event_model_ids()
        unknown_fill_event_models = sorted(
            set(fill_event_model_ids).difference(model_definitions)
        )
        if unknown_fill_event_models:
            raise ValueError(
                "La configuracion de modelos para eventos/Telegram contiene ids no configurados: "
                + ", ".join(unknown_fill_event_models)
            )

        if self.telegram_enabled:
            if not self.telegram_bot_token:
                raise ValueError("Define YOLO_WS_TELEGRAM_BOT_TOKEN.")
            if not self.telegram_chat_id:
                raise ValueError("Define YOLO_WS_TELEGRAM_CHAT_ID.")
            if not fill_event_model_ids:
                raise ValueError(
                    "Debes marcar al menos un modelo para eventos de llenado o Telegram."
                )

        if self.clip_seconds <= 0:
            raise ValueError("YOLO_WS_CLIP_SECONDS debe ser mayor que 0.")
        if self.event_cooldown_seconds < 0:
            raise ValueError(
                "YOLO_WS_EVENT_COOLDOWN_SECONDS no puede ser negativo."
            )
        if self.event_record_fps <= 0:
            raise ValueError("YOLO_WS_EVENT_RECORD_FPS debe ser mayor que 0.")
        if self.snapshot_interval_seconds <= 0:
            raise ValueError(
                "YOLO_WS_SNAPSHOT_INTERVAL_SECONDS debe ser mayor que 0."
            )
        if self.storage_backend not in {"local", "minio"}:
            raise ValueError("YOLO_WS_STORAGE_BACKEND debe ser 'local' o 'minio'.")
        if self.storage_backend == "minio":
            if not self.minio_endpoint:
                raise ValueError("Define YOLO_WS_MINIO_ENDPOINT.")
            if not self.minio_access_key:
                raise ValueError("Define YOLO_WS_MINIO_ACCESS_KEY.")
            if not self.minio_secret_key:
                raise ValueError("Define YOLO_WS_MINIO_SECRET_KEY.")
            if not self.minio_bucket:
                raise ValueError("Define YOLO_WS_MINIO_BUCKET.")
        if self.presigned_url_expiry_seconds <= 0:
            raise ValueError(
                "YOLO_WS_PRESIGNED_URL_EXPIRY_SECONDS debe ser mayor que 0."
            )
        return self


def load_settings() -> ServiceSettings:
    return ServiceSettings().validate()
