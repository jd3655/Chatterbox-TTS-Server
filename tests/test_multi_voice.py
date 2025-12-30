import sys
from pathlib import Path

import types

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

numpy_stub = types.SimpleNamespace(
    full=lambda length, value, dtype=None: [value] * length,
    zeros=lambda length, dtype=None: [0] * length,
    concatenate=lambda parts: sum(parts, []),
    abs=lambda x: x,
    max=max,
    float32=float,
    int16=int,
    ndarray=type("ndarray", (), {}),
    clip=lambda x, *args, **kwargs: x,
    expand_dims=lambda x, *args, **kwargs: x,
    asarray=lambda x, **kwargs: x,
    mean=lambda x, *args, **kwargs: 0,
)
sys.modules.setdefault("numpy", numpy_stub)
torch_stub = types.SimpleNamespace(
    Tensor=type("Tensor", (), {}),
    device=type("device", (), {}),
    cuda=types.SimpleNamespace(is_available=lambda: False, get_device_name=lambda *a, **k: "cpu"),
    backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
    tensor=lambda *args, **kwargs: types.SimpleNamespace(to=lambda *a, **k: None),
)
librosa_stub = types.SimpleNamespace(resample=lambda data, orig_sr=None, target_sr=None: data)
sys.modules.setdefault("torch", torch_stub)
sys.modules.setdefault("librosa", librosa_stub)
sys.modules.setdefault("pydub", types.SimpleNamespace(AudioSegment=None))
sys.modules.setdefault("soundfile", types.SimpleNamespace())
sys.modules.setdefault("torchaudio", types.SimpleNamespace())
class _BaseModel:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def copy(self, *args, **kwargs):
        return _BaseModel(**self.__dict__)


def _Field(default=None, *args, **kwargs):
    return default


sys.modules.setdefault(
    "pydantic", types.SimpleNamespace(BaseModel=_BaseModel, Field=_Field)
)
sys.modules.setdefault(
    "engine",
    types.SimpleNamespace(
        MODEL_LOADED=True,
        synthesize=lambda *a, **k: (None, None),
        chatterbox_model=types.SimpleNamespace(sr=24000),
        reload_model=lambda: True,
        get_model_info=lambda: {},
    ),
)
fastapi_stub = types.ModuleType("fastapi")


class HTTPException(Exception):
    def __init__(self, status_code, detail=None, headers=None):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
        self.headers = headers or {}


class BackgroundTasks:
    def add_task(self, *args, **kwargs):
        return None


class FastAPI:
    def __init__(self, *args, **kwargs):
        self.routes = []

    def get(self, *args, **kwargs):
        def decorator(func):
            return func

        return decorator

    def post(self, *args, **kwargs):
        def decorator(func):
            return func

        return decorator

    def mount(self, *args, **kwargs):
        return None

    def add_middleware(self, *args, **kwargs):
        return None


fastapi_stub.FastAPI = FastAPI
fastapi_stub.HTTPException = HTTPException
fastapi_stub.Request = object
fastapi_stub.File = lambda *args, **kwargs: None
fastapi_stub.UploadFile = type("UploadFile", (), {})
fastapi_stub.Form = lambda *args, **kwargs: None
fastapi_stub.BackgroundTasks = BackgroundTasks

responses_stub = types.ModuleType("fastapi.responses")


class _SimpleResponse:
    def __init__(self, content=None, media_type=None, headers=None):
        self.body_iterator = self._aiter(content)
        self.media_type = media_type
        self.headers = headers or {}

    async def _aiter(self, content):
        if content is None:
            yield b""
        elif isinstance(content, (bytes, bytearray)):
            yield content
        elif hasattr(content, "read"):
            yield content.read()
        else:
            yield bytes(content)


class StreamingResponse(_SimpleResponse):
    pass


class JSONResponse(_SimpleResponse):
    pass


class HTMLResponse(_SimpleResponse):
    pass


class FileResponse(_SimpleResponse):
    pass


responses_stub.StreamingResponse = StreamingResponse
responses_stub.JSONResponse = JSONResponse
responses_stub.HTMLResponse = HTMLResponse
responses_stub.FileResponse = FileResponse

staticfiles_stub = types.ModuleType("fastapi.staticfiles")
staticfiles_stub.StaticFiles = type("StaticFiles", (), {"__init__": lambda self, *a, **k: None})

templating_stub = types.ModuleType("fastapi.templating")
templating_stub.Jinja2Templates = type("Jinja2Templates", (), {"__init__": lambda self, *a, **k: None})

cors_stub = types.ModuleType("fastapi.middleware.cors")
cors_stub.CORSMiddleware = type("CORSMiddleware", (), {})

sys.modules.setdefault("fastapi", fastapi_stub)
sys.modules.setdefault("fastapi.responses", responses_stub)
sys.modules.setdefault("fastapi.staticfiles", staticfiles_stub)
sys.modules.setdefault("fastapi.templating", templating_stub)
sys.modules.setdefault("fastapi.middleware.cors", cors_stub)
sys.modules.setdefault("yaml", types.SimpleNamespace(safe_load=lambda *a, **k: {}, safe_dump=lambda *a, **k: ""))

import numpy as np  # noqa: E402
import multi_voice  # noqa: E402
import server  # noqa: E402
from models import CustomTTSRequest  # noqa: E402


def test_split_paragraphs_blank_line():
    text = "First paragraph.\nStill first.\n\nSecond paragraph starts here."
    paragraphs = multi_voice.split_paragraphs(text)
    assert paragraphs == ["First paragraph.\nStill first.", "Second paragraph starts here."]


def test_build_segments_assignments_fallback():
    paragraphs = ["One", "Two", "Three"]
    assignments = ["clay"]
    segments = multi_voice.build_segments_from_paragraphs(paragraphs, assignments, "emily")
    assert segments[0]["voice_id"] == "clay"
    assert segments[1]["voice_id"] == "emily"
    assert segments[2]["voice_id"] == "emily"


def test_parse_voice_directives_respects_text_before_first():
    text = "Narrator intro.\n<voice:clay>\nHello!\n<voice:emily>\nHi."
    segments = multi_voice.parse_voice_directives(text)
    assert segments[0]["voice_id"] is None
    assert "Narrator intro" in segments[0]["text"]
    assert segments[1]["voice_id"] == "clay"
    assert segments[2]["voice_id"] == "emily"


def test_directives_ignored_inside_brackets():
    text = "[<voice:clay>] Actual <voice:emily>Outside"
    segments = multi_voice.parse_voice_directives(text)
    assert segments[0]["voice_id"] is None
    assert "<voice:clay>" in segments[0]["text"]
    assert segments[1]["voice_id"] == "emily"


def test_multi_voice_concatenation(monkeypatch):
    calls = []

    monkeypatch.setattr(server, "engine", type("E", (), {"MODEL_LOADED": True}))
    monkeypatch.setattr(server, "get_audio_sample_rate", lambda: 24000)
    monkeypatch.setattr(server.config_manager, "get_bool", lambda *args, **kwargs: False)

    def fake_predefined_voices():
        return [
            {"filename": "clay.wav", "display_name": "Clay"},
            {"filename": "emily.wav", "display_name": "Emily"},
        ]

    monkeypatch.setattr(server.utils, "get_predefined_voices", fake_predefined_voices)
    monkeypatch.setattr(server, "_resolve_predefined_voice_path", lambda vid: Path(vid))
    monkeypatch.setattr(server, "_prepare_text_for_request", lambda text, req: text)

    class FakeArray(list):
        def astype(self, *args, **kwargs):
            return self

    def fake_synthesize_processed_text(processed_text, req, audio_prompt_path_for_engine=None, perf_monitor=None):
        value = 1.0 if req.predefined_voice_id == "clay.wav" else 2.0
        audio = FakeArray(np.full(4, value, dtype=np.float32))
        calls.append((req.predefined_voice_id, processed_text))
        return audio, 24000

    monkeypatch.setattr(server, "_synthesize_processed_text", fake_synthesize_processed_text)

    encoded = {}

    def fake_encode_audio(audio_array, sample_rate, output_format, target_sample_rate):
        encoded["sample_rate"] = sample_rate
        encoded["target"] = target_sample_rate
        encoded["length"] = len(audio_array)
        return b"x" * 200

    monkeypatch.setattr(server.utils, "encode_audio", fake_encode_audio)

    request = CustomTTSRequest(
        text="One\n\nTwo",
        voice_mode="predefined",
        predefined_voice_id="clay.wav",
        output_format="wav",
        multi_voice=True,
        multi_voice_mode="paragraphs",
        multi_voice_assignments=["clay.wav", "emily.wav"],
        multi_voice_insert_pause_between_segments=True,
        multi_voice_pause_seconds=0.25,
    )

    import asyncio

    response = asyncio.get_event_loop().run_until_complete(
        server.custom_tts_endpoint(request, BackgroundTasks())
    )
    async def _collect():
        return b"".join([chunk async for chunk in response.body_iterator])

    body = asyncio.get_event_loop().run_until_complete(_collect())
    assert len(body) == 200
    assert encoded["sample_rate"] == 24000
    assert encoded["target"] == 24000
    assert encoded["length"] == 4 + int(0.25 * 24000) + 4
    assert calls[0][0] == "clay.wav"
    assert calls[1][0] == "emily.wav"


def test_directives_accept_extensionless_case_insensitive(monkeypatch):
    calls = []

    monkeypatch.setattr(server, "engine", type("E", (), {"MODEL_LOADED": True}))
    monkeypatch.setattr(server, "get_audio_sample_rate", lambda: 24000)
    monkeypatch.setattr(server.config_manager, "get_bool", lambda *args, **kwargs: False)

    def fake_predefined_voices():
        return [
            {"filename": "Clay.wav", "display_name": "Clay"},
            {"filename": "Emily.wav", "display_name": "Emily"},
        ]

    monkeypatch.setattr(server.utils, "get_predefined_voices", fake_predefined_voices)
    monkeypatch.setattr(server, "_resolve_predefined_voice_path", lambda vid: Path(vid))
    monkeypatch.setattr(server, "_prepare_text_for_request", lambda text, req: text)

    class FakeArray(list):
        def astype(self, *args, **kwargs):
            return self

    def fake_synthesize_processed_text(processed_text, req, audio_prompt_path_for_engine=None, perf_monitor=None):
        calls.append(req.predefined_voice_id)
        return FakeArray(np.full(2, 1.0, dtype=np.float32)), 24000

    monkeypatch.setattr(server, "_synthesize_processed_text", fake_synthesize_processed_text)
    monkeypatch.setattr(server.utils, "encode_audio", lambda *a, **k: b"x" * 200)

    request = CustomTTSRequest(
        text="<voice:clay>\nHello\n<voice:EMILY>\nHi",
        voice_mode="predefined",
        predefined_voice_id="clay",
        output_format="wav",
        multi_voice=True,
        multi_voice_mode="directives",
    )

    import asyncio

    response = asyncio.get_event_loop().run_until_complete(
        server.custom_tts_endpoint(request, BackgroundTasks())
    )
    async def _collect():
        return b"".join([chunk async for chunk in response.body_iterator])

    _ = asyncio.get_event_loop().run_until_complete(_collect())
    assert calls[0] == "Clay.wav"
    assert calls[1] == "Emily.wav"
