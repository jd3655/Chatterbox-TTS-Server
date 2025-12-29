import pathlib
import re
import sys
import types
from pathlib import Path

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.modules.setdefault("pydub", types.SimpleNamespace(AudioSegment=None))
sys.modules.setdefault("soundfile", types.SimpleNamespace())
sys.modules.setdefault("torchaudio", types.SimpleNamespace())
torch_stub = types.ModuleType("torch")
torch_stub.Tensor = type("Tensor", (), {})
torch_stub.device = type("device", (), {})
torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
torch_stub.no_grad = types.SimpleNamespace()
torch_stub.from_numpy = lambda *args, **kwargs: None
torch_stub.cat = lambda *args, **kwargs: None
torch_stub.nn = types.SimpleNamespace(Conv1d=object)
sys.modules.setdefault("torch", torch_stub)
sys.modules.setdefault("librosa", types.SimpleNamespace())
numpy_stub = types.ModuleType("numpy")
numpy_stub.ndarray = type("ndarray", (), {})
numpy_stub.float32 = float
numpy_stub.int16 = int
numpy_stub.zeros = lambda *args, **kwargs: []
numpy_stub.asarray = lambda x, **kwargs: x
numpy_stub.clip = lambda x, *args, **kwargs: x
numpy_stub.concatenate = lambda *args, **kwargs: []
numpy_stub.expand_dims = lambda x, *args, **kwargs: x
numpy_stub.mean = lambda x, *args, **kwargs: 0
numpy_stub.abs = lambda x, *args, **kwargs: x
sys.modules.setdefault("numpy", numpy_stub)
fake_config = types.ModuleType("config")
fake_config.get_predefined_voices_path = lambda ensure_absolute=False: Path(".")
fake_config.get_reference_audio_path = lambda ensure_absolute=False: Path(".")
fake_config.config_manager = types.SimpleNamespace(
    get_path=lambda *args, **kwargs: Path("."),
    get_bool=lambda *args, **kwargs: False,
    get_int=lambda *args, **kwargs: 0,
    get_float=lambda *args, **kwargs: 0.0,
    get_string=lambda *args, **kwargs: "",
)
sys.modules.setdefault("config", fake_config)

import utils


def _count_words(text: str) -> int:
    return len(re.findall(r"\b[\w']+\b", text))


def test_smart_split_respects_paragraphs():
    text = (
        "Paragraph one has enough words to trigger splitting and it should respect paragraph boundaries by placing this entire thought together before moving on to the next area.\n\n"
        "Paragraph two follows with its own set of sentences and should land in a separate chunk to demonstrate newline awareness by the splitter and keep ideas grouped."
    )
    chunks = utils.smart_split_text(text)
    assert len(chunks) == 2
    assert "Paragraph one" in chunks[0]
    assert "Paragraph two" in chunks[1]


def test_chunks_stay_within_bounds():
    text = " ".join(
        [
            "This sentence contains many words to test the splitter behavior across boundaries and maintain ranges."
        ]
        * 7
    )
    chunks = utils.smart_split_text(text)
    min_words = int(10.0 * 2.7)
    max_words = int(18.0 * 2.7)
    for chunk in chunks:
        words = _count_words(chunk)
        assert min_words <= words <= max_words


def test_avoids_weak_endings_when_possible():
    text = (
        "This is how to. "
        "We continue the explanation with more detail to illustrate the concept clearly. "
        "Ending cleanly now with plenty of buffer words to keep the final chunk healthy."
    )
    chunks = utils.smart_split_text(
        text,
        target_seconds=7.0,
        min_seconds=4.0,
        max_seconds=20.0,
        base_words_per_second=1.0,
    )
    assert not chunks[0].strip().lower().endswith("to.")


def test_avoids_weak_start_when_possible():
    text = (
        "The first sentence wraps up here. "
        "And then another starts with a connector to continue the story and set up the next idea. "
        "Final ending waits with additional words for balance and smoother pacing."
    )
    chunks = utils.smart_split_text(
        text,
        target_seconds=8.0,
        min_seconds=5.0,
        max_seconds=22.0,
        base_words_per_second=1.0,
    )
    for chunk in chunks[1:]:
        first_words = re.findall(r"\b[\w']+\b", chunk)
        if first_words:
            assert first_words[0].lower() not in utils.WEAK_START_WORDS


def test_bracket_tags_not_split():
    text = (
        "Hello [laugh] there friend; this keeps going to ensure length; continuing more words to reach a limit. "
        "Another sentence follows to close the idea cleanly."
    )
    chunks = utils.smart_split_text(
        text,
        target_seconds=8.0,
        min_seconds=5.0,
        max_seconds=16.0,
        base_words_per_second=2.0,
    )
    for chunk in chunks:
        assert not re.search(r"\[[^\]]*$", chunk)
        assert not re.search(r"^[^\[]*\]", chunk)


def test_soft_boundaries_used_for_long_sentences():
    text = (
        "This is a long sentence; it keeps running with many clauses; another clause — and continues to ensure we need splits for readability; ending eventually."
    )
    chunks = utils.smart_split_text(
        text,
        target_seconds=5.0,
        min_seconds=3.0,
        max_seconds=10.0,
        base_words_per_second=1.5,
    )
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.strip()
