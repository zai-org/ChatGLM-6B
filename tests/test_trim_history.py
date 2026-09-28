import importlib.util
import sys
import types
import unittest
from contextlib import nullcontext


def load_trim():
    streamlit = types.ModuleType("streamlit")

    class Sidebar:
        def slider(self, *args, **kwargs):
            return 1

    streamlit.sidebar = Sidebar()
    streamlit.session_state = {}
    streamlit.set_page_config = lambda **kwargs: None
    streamlit.cache_resource = lambda fn: fn
    streamlit.container = lambda: None
    streamlit.text_area = lambda *args, **kwargs: ""
    streamlit.button = lambda *args, **kwargs: False
    streamlit.spinner = lambda *args, **kwargs: nullcontext()
    sys.modules["streamlit"] = streamlit
    chat = types.ModuleType("streamlit_chat")
    chat.message = lambda *args, **kwargs: None
    sys.modules["streamlit_chat"] = chat
    transformers = types.ModuleType("transformers")
    transformers.AutoModel = object
    transformers.AutoTokenizer = object
    sys.modules["transformers"] = transformers
    spec = importlib.util.spec_from_file_location("web_demo2", "web_demo2.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.trim_history


class TrimHistoryTest(unittest.TestCase):
    def test_a_history_past_twenty_turns_keeps_the_last_twenty(self):
        trim_history = load_trim()
        history = [(str(i), str(i)) for i in range(25)]
        trimmed = trim_history(history)
        self.assertEqual(len(trimmed), 20)
        self.assertEqual(trimmed[0], ("5", "5"))
        self.assertEqual(trimmed[-1], ("24", "24"))

    def test_twenty_turns_stay(self):
        trim_history = load_trim()
        history = [(str(i), str(i)) for i in range(20)]
        self.assertEqual(trim_history(history), history)


if __name__ == "__main__":
    unittest.main()
