import unittest
from pathlib import Path

from api.streaming import (
    _fallback_title_from_exchange,
    _first_exchange_snippets,
    _is_bad_new_title,
    _script_counts,
    _sanitize_generated_title,
    _title_language_mismatch,
    _title_prompt_emoji_rule,
    _title_prompts,
)


class TestGeneratedTitleSanitization(unittest.TestCase):
    def test_strips_session_title_markdown_prefix(self):
        self.assertEqual(
            _sanitize_generated_title("**Session Title:** Clarifying Topic for Discussion"),
            "Clarifying Topic for Discussion",
        )

    def test_strips_plain_title_prefix(self):
        self.assertEqual(
            _sanitize_generated_title("Title: Clarifying Topic for Discussion"),
            "Clarifying Topic for Discussion",
        )

    def test_strips_wrapping_markdown_emphasis(self):
        self.assertEqual(
            _sanitize_generated_title("**Clarifying Topic for Discussion**"),
            "Clarifying Topic for Discussion",
        )

    def test_first_exchange_skips_empty_assistant_tool_call_placeholder(self):
        messages = [
            {"role": "user", "content": "What time is it in San Francisco?"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1"}]},
            {"role": "tool", "content": "tool output", "tool_call_id": "call_1"},
            {"role": "assistant", "content": "It is 6:16 PM in San Francisco."},
        ]
        self.assertEqual(
            _first_exchange_snippets(messages),
            ("What time is it in San Francisco?", "It is 6:16 PM in San Francisco."),
        )

    def test_fallback_title_uses_english_discussion_suffix(self):
        self.assertEqual(
            _fallback_title_from_exchange('Please review "random cancel"', ""),
            "random cancel discussion",
        )

    def test_fallback_title_summary_label_is_english(self):
        self.assertEqual(
            _fallback_title_from_exchange("Generate a short title summary test", ""),
            "Session title auto-summary test",
        )

    def test_fallback_title_non_latin_input_uses_english_placeholder(self):
        self.assertEqual(
            _fallback_title_from_exchange("讨论一下这个问题", ""),
            "Conversation topic",
        )

    def test_fallback_title_non_latin_quoted_topic_uses_english_placeholder(self):
        self.assertEqual(
            _fallback_title_from_exchange('Please review "讨论主题"', ""),
            "Conversation topic",
        )

    def test_title_generation_source_has_no_cjk_literals(self):
        src = Path("api/streaming.py").read_text(encoding="utf-8")
        self.assertNotRegex(src, r"[\u4e00-\u9fff]", "title generation code should stay English-only")


class TestEmojiTitleHint(unittest.TestCase):
    """The title prompt hints at a single leading emoji; downstream guards must
    keep emoji titles (no strip, no CoT/echo rejection, no script-drift reject)."""

    def test_prompt_contains_emoji_rule(self):
        _qa, prompts = _title_prompts("How do I fix a Python import error?", "Check your virtualenv.")
        self.assertEqual(len(prompts), 2)
        for p in prompts:
            self.assertIn("ONE emoji", p)
            self.assertIn("never use more than one", p)

    def test_prompt_examples_include_emoji(self):
        _qa, prompts = _title_prompts("a", "b")
        self.assertIn("Good (with emoji)", prompts[0])

    def test_emoji_rule_has_no_cjk_literals(self):
        rule = _title_prompt_emoji_rule()
        self.assertNotRegex(rule, r"[\u4e00-\u9fff]", "emoji rule must stay English-only")

    def test_sanitizer_keeps_leading_emoji(self):
        self.assertEqual(
            _sanitize_generated_title("🐍 Fix Python Import Error"),
            "🐍 Fix Python Import Error",
        )

    def test_bad_title_check_keeps_emoji(self):
        self.assertFalse(_is_bad_new_title("⛅ Weekend Weather Forecast"))

    def test_script_counts_ignore_emoji(self):
        self.assertEqual(_script_counts("🐍 Fix Python"), {"latin": 9})

    def test_language_mismatch_ignores_emoji(self):
        # Latin conversation + emoji-prefixed Latin title is NOT a drift.
        self.assertFalse(_title_language_mismatch("How do I fix a Python import error?", "🐍 Fix Python Import Error"))
