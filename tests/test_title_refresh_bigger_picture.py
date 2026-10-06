"""Tests for the 'bigger picture' nudge in adaptive title refresh.

Covers the refresh-context additions to api/streaming.py:
  - _build_title_refresh_digest (bounded digest: opening + title + latest)
  - _title_prompts(kind='refresh') (refresh prompt variant with current title)
  - _run_background_title_refresh (threads digest + current_title into the
    generation call; first-generation path stays byte-identical in behavior)
  - _maybe_schedule_title_refresh (builds and passes the digest)
  - generate_session_title_for_session(prefer_latest=True) (nudge + title)
"""
import sys
import os
import threading
import unittest
from unittest.mock import MagicMock, patch

import pytest

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from api.streaming import (
    _build_title_refresh_digest,
    _title_prompts,
    _run_background_title_refresh,
    _maybe_schedule_title_refresh,
    generate_session_title_for_session,
)


def _user_msg(text):
    return {'role': 'user', 'content': text}


def _asst_msg(text, tool_calls=None):
    msg = {'role': 'assistant', 'content': text}
    if tool_calls is not None:
        msg['tool_calls'] = tool_calls
    return msg


def _make_session(title='My Title', llm_title_generated=True, messages=None, session_id='sid1'):
    s = MagicMock()
    s.title = title
    s.llm_title_generated = llm_title_generated
    s.messages = messages or []
    s.session_id = session_id
    s.save = MagicMock()
    return s


# ---------------------------------------------------------------------------
# _build_title_refresh_digest
# ---------------------------------------------------------------------------

class TestBuildTitleRefreshDigest:
    def test_includes_current_title(self):
        msgs = [_user_msg('q'), _asst_msg('a')]
        d = _build_title_refresh_digest(msgs, 'Existing Topic Title')
        assert 'Existing Topic Title' in d

    def test_includes_opening_exchange(self):
        msgs = [_user_msg('opening question'), _asst_msg('opening answer'),
                _user_msg('later q'), _asst_msg('later a')]
        d = _build_title_refresh_digest(msgs, 'T')
        assert 'opening question' in d
        assert 'opening answer' in d

    def test_includes_latest_exchange(self):
        msgs = [_user_msg('opening question'), _asst_msg('opening answer'),
                _user_msg('latest question'), _asst_msg('latest answer')]
        d = _build_title_refresh_digest(msgs, 'T')
        assert 'latest question' in d
        assert 'latest answer' in d

    def test_hard_cap_respected(self):
        msgs = [_user_msg('x' * 3000), _asst_msg('y' * 3000),
                _user_msg('z' * 3000), _asst_msg('w' * 3000)]
        d = _build_title_refresh_digest(msgs, 'T')
        assert len(d) <= 1200

    def test_custom_cap(self):
        msgs = [_user_msg('x' * 3000)]
        d = _build_title_refresh_digest(msgs, 'T', max_chars=100)
        assert len(d) <= 100

    def test_empty_messages_returns_empty_or_title_only(self):
        d = _build_title_refresh_digest([], 'Some Title')
        assert len(d) <= 1200
        assert 'Some Title' in d

    def test_none_messages_safe(self):
        d = _build_title_refresh_digest(None, 'T')
        assert isinstance(d, str)

    def test_skips_tool_call_only_assistant_in_latest(self):
        tool_only = {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 't1'}]}
        msgs = [_user_msg('q'), _asst_msg('real answer'), _user_msg('q2'), tool_only]
        d = _build_title_refresh_digest(msgs, 'T')
        assert 'real answer' in d


# ---------------------------------------------------------------------------
# _title_prompts refresh variant
# ---------------------------------------------------------------------------

class TestTitlePromptsRefreshVariant:
    def test_refresh_includes_current_title(self):
        _, prompts = _title_prompts('q', 'a', None, kind='refresh',
                                    current_title='My Current Title')
        assert any('My Current Title' in p for p in prompts)

    def test_refresh_digest_is_user_message(self):
        digest = 'Current title: T\n\nOpening exchange:\nUser: hello'
        qa, prompts = _title_prompts('q', 'a', None, kind='refresh',
                                     current_title='T', context_digest=digest)
        assert qa == digest

    def test_refresh_prompt_worded_for_refresh_not_first(self):
        _, prompts = _title_prompts('q', 'a', None, kind='refresh',
                                    current_title='T')
        # Must NOT use the first-generation "conversation start" wording.
        assert all('conversation start' not in p for p in prompts)
        # Must instruct to keep the bigger picture / established topic.
        assert any('bigger picture' in p for p in prompts)

    def test_refresh_prompt_keeps_language_and_emoji_rules(self):
        _, prompts = _title_prompts('How do I fix this bug', 'Like this', None,
                                    kind='refresh', current_title='T')
        joined = '\n'.join(prompts)
        # Language rule + emoji rule must be present in a refresh prompt.
        assert 'language' in joined.lower()
        assert 'emoji' in joined.lower()

    def test_refresh_prompt_keeps_word_label_rules(self):
        _, prompts = _title_prompts('q', 'a', None, kind='refresh', current_title='T')
        joined = '\n'.join(prompts)
        assert '3-8 words' in joined

    def test_refresh_falls_back_to_snippets_when_no_digest(self):
        qa, prompts = _title_prompts('my question', 'my answer', None,
                                     kind='refresh', current_title='T')
        assert 'my question' in qa
        assert 'my answer' in qa


# ---------------------------------------------------------------------------
# First-generation path unchanged
# ---------------------------------------------------------------------------

class TestFirstGenerationUnchanged:
    def test_first_prompt_text_unchanged(self):
        qa, prompts = _title_prompts('How do I fix this bug', 'Like this')
        assert len(prompts) == 2
        assert prompts[0].startswith(
            'Generate a short session title from this conversation start.\n'
            "Use BOTH the user's question and the assistant's visible answer.\n"
        )
        assert 'Do not output acknowledgements' in prompts[0]
        assert prompts[1].startswith('Rewrite this conversation start as a concise noun-phrase title.\n')
        assert 'Use the actual topic, not the task outcome.\n' in prompts[1]

    def test_first_qa_format_unchanged(self):
        qa, _ = _title_prompts('How do I fix this bug', 'Like this')
        assert qa == 'User question:\nHow do I fix this bug\n\nAssistant answer:\nLike this'

    def test_default_kind_is_first(self):
        qa, prompts = _title_prompts('q', 'a')
        assert len(prompts) == 2


# ---------------------------------------------------------------------------
# _run_background_title_refresh threads refresh_context through
# ---------------------------------------------------------------------------

class TestRunBackgroundTitleRefreshThreadsContext:
    def _make_put_event(self):
        events = []
        def put(name, data):
            events.append((name, data))
        return put, events

    def _make_session_obj(self, title='Old Title'):
        s = MagicMock()
        s.title = title
        s.llm_title_generated = True
        s.save = MagicMock()
        return s

    def test_refresh_context_passed_to_aux(self):
        """When the aux route is used, the refresh call receives
        refresh_context with kind/current_title/context_digest."""
        put, events = self._make_put_event()
        captured = {}
        def fake_aux(user_text, assistant_text, agent=None, use_agent_model=False,
                     conversation_id='', refresh_context=None):
            captured['refresh_context'] = refresh_context
            return 'New Title', 'llm_aux', ''
        with patch('api.streaming.get_session') as mock_get, \
             patch('api.streaming.SESSIONS', {}), \
             patch('api.streaming.LOCK', threading.Lock()), \
             patch('api.streaming._aux_title_generation_enabled', return_value=True), \
             patch('api.streaming._aux_title_configured', return_value=True), \
             patch('api.streaming._generate_llm_session_title_via_aux', side_effect=fake_aux):
            s = self._make_session_obj(title='Old Title')
            mock_get.return_value = s
            _run_background_title_refresh(
                'sid', 'u', 'a', 'Old Title', put, agent=None,
                context_digest='Current title: Old Title\n\nOpening exchange:\nUser: hi',
            )
        rc = captured['refresh_context']
        assert rc is not None
        assert rc['kind'] == 'refresh'
        assert rc['current_title'] == 'Old Title'
        assert rc['context_digest'].startswith('Current title: Old Title')

    def test_refresh_context_passed_to_agent_route(self):
        """When the agent route is used, the refresh call receives refresh_context."""
        put, events = self._make_put_event()
        captured = {}
        def fake_agent(agent, user_text, assistant_text, refresh_context=None):
            captured['refresh_context'] = refresh_context
            return 'New Title', 'llm', ''
        with patch('api.streaming.get_session') as mock_get, \
             patch('api.streaming.SESSIONS', {}), \
             patch('api.streaming.LOCK', threading.Lock()), \
             patch('api.streaming._aux_title_generation_enabled', return_value=True), \
             patch('api.streaming._aux_title_configured', return_value=False), \
             patch('api.streaming._generate_llm_session_title_for_agent', side_effect=fake_agent):
            s = self._make_session_obj(title='Old Title')
            mock_get.return_value = s
            agent = MagicMock()
            _run_background_title_refresh(
                'sid', 'u', 'a', 'Old Title', put, agent=agent,
                context_digest='Current title: Old Title',
            )
        rc = captured['refresh_context']
        assert rc is not None
        assert rc['kind'] == 'refresh'
        assert rc['current_title'] == 'Old Title'
        assert rc['context_digest'] == 'Current title: Old Title'

    def test_refresh_still_emits_title_event(self):
        put, events = self._make_put_event()
        with patch('api.streaming.get_session') as mock_get, \
             patch('api.streaming.SESSIONS', {}), \
             patch('api.streaming.LOCK', threading.Lock()), \
             patch('api.streaming._aux_title_generation_enabled', return_value=True), \
             patch('api.streaming._aux_title_configured', return_value=True), \
             patch('api.streaming._generate_llm_session_title_via_aux',
                   return_value=('New Title', 'llm_aux', '')):
            s = self._make_session_obj(title='Old Title')
            mock_get.return_value = s
            _run_background_title_refresh('sid', 'u', 'a', 'Old Title', put,
                                          agent=None, context_digest='digest')
        assert any(name == 'title' and data.get('title') == 'New Title'
                   for name, data in events)


# ---------------------------------------------------------------------------
# _maybe_schedule_title_refresh builds and passes the digest
# ---------------------------------------------------------------------------

class TestMaybeScheduleTitleRefreshPassesDigest:
    def test_schedules_with_digest_and_current_title(self):
        session = _make_session(title='My Topic', llm_title_generated=True,
                                messages=[_user_msg('q1'), _asst_msg('a1'),
                                          _user_msg('q2'), _asst_msg('a2'),
                                          _user_msg('q3'), _asst_msg('a3')],
                                session_id='sid9')
        captured = {}
        class FakeThread:
            def __init__(self, target=None, args=None, kwargs=None, daemon=False):
                captured['target'] = target
                captured['args'] = args
                captured['kwargs'] = kwargs
            def start(self):
                pass
        with patch('api.streaming._get_title_refresh_interval', return_value=3), \
             patch('api.streaming.session_has_manual_title', return_value=False), \
             patch('api.streaming.threading.Thread', FakeThread):
            _maybe_schedule_title_refresh(session, lambda *_a: None, None)
        # args: (session_id, last_u, last_a, current_title, put_event, agent)
        args = captured['args']
        assert args[0] == 'sid9'
        assert args[1] == 'q3'         # last_u
        assert args[2] == 'a3'         # last_a
        assert args[3] == 'My Topic'   # current_title
        # kwargs: context_digest is the bounded bigger-picture digest
        kwargs = captured['kwargs']
        assert 'context_digest' in kwargs
        assert 'My Topic' in kwargs['context_digest']
        assert 'q1' in kwargs['context_digest']   # opening exchange
        assert 'q3' in kwargs['context_digest']   # latest exchange

    def test_no_schedule_when_interval_disabled(self):
        session = _make_session(title='T', messages=[_user_msg('q')])
        called = {}
        class FakeThread:
            def __init__(self, *a, **k):
                called['yes'] = True
            def start(self):
                pass
        with patch('api.streaming._get_title_refresh_interval', return_value=0), \
             patch('api.streaming.threading.Thread', FakeThread):
            _maybe_schedule_title_refresh(session, lambda *_a: None, None)
        assert 'yes' not in called


# ---------------------------------------------------------------------------
# generate_session_title_for_session prefer_latest nudge
# ---------------------------------------------------------------------------

class TestGenerateSessionTitlePreferLatest:
    def test_prefer_latest_builds_refresh_context(self):
        session = _make_session(title='Existing Topic',
                                messages=[_user_msg('q'), _asst_msg('a')])
        captured = {}
        def fake_aux(user_text, assistant_text, agent=None, use_agent_model=False,
                     conversation_id='', refresh_context=None):
            captured['refresh_context'] = refresh_context
            return 'New Topic', 'llm_aux', ''
        with patch('api.streaming._aux_title_generation_enabled', return_value=True), \
             patch('api.streaming._generate_llm_session_title_via_aux', side_effect=fake_aux):
            generate_session_title_for_session(session, prefer_latest=True)
        rc = captured['refresh_context']
        assert rc is not None
        assert rc['kind'] == 'refresh'
        assert rc['current_title'] == 'Existing Topic'

    def test_prefer_latest_returns_generated_title(self):
        session = _make_session(title='Existing Topic',
                                messages=[_user_msg('q'), _asst_msg('a')])
        with patch('api.streaming._aux_title_generation_enabled', return_value=True), \
             patch('api.streaming._generate_llm_session_title_via_aux',
                   return_value=('New Topic', 'llm_aux', '')):
            title, status, _ = generate_session_title_for_session(session, prefer_latest=True)
        assert title == 'New Topic'
        assert status == 'llm_aux'

    def test_non_prefer_latest_no_refresh_context(self):
        session = _make_session(title='Existing Topic',
                                messages=[_user_msg('q'), _asst_msg('a')])
        captured = {}
        def fake_aux(user_text, assistant_text, agent=None, use_agent_model=False,
                     conversation_id='', refresh_context=None):
            captured['refresh_context'] = refresh_context
            return 'New Topic', 'llm_aux', ''
        with patch('api.streaming._aux_title_generation_enabled', return_value=True), \
             patch('api.streaming._generate_llm_session_title_via_aux', side_effect=fake_aux):
            generate_session_title_for_session(session, prefer_latest=False)
        # First-generation path must NOT receive a refresh_context.
        assert captured['refresh_context'] is None
