"""
title: Summarizer
author: assistant
author_url: https://github.com/pkeffect
funding_url: https://github.com/open-webui
project_url: 
version: 0.1.0
description: Full-featured conversation summarizer with model selection, priority control, intelligent detection, caching, and other quality improvements.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import aiohttp
import json
import time
import hashlib
from uuid import uuid4


class Filter:
    class Valves(BaseModel):
        # Core functionality
        preserve_recent_turns: int = Field(
            default=4,
            description="Minimum number of recent turns to keep unsummarized during compaction",
        )
        max_context_tokens: int = Field(
            default=32768,
            description="Fallback maximum context window size when exact tokenization does not return model capacity",
        )
        summary_trigger_percent: int = Field(
            default=70,
            description="Trigger compaction when exact context usage reaches this percentage of model capacity",
        )
        summary_trigger_turns: int = Field(
            default=6,
            description="Fallback minimum conversation turns before summarization when exact tokenization is unavailable",
        )
        summary_target_percent: int = Field(
            default=45,
            description="After compaction, target this percentage of model capacity",
        )
        estimated_chars_per_token: float = Field(
            default=4.0,
            description="Approximate characters per token used only for diagnostic estimates",
        )

        # Model and processing control
        summary_model: str = Field(
            default="auto",
            description="Model for summarization: 'auto' (current model), or specify model name (e.g., 'llama3.2:3b', 'gpt-3.5-turbo')",
        )
        priority: int = Field(
            default=0,
            description="Filter priority (lower number = higher priority, executed first)",
        )

        # Quality and intelligence settings
        summary_quality: str = Field(
            default="balanced",
            description="Summary quality: 'quick', 'balanced', or 'detailed'",
        )

        # Performance optimization
        enable_caching: bool = Field(
            default=True,
            description="Cache summaries to avoid regenerating identical content",
        )
        summary_api_base_url: str = Field(
            default="",
            description="Optional base URL override for AI summary self-calls (defaults to current Open WebUI request base URL)",
        )
        tokenizer_api_base_url: str = Field(
            default="",
            description="Base URL for exact /tokenize preflight calls against the active inference server",
        )
        summary_api_key: str = Field(
            default="",
            description="Optional API key override for AI summary self-calls to /api/chat/completions",
        )
        tokenizer_api_key: str = Field(
            default="",
            description="Optional API key override for exact /tokenize preflight calls",
        )
        summary_api_timeout_seconds: int = Field(
            default=60,
            description="Timeout in seconds for AI summary HTTP self-calls",
        )

        # Content filtering and enhancement
        summary_max_chars_quick: int = Field(
            default=250,
            description="Maximum number of characters allowed in quick summaries",
        )
        summary_max_chars_balanced: int = Field(
            default=500,
            description="Maximum number of characters allowed in balanced summaries",
        )
        summary_max_chars_detailed: int = Field(
            default=800,
            description="Maximum number of characters allowed in detailed summaries",
        )

        # Debug and testing
        enable_debug: bool = Field(
            default=True, description="Enable debug logging to console"
        )
        test_mode: bool = Field(
            default=True, description="Enable test mode with extra status updates"
        )
        force_summarize_next: bool = Field(
            default=False,
            description="Force summarization on next message (toggle for testing)",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.toggle = True

        # Enhanced compress/summarize icon
        self.icon = """data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGZpbGw9Im5vbmUiIHZpZXdCb3g9IjAgMCAyNCAyNCIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZT0iY3VycmVudENvbG9yIj4KICA8cGF0aCBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIGQ9Ik0zIDEyaDEybTAtNiA0IDQtNC00bTYgNkgxbTAgNi00LTQgNCAxNCIvPgo8L3N2Zz4="""

        # State tracking
        self.request_count = 0
        self.summary_cache = {}  # Cache summaries to avoid regeneration
        self.conversation_states = {}  # Track conversation analysis
        self.last_summary_turn_counts = {}
        self.performance_stats = {"cache_hits": 0, "summaries_created": 0}

    def _debug_log(self, message: str):
        """Debug logging that's always visible"""
        if self.valves.enable_debug:
            print(f"\n=== CONV_SUMMARIZER DEBUG ===")
            print(f"[{time.strftime('%H:%M:%S')}] {message}")
            print("=============================\n")

    def _extract_text_fragments(
        self, value: Any, parent_key: str = ""
    ) -> List[str]:
        """Recursively collect text-like fields from structured message content."""
        if value is None:
            return []

        if isinstance(value, str):
            return [value]

        if isinstance(value, list):
            fragments: List[str] = []
            for item in value:
                fragments.extend(self._extract_text_fragments(item, parent_key))
            return fragments

        if isinstance(value, dict):
            value_type = str(value.get("type", "")).lower()
            if value_type in {
                "image",
                "image_url",
                "input_image",
                "audio",
                "input_audio",
                "file",
            }:
                return []

            preferred_keys = (
                "text",
                "input_text",
                "output_text",
                "content",
                "arguments",
                "argument",
                "summary",
                "title",
                "name",
            )
            ignored_keys = {
                "type",
                "id",
                "index",
                "url",
                "image_url",
                "source",
                "mime_type",
                "media_type",
                "data",
                "b64_json",
                "file_id",
            }

            fragments: List[str] = []
            for key in preferred_keys:
                if key in value:
                    fragments.extend(self._extract_text_fragments(value[key], key))

            for key, item in value.items():
                if key in preferred_keys or key in ignored_keys:
                    continue
                fragments.extend(self._extract_text_fragments(item, key))

            return fragments

        return []

    def _safe_get_text_content(self, msg: Dict[str, Any]) -> str:
        """Safely extract text content from a message, handling list-based formats."""
        raw_content = msg.get("content", "")

        if isinstance(raw_content, str):
            return raw_content

        fragments = self._extract_text_fragments(raw_content)
        if fragments:
            return " ".join(fragment for fragment in fragments if fragment)

        return str(raw_content)

    def _is_summary_message(self, msg: Dict[str, Any]) -> bool:
        """Detect synthetic conversation summary messages regardless of role."""
        return "📋 **Conversation Summary**" in self._safe_get_text_content(msg)

    def _get_summary_model(self, current_model: str) -> str:
        """Determine which model to use for summarization"""
        if self.valves.summary_model == "auto":
            return current_model
        else:
            return self.valves.summary_model

    def _log_model_info(self, current_model: str, summary_model: str):
        """Log model selection information"""
        if summary_model != current_model:
            self._debug_log(
                f"Model selection - Conversation: {current_model}, Summarization: {summary_model}"
            )
        else:
            self._debug_log(
                f"Using same model for conversation and summarization: {current_model}"
            )

    def _estimate_text_tokens(self, text: str) -> int:
        """Estimate token count for diagnostics using a configurable chars/token ratio."""
        if not text:
            return 0

        chars_per_token = max(self.valves.estimated_chars_per_token, 1.0)
        return max(1, int(len(text) / chars_per_token) + 1)

    def _estimate_message_tokens(self, msg: Dict[str, Any]) -> int:
        """Estimate token count for diagnostics including a small role overhead."""
        content = self._safe_get_text_content(msg)
        role = msg.get("role", "")
        return self._estimate_text_tokens(content) + self._estimate_text_tokens(role) + 4

    def _estimate_messages_tokens(self, messages: List[Dict]) -> int:
        """Estimate total token count across messages for diagnostics."""
        return sum(self._estimate_message_tokens(msg) for msg in messages)

    def _get_summary_char_limit(self, quality: str) -> int:
        """Return the configured max summary length for a quality mode."""
        return {
            "quick": self.valves.summary_max_chars_quick,
            "balanced": self.valves.summary_max_chars_balanced,
            "detailed": self.valves.summary_max_chars_detailed,
        }.get(quality, self.valves.summary_max_chars_balanced)

    def _enforce_summary_length(self, summary: str, quality: str) -> str:
        """Trim a summary to its configured length budget."""
        max_length = max(self._get_summary_char_limit(quality), 32)
        if len(summary) > max_length:
            summary = summary[: max_length - 3] + "..."
        return summary

    def _get_conversation_id(
        self, body: Dict[str, Any], __user__: Optional[dict], model: str
    ) -> str:
        """Build a stable conversation identifier for summary tracking."""
        chat_id = body.get("chat_id")
        if chat_id:
            self._debug_log(f"Using chat_id conversation key: {chat_id}")
            return str(chat_id)

        request_id = body.get("id")
        if request_id:
            self._debug_log(
                f"Ignoring request-scoped id for summary state tracking: {request_id}"
            )

        user_id = (__user__ or {}).get("id", "anon")
        fallback_id = f"fallback:{user_id}:{model}"
        self._debug_log(
            f"Using fallback conversation key (user/model scoped): {fallback_id}"
        )
        return fallback_id

    def _hash_messages(self, messages: List[Dict[str, Any]]) -> str:
        """Build a stable hash of message roles and extracted text content."""
        normalized = []
        for msg in messages:
            normalized.append(
                {
                    "role": msg.get("role", ""),
                    "content": self._safe_get_text_content(msg),
                }
            )

        payload = json.dumps(normalized, sort_keys=True, ensure_ascii=True)
        return hashlib.md5(payload.encode()).hexdigest()

    def _restore_summarized_messages(
        self, conversation_id: str, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Reapply the last stored summary to a full chat payload when possible."""
        state = self.conversation_states.get(conversation_id)
        if not state:
            self._debug_log(
                f"No stored summary state for conversation key: {conversation_id}"
            )
            return messages

        if any(self._is_summary_message(msg) for msg in messages):
            return messages

        summarized_count = state.get("summarized_message_count", 0)
        summary_message = state.get("summary_message")
        prefix_hash = state.get("summarized_prefix_hash")

        if not summarized_count or not summary_message or not prefix_hash:
            return messages

        system_msgs = [m for m in messages if m.get("role") == "system"]
        conversation_messages = [
            m for m in messages if m.get("role") in ["user", "assistant"]
        ]

        if len(conversation_messages) < summarized_count:
            self._debug_log(
                "Stored summary state invalidated: conversation is shorter than the summarized prefix"
            )
            self._clear_summary_state(conversation_id)
            return messages

        current_prefix_hash = self._hash_messages(
            conversation_messages[:summarized_count]
        )
        if current_prefix_hash != prefix_hash:
            self._debug_log(
                "Stored summary state invalidated: summarized prefix no longer matches current chat history"
            )
            self._clear_summary_state(conversation_id)
            return messages

        restored_messages = []
        for sys_msg in system_msgs:
            content = self._safe_get_text_content(sys_msg)
            if (
                "📋" not in content
                and "Summary" not in content
                and "Previous conversation" not in content
            ):
                restored_messages.append(sys_msg)

        restored_messages.append(summary_message)
        restored_messages.extend(conversation_messages[summarized_count:])

        self._debug_log(
            f"Reapplied stored summary state to incoming request ({len(messages)} -> {len(restored_messages)} messages)"
        )
        return restored_messages

    def _clear_summary_state(self, conversation_id: str) -> None:
        """Drop any stored summary baseline for a conversation."""
        self.conversation_states.pop(conversation_id, None)
        self.last_summary_turn_counts.pop(conversation_id, None)

    def _should_persist_summary_state(
        self, conv_state: Dict[str, Any], force_resync: bool
    ) -> bool:
        """Persist summary state after any successful summary creation."""
        return True

    def _record_summary_state(
        self,
        conversation_id: str,
        conv_state: Dict[str, Any],
        new_messages: List[Dict],
        messages_to_summarize: List[Dict],
        summary_message: Dict[str, Any],
    ) -> None:
        """Persist the compacted summary so future full payloads can be restored."""
        self.conversation_states[conversation_id] = {
            "valid_turns_at_summary": conv_state.get("valid_turns", 0),
            "summarized_message_count": len(messages_to_summarize),
            "summarized_prefix_hash": self._hash_messages(messages_to_summarize),
            "summary_message": summary_message,
        }

    def _analyze_conversation_state(self, messages: List[Dict]) -> Dict[str, Any]:
        """Analyze current conversation size and summary presence."""

        system_msgs = [m for m in messages if m.get("role") == "system"]
        conv_messages = [
            m
            for m in messages
            if m.get("role") in ["user", "assistant"] and not self._is_summary_message(m)
        ]

        # Check for existing summaries
        existing_summaries = 0
        for msg in messages:
            if self._is_summary_message(msg):
                existing_summaries += 1

        estimated_tokens = self._estimate_messages_tokens(messages)

        self._debug_log(
            f"Conversation analysis - Total: {len(conv_messages)}, Summaries: {existing_summaries}, Est. tokens: {estimated_tokens}"
        )

        return {
            "total_turns": len(conv_messages),
            "valid_turns": len(conv_messages),
            "has_existing_summary": existing_summaries > 0,
            "summary_count": existing_summaries,
            "estimated_tokens": estimated_tokens,
        }

    def _get_active_context_thresholds(self, max_model_len: Optional[int]) -> Dict[str, int]:
        """Return trigger and target budgets using exact model limits when available."""
        max_tokens = max(max_model_len or self.valves.max_context_tokens, 1)
        trigger_percent = min(max(self.valves.summary_trigger_percent, 1), 100)
        target_percent = min(max(self.valves.summary_target_percent, 1), trigger_percent)

        return {
            "max_tokens": max_tokens,
            "trigger_tokens": max(1, int(max_tokens * (trigger_percent / 100))),
            "target_tokens": max(1, int(max_tokens * (target_percent / 100))),
        }

    def _get_tokenizer_api_url(self) -> Optional[str]:
        """Resolve the exact tokenizer URL for provider preflight checks."""
        base_url = (self.valves.tokenizer_api_base_url or "").strip()
        if not base_url:
            return None

        return f"{base_url.rstrip('/')}/tokenize"

    def _normalize_messages_for_tokenizer(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Canonicalize messages for tokenizers that require a single leading system message."""
        if not messages:
            return []

        system_messages = [msg for msg in messages if msg.get("role") == "system"]
        non_system_messages = [msg for msg in messages if msg.get("role") != "system"]

        if not system_messages:
            return messages

        merged_system_parts = []
        for msg in system_messages:
            content = self._safe_get_text_content(msg).strip()
            if content:
                merged_system_parts.append(content)

        if not merged_system_parts:
            return non_system_messages

        merged_system_message = {
            "role": "system",
            "content": "\n\n".join(merged_system_parts),
        }
        return [merged_system_message, *non_system_messages]

    def _build_tokenizer_api_headers(self, __request__) -> Dict[str, str]:
        """Build auth headers for exact tokenizer preflight calls."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        if self.valves.tokenizer_api_key:
            headers["Authorization"] = f"Bearer {self.valves.tokenizer_api_key}"
            return headers

        if __request__ is None:
            return headers

        request_headers = getattr(__request__, "headers", None)
        if not request_headers:
            return headers

        authorization = request_headers.get("authorization") or request_headers.get(
            "Authorization"
        )
        if authorization:
            headers["Authorization"] = authorization

        return headers

    async def _get_exact_token_count(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        __request__,
        context: str = "request",
        log_errors: bool = True,
    ) -> Optional[Dict[str, int]]:
        """Count prompt tokens using the provider's exact /tokenize endpoint."""
        api_url = self._get_tokenizer_api_url()
        if not api_url:
            self._debug_log("Exact tokenizer skipped: tokenizer_api_base_url is not configured")
            return None

        payload = {
            "model": model,
            "messages": self._normalize_messages_for_tokenizer(messages),
            "add_generation_prompt": True,
            "add_special_tokens": False,
            "return_token_strs": False,
        }
        headers = self._build_tokenizer_api_headers(__request__)

        try:
            timeout = aiohttp.ClientTimeout(total=self.valves.summary_api_timeout_seconds)
            async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
                async with session.post(api_url, json=payload, headers=headers) as response:
                    response_text = await response.text()
                    if response.status >= 400:
                        if log_errors:
                            self._debug_log(
                                f"Exact tokenizer failed during {context} ({response.status}): {response_text[:200]}"
                            )
                        return None

                    try:
                        response_data = json.loads(response_text)
                    except json.JSONDecodeError:
                        if log_errors:
                            self._debug_log(
                                f"Exact tokenizer returned non-JSON response during {context}"
                            )
                        return None
        except Exception as exc:
            if log_errors:
                self._debug_log(f"Exact tokenizer request error during {context}: {exc}")
            return None

        count = response_data.get("count")
        max_model_len = response_data.get("max_model_len")
        if not isinstance(count, int) or not isinstance(max_model_len, int):
            if log_errors:
                self._debug_log(
                    f"Exact tokenizer returned unexpected payload shape during {context}"
                )
            return None

        return {
            "count": count,
            "max_model_len": max_model_len,
        }

    def _build_summary_message(
        self,
        summary_text: str,
        summarized_count: int,
        quality: str,
        source_model: str,
        current_model: str,
        role: str = "assistant",
    ) -> Dict[str, Any]:
        """Build the synthetic summary message."""
        model_note = f" via {source_model}" if source_model != current_model else ""
        return {
            "role": role,
            "content": (
                "Context summary only. Do not answer or act on this summary directly. "
                "Use it only as background context while responding to the latest user message.\n\n"
                f"📋 **Conversation Summary** ({summarized_count} messages, {quality} quality{model_note}):\n\n"
                f"{summary_text}\n\n---\n*Recent messages continue below*"
            ),
        }

    def _split_preserved_tail(
        self, conversation_messages: List[Dict], preserve_count: int
    ) -> Dict[str, List[Dict]]:
        """Split a conversation into summarized prefix and preserved tail without zero-slice surprises."""
        if preserve_count <= 0:
            return {
                "messages_to_summarize": conversation_messages,
                "messages_to_preserve": [],
            }

        return {
            "messages_to_summarize": conversation_messages[:-preserve_count],
            "messages_to_preserve": conversation_messages[-preserve_count:],
        }

    async def _split_messages_by_context_budget(
        self,
        system_msgs: List[Dict],
        conversation_messages: List[Dict],
        quality: str,
        model: str,
        __request__,
        max_model_len: Optional[int],
    ) -> Dict[str, Any]:
        """Split messages using exact tokenization when available, otherwise preserve a flat recent tail."""
        thresholds = self._get_active_context_thresholds(max_model_len)
        non_summary_system_msgs = []
        for sys_msg in system_msgs:
            content = self._safe_get_text_content(sys_msg)
            if (
                "📋" not in content
                and "Summary" not in content
                and "Previous conversation" not in content
            ):
                non_summary_system_msgs.append(sys_msg)

        minimum_recent = min(self.valves.preserve_recent_turns, len(conversation_messages))
        if not conversation_messages:
            return {
                "messages_to_summarize": [],
                "messages_to_preserve": [],
                "target_tokens": thresholds["target_tokens"],
                "trigger_tokens": thresholds["trigger_tokens"],
                "preserved_count": 0,
                "used_exact_tokenizer": False,
            }

        exact_tokenizer_available = self._get_tokenizer_api_url() is not None
        if not exact_tokenizer_available:
            preserve_count = minimum_recent
            split_messages = self._split_preserved_tail(
                conversation_messages, preserve_count
            )
            return {
                "messages_to_summarize": split_messages["messages_to_summarize"],
                "messages_to_preserve": split_messages["messages_to_preserve"],
                "target_tokens": thresholds["target_tokens"],
                "trigger_tokens": thresholds["trigger_tokens"],
                "preserved_count": len(split_messages["messages_to_preserve"]),
                "used_exact_tokenizer": False,
            }

        placeholder_text = "x" * max(self._get_summary_char_limit(quality), 32)
        best_preserve_count = minimum_recent
        used_exact_tokenizer = False
        planning_fell_back = False

        for preserve_count in range(minimum_recent, len(conversation_messages) + 1):
            split_messages = self._split_preserved_tail(
                conversation_messages, preserve_count
            )
            messages_to_preserve = split_messages["messages_to_preserve"]
            messages_to_summarize = split_messages["messages_to_summarize"]
            if not messages_to_summarize:
                break

            placeholder_summary = self._build_summary_message(
                placeholder_text,
                len(messages_to_summarize),
                quality,
                model,
                model,
                role="assistant",
            )
            provisional_messages = [
                *non_summary_system_msgs,
                placeholder_summary,
                *messages_to_preserve,
            ]
            exact_context = await self._get_exact_token_count(
                provisional_messages,
                model,
                __request__,
                context="compaction planning",
            )
            if exact_context is None:
                planning_fell_back = True
                break

            used_exact_tokenizer = True
            if exact_context["count"] <= thresholds["target_tokens"]:
                best_preserve_count = preserve_count
                continue

            break

        best_split_messages = self._split_preserved_tail(
            conversation_messages, best_preserve_count
        )
        return {
            "messages_to_summarize": best_split_messages["messages_to_summarize"],
            "messages_to_preserve": best_split_messages["messages_to_preserve"],
            "target_tokens": thresholds["target_tokens"],
            "trigger_tokens": thresholds["trigger_tokens"],
            "preserved_count": len(best_split_messages["messages_to_preserve"]),
            "used_exact_tokenizer": used_exact_tokenizer,
            "planning_fell_back": planning_fell_back,
        }

    def _should_summarize_smart(
        self, conv_state: Dict[str, Any], conversation_id: str
    ) -> bool:
        """Summarize when exact token usage reaches the trigger budget, or fall back to turn count."""
        exact_tokens = conv_state.get("exact_tokens")
        max_model_len = conv_state.get("max_model_len")
        thresholds = self._get_active_context_thresholds(max_model_len)

        if exact_tokens is not None:
            if exact_tokens >= thresholds["max_tokens"]:
                self._debug_log(
                    f"Exact context exceeds configured max ({exact_tokens}/{thresholds['max_tokens']} tokens)"
                )
                return True

            if exact_tokens >= thresholds["trigger_tokens"]:
                self._debug_log(
                    f"Exact context threshold reached ({exact_tokens}/{thresholds['max_tokens']} tokens, trigger {thresholds['trigger_tokens']})"
                )
                return True

            return False

        fallback_turns = max(self.valves.summary_trigger_turns, 1)
        if conv_state.get("valid_turns", 0) >= fallback_turns:
            self._debug_log(
                f"Exact tokenizer unavailable; falling back to turn threshold ({conv_state.get('valid_turns', 0)}/{fallback_turns} turns)"
            )
            return True

        return False

    def _get_cache_key(self, messages: List[Dict]) -> str:
        """Generate cache key for messages"""
        if not self.valves.enable_caching:
            return ""

        # Create a hash based on message content
        content_string = (
            f"model:{self.valves.summary_model}|"
            f"quality:{self.valves.summary_quality}|"
        )
        for msg in messages[-25:]:  # Use last 25 messages for key
            content = self._safe_get_text_content(msg)
            content_string += f"{msg.get('role', '')}:{content[:150]}"

        return hashlib.md5(content_string.encode()).hexdigest()[:16]

    def _extract_existing_summary_text(self, messages: List[Dict]) -> str:
        """Extract previously generated summary text from summary messages."""
        for msg in reversed(messages):
            content = self._safe_get_text_content(msg)
            if "📋 **Conversation Summary**" not in content:
                continue

            summary_text = content
            if "):\n\n" in summary_text:
                summary_text = summary_text.split("):\n\n", 1)[1]
            if "\n\n---\n*Recent messages continue below*" in summary_text:
                summary_text = summary_text.split(
                    "\n\n---\n*Recent messages continue below*", 1
                )[0]

            return summary_text.strip()

        return ""

    def _build_summary_input_messages(
        self, existing_summary_text: str, messages_to_summarize: List[Dict]
    ) -> List[Dict]:
        """Create the input slice for cumulative summarization."""
        summary_input_messages: List[Dict] = []

        if existing_summary_text:
            summary_input_messages.append(
                {
                    "role": "system",
                    "content": (
                        "Previous conversation summary to roll forward:\n\n"
                        f"{existing_summary_text}"
                    ),
                }
            )

        summary_input_messages.extend(messages_to_summarize)
        return summary_input_messages

    def _get_summary_api_url(self, __request__) -> Optional[str]:
        """Resolve the self-call chat completions URL."""
        base_url = (self.valves.summary_api_base_url or "").strip()

        if not base_url and __request__ is not None:
            request_base_url = getattr(__request__, "base_url", None)
            if request_base_url:
                base_url = str(request_base_url).strip()

        if not base_url:
            return None

        return f"{base_url.rstrip('/')}/api/chat/completions"

    def _build_summary_api_headers(self, __request__) -> Dict[str, str]:
        """Build auth headers for the self-call, preferring explicit valve auth."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        if self.valves.summary_api_key:
            headers["Authorization"] = f"Bearer {self.valves.summary_api_key}"
            return headers

        if __request__ is None:
            return headers

        request_headers = getattr(__request__, "headers", None)
        if not request_headers:
            return headers

        authorization = request_headers.get("authorization") or request_headers.get(
            "Authorization"
        )
        if authorization:
            headers["Authorization"] = authorization

        cookie = request_headers.get("cookie") or request_headers.get("Cookie")
        if cookie:
            headers["Cookie"] = cookie

        return headers

    async def _create_ai_summary_via_http(
        self,
        messages: List[Dict],
        summary_model: str,
        quality: str,
        __request__,
    ) -> Optional[str]:
        """Generate a summary by calling Open WebUI's public chat completions API."""
        api_url = self._get_summary_api_url(__request__)
        if not api_url:
            self._debug_log("AI summarization HTTP path skipped: no base URL available")
            return None

        prompt_messages = self._build_ai_summary_messages(messages, quality)
        if len(prompt_messages) < 2:
            return None

        payload = {
            "model": summary_model,
            "messages": prompt_messages,
            "stream": False,
            "chat_id": f"local:summarizer:{uuid4().hex}",
            "id": f"summarizer-{uuid4().hex}",
        }
        headers = self._build_summary_api_headers(__request__)

        self._debug_log(
            f"Attempting AI summarization via HTTP API {api_url} with model {summary_model}"
        )

        timeout = aiohttp.ClientTimeout(total=self.valves.summary_api_timeout_seconds)
        try:
            async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
                async with session.post(api_url, json=payload, headers=headers) as response:
                    response_text = await response.text()
                    if response.status >= 400:
                        self._debug_log(
                            f"AI summarization HTTP path failed ({response.status}): {response_text[:200]}"
                        )
                        return None

                    try:
                        response_data = json.loads(response_text)
                    except json.JSONDecodeError:
                        self._debug_log(
                            "AI summarization HTTP path returned non-JSON response"
                        )
                        return None
        except Exception as exc:
            self._debug_log(f"AI summarization HTTP path error: {exc}")
            return None

        summary = self._extract_ai_summary_text(response_data)
        if summary:
            summary = self._enforce_summary_length(summary, quality)
            self._debug_log(f"AI summarization via HTTP succeeded ({len(summary)} chars)")
        else:
            self._debug_log("AI summarization HTTP path returned no usable text")

        return summary

    def _build_ai_summary_messages(
        self, messages: List[Dict], quality: str
    ) -> List[Dict[str, str]]:
        """Build a compact prompt for model-based summarization."""
        style_map = {
            "quick": "Keep it brief and high-signal in 3-5 sentences.",
            "balanced": "Keep it concise but complete in 1 short paragraph or 4-6 sentences.",
            "detailed": "Keep it comprehensive in 2 short paragraphs, preserving important decisions and constraints.",
        }

        conversation_lines = []
        for msg in messages:
            content = self._safe_get_text_content(msg).strip()
            if not content:
                continue
            role = msg.get("role", "assistant").upper()
            conversation_lines.append(f"{role}: {content}")

        transcript = "\n\n".join(conversation_lines)

        return [
            {
                "role": "system",
                "content": (
                    "You summarize prior chat context for continuation. "
                    "Return plain text only. Do not use markdown, bullet lists, or speaker labels unless necessary. "
                    "Preserve user goals, constraints, decisions, unresolved questions, and important factual details. "
                    f"{style_map.get(quality, style_map['balanced'])}"
                ),
            },
            {
                "role": "user",
                "content": (
                    "Summarize this earlier conversation so a model can continue it naturally. "
                    "Focus on requests, answers, decisions, technical details, and any open work.\n\n"
                    f"Conversation:\n{transcript}"
                ),
            },
        ]

    def _extract_ai_summary_text(self, response: Any) -> Optional[str]:
        """Extract summary text from an OpenAI-style chat completion response."""
        if not isinstance(response, dict):
            return None

        choices = response.get("choices") or []
        if not choices:
            for key in ("response", "content", "generated_text", "text"):
                text = self._extract_structured_response_text(response.get(key))
                if text:
                    return text
            return None

        message = choices[0].get("message") or {}
        candidates = [
            message.get("content"),
            message.get("text"),
            choices[0].get("text"),
            response.get("response"),
            response.get("content"),
            response.get("generated_text"),
            response.get("text"),
        ]

        for candidate in candidates:
            text = self._extract_structured_response_text(candidate)
            if text:
                return text

        return None

    def _extract_structured_response_text(self, value: Any) -> Optional[str]:
        """Extract text from string or structured response content."""
        if value is None:
            return None

        if isinstance(value, str):
            return value.strip() or None

        fragments = self._extract_text_fragments(value)
        if fragments:
            joined = " ".join(fragment for fragment in fragments if fragment).strip()
            return joined or None

        if isinstance(value, dict):
            for key in ("text", "content", "value"):
                text = self._extract_structured_response_text(value.get(key))
                if text:
                    return text

        if isinstance(value, list):
            parts = []
            for item in value:
                text = self._extract_structured_response_text(item)
                if text:
                    parts.append(text)

            joined = " ".join(parts).strip()
            return joined or None

        return None

    async def _create_ai_summary(
        self,
        messages: List[Dict],
        summary_model: str,
        quality: str,
        __request__,
    ) -> Optional[str]:
        """Generate a summary through the public chat completions API."""
        if not __request__:
            self._debug_log("AI summarization skipped: missing request context")
            return None

        return await self._create_ai_summary_via_http(
            messages, summary_model, quality, __request__
        )

    async def inlet(
        self,
        body: dict,
        __event_emitter__,
        __user__: Optional[dict] = None,
        __request__=None,
    ) -> dict:

        self._debug_log("=== INLET CALLED ===")

        if str(body.get("chat_id", "")).startswith("local:summarizer:"):
            self._debug_log("Skipping summarizer for internal AI summary request")
            return body

        if not self.toggle:
            self._debug_log("Filter disabled via toggle")
            return body

        try:
            messages = body.get("messages", [])
            model = body.get("model", "unknown")

            # Determine which model to use for summarization
            summary_model = self._get_summary_model(model)
            self._log_model_info(model, summary_model)

            self._debug_log(f"Total messages: {len(messages)}")
            self._debug_log(f"Current model: {model}")
            self._debug_log(f"Summary model: {summary_model}")
            self._debug_log(f"Filter priority: {self.valves.priority}")

            if self.valves.test_mode:
                model_info = (
                    f" (using {summary_model})" if summary_model != model else ""
                )
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"🔍 Summarizer analyzing {len(messages)} messages{model_info}...",
                            "done": False,
                            "hidden": False,
                        },
                    }
                )

            # Enhanced conversation analysis
            conversation_id = self._get_conversation_id(body, __user__, model)
            force_resync = self.valves.force_summarize_next
            if force_resync:
                self._debug_log(
                    "Force summarize requested: clearing stored summary baseline before processing"
                )
                self._clear_summary_state(conversation_id)

            messages = self._restore_summarized_messages(conversation_id, messages)
            body["messages"] = messages

            conv_state = self._analyze_conversation_state(messages)
            exact_context = await self._get_exact_token_count(
                messages, model, __request__, context="incoming request"
            )
            if exact_context:
                conv_state["exact_tokens"] = exact_context["count"]
                conv_state["max_model_len"] = exact_context["max_model_len"]
            else:
                conv_state["exact_tokens"] = None
                conv_state["max_model_len"] = None

            self._debug_log(f"Enhanced analysis: {conv_state}")

            # Smart decision on summarization
            should_summarize_smart = self._should_summarize_smart(
                conv_state, conversation_id
            )
            should_summarize = (
                self.valves.force_summarize_next or should_summarize_smart
            )

            self._debug_log(
                f"Should summarize: {should_summarize} (smart: {should_summarize_smart}, force: {self.valves.force_summarize_next})"
            )

            if should_summarize:

                self._debug_log("=== STARTING ENHANCED SUMMARIZATION ===")

                # Check cache first
                cache_key = self._get_cache_key(messages)
                cached_summary = None
                if cache_key and cache_key in self.summary_cache:
                    cached_summary = self.summary_cache[cache_key]
                    self.performance_stats["cache_hits"] += 1
                    self._debug_log(f"Found cached summary for key: {cache_key}")

                model_display = (
                    summary_model if summary_model != model else "current model"
                )
                thresholds = self._get_active_context_thresholds(
                    conv_state.get("max_model_len")
                )
                if conv_state.get("exact_tokens") is not None:
                    usage_description = (
                        f"{conv_state['exact_tokens']} exact tokens, target {thresholds['target_tokens']}"
                    )
                else:
                    usage_description = (
                        f"fallback at {self.valves.summary_trigger_turns} turns, currently {conv_state['valid_turns']}"
                    )
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"📝 Creating {self.valves.summary_quality} summary using {model_display} ({usage_description})",
                            "done": False,
                            "hidden": False,
                        },
                    }
                )

                # Get all message types
                system_msgs = [m for m in messages if m.get("role") == "system"]
                conversation_messages = [
                    m
                    for m in messages
                    if m.get("role") in ["user", "assistant"]
                    and not self._is_summary_message(m)
                ]

                split_result = await self._split_messages_by_context_budget(
                    system_msgs,
                    conversation_messages,
                    self.valves.summary_quality,
                    model,
                    __request__,
                    conv_state.get("max_model_len"),
                )
                messages_to_summarize = split_result["messages_to_summarize"]
                messages_to_preserve = split_result["messages_to_preserve"]

                if messages_to_summarize:

                    existing_summary_text = self._extract_existing_summary_text(
                        messages
                    )
                    summary_input_messages = self._build_summary_input_messages(
                        existing_summary_text, messages_to_summarize
                    )

                    self._debug_log(
                        f"Summarizing oldest {len(messages_to_summarize)} messages"
                    )
                    self._debug_log(
                        f"Preserving {len(messages_to_preserve)} recent messages"
                    )
                    if split_result.get("used_exact_tokenizer"):
                        self._debug_log(
                            f"Context budget - current {conv_state['exact_tokens']} exact tokens, trigger {split_result['trigger_tokens']}, target {split_result['target_tokens']}, preserved {split_result['preserved_count']} messages"
                        )
                    elif conv_state.get("exact_tokens") is not None:
                        self._debug_log(
                            f"Compaction planning fell back after tokenizer rejected the provisional compacted payload; current request exact size is {conv_state['exact_tokens']} tokens, preserving {split_result['preserved_count']} messages by fallback"
                        )
                    else:
                        self._debug_log(
                            f"Context budget fallback - {conv_state['valid_turns']} turns, trigger {self.valves.summary_trigger_turns}, preserving {split_result['preserved_count']} messages"
                        )
                    if existing_summary_text:
                        self._debug_log(
                            f"Rolling forward existing summary ({len(existing_summary_text)} chars)"
                        )

                    # Create enhanced summary
                    summary_source = "ai"
                    if cached_summary:
                        summary_text = cached_summary
                        summary_source = "cached"
                        self._debug_log(f"Using cached summary")
                    else:
                        summary_text = await self._create_ai_summary(
                            summary_input_messages,
                            summary_model,
                            self.valves.summary_quality,
                            __request__,
                        )
                        if not summary_text:
                            raise RuntimeError(
                                "AI summarization failed or returned no usable text"
                            )
                        self._debug_log(
                            f"Generated AI summary using {summary_model}"
                        )

                        # Cache the result
                        if cache_key:
                            self.summary_cache[cache_key] = summary_text
                            # Clean old cache entries (keep last 15)
                            if len(self.summary_cache) > 15:
                                oldest_key = list(self.summary_cache.keys())[0]
                                del self.summary_cache[oldest_key]

                        self.performance_stats["summaries_created"] += 1

                    self._debug_log(f"Final summary: {summary_text}")

                    # Build new message list (keep working logic)
                    new_messages = []

                    # Keep system messages that aren't summaries
                    for sys_msg in system_msgs:
                        content = sys_msg.get("content", "")
                        if (
                            "📋" not in content
                            and "Summary" not in content
                            and "Previous conversation" not in content
                        ):
                            new_messages.append(sys_msg)
                            self._debug_log(f"Kept system message: {content[:50]}...")

                    # Add our enhanced summary as a system message
                    summary_message = self._build_summary_message(
                        summary_text,
                        len(messages_to_summarize),
                        self.valves.summary_quality,
                        summary_model,
                        model,
                    )
                    new_messages.append(summary_message)
                    self._debug_log("Added enhanced summary message")

                    # Add preserved recent messages
                    new_messages.extend(messages_to_preserve)
                    self._debug_log(
                        f"Added {len(messages_to_preserve)} preserved messages"
                    )

                    # Update the body
                    body["messages"] = new_messages

                    # Update tracking
                    if self._should_persist_summary_state(conv_state, force_resync):
                        self._record_summary_state(
                            conversation_id,
                            conv_state,
                            new_messages,
                            messages_to_summarize,
                            summary_message,
                        )
                        self.last_summary_turn_counts[conversation_id] = conv_state[
                            "valid_turns"
                        ]
                    else:
                        self._debug_log(
                            "Summary state not persisted because the request did not include a retained summary"
                        )
                        self._clear_summary_state(conversation_id)

                    self._debug_log(
                        f"Final message count: {len(new_messages)} (was {len(messages)})"
                    )
                    self._debug_log(f"Performance stats: {self.performance_stats}")

                    # Create status message with model info
                    source_label = {
                        "ai": "AI",
                        "cached": "cached",
                    }[summary_source]
                    new_exact_context = await self._get_exact_token_count(
                        new_messages,
                        model,
                        __request__,
                        context="post-summary recount",
                        log_errors=False,
                    )
                    if not new_exact_context:
                        self._debug_log(
                            "Post-summary exact recount unavailable; using fallback success status"
                        )
                    if new_exact_context:
                        status_msg = (
                            f"✅ {source_label.capitalize()} summary created using {model_display}! "
                            f"{conv_state.get('exact_tokens', 0)} exact tokens -> {new_exact_context['count']} exact (target {split_result['target_tokens']})"
                        )
                    else:
                        status_msg = (
                            f"✅ {source_label.capitalize()} summary created using {model_display}! "
                            f"fallback turn threshold {self.valves.summary_trigger_turns}, preserved {len(messages_to_preserve)} messages"
                        )

                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": status_msg,
                                "done": True,
                                "hidden": False,
                            },
                        }
                    )

                    # Reset force flag
                    if self.valves.force_summarize_next:
                        self.valves.force_summarize_next = False
                        self._debug_log("Reset force_summarize_next flag")

                else:
                    self._debug_log("Not enough messages to summarize")
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": f"ℹ️ Not enough messages to summarize (need >{self.valves.preserve_recent_turns})",
                                "done": True,
                                "hidden": False,
                            },
                        }
                    )

            else:
                self._debug_log("No summarization needed")
                if self.valves.test_mode:
                    thresholds = self._get_active_context_thresholds(
                        conv_state.get("max_model_len")
                    )
                    if conv_state.get("exact_tokens") is not None:
                        idle_status = (
                            f"✋ No summarization needed ({conv_state['exact_tokens']} exact tokens, trigger {thresholds['trigger_tokens']}) | Model: {summary_model}"
                        )
                    else:
                        idle_status = (
                            f"✋ No summarization needed ({conv_state['valid_turns']} turns, fallback trigger {self.valves.summary_trigger_turns}) | Model: {summary_model}"
                        )
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": idle_status,
                                "done": True,
                                "hidden": False,
                            },
                        }
                    )

            self.request_count += 1
            self._debug_log(f"Filter processed request #{self.request_count}")

            return body

        except Exception as e:
            error_msg = f"Filter error: {str(e)}"
            self._debug_log(f"ERROR: {error_msg}")

            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"❌ Summarizer error: {str(e)[:80]}",
                        "done": True,
                        "hidden": False,
                    },
                }
            )

            # Return original body on error
            return body

    async def outlet(
        self, body: dict, __event_emitter__, __user__: Optional[dict] = None
    ) -> dict:
        """Outlet - log that we completed processing"""
        self._debug_log("=== OUTLET CALLED ===")
        return body
