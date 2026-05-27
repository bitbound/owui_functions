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
import inspect
from uuid import uuid4


class Filter:
    class Valves(BaseModel):
        # Core functionality
        summary_trigger_turns: int = Field(
            default=8,
            description="Number of conversation turns that triggers summarization",
        )
        preserve_recent_turns: int = Field(
            default=4, description="Number of recent turns to keep unsummarized"
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
        smart_detection: bool = Field(
            default=True,
            description="Detect mid-conversation loading and existing summaries",
        )
        adaptive_threshold: bool = Field(
            default=True,
            description="Adjust trigger based on message complexity and length",
        )

        # Performance optimization
        enable_caching: bool = Field(
            default=True,
            description="Cache summaries to avoid regenerating identical content",
        )
        enable_ai_summarization: bool = Field(
            default=False,
            description="Use AI model for summarization (experimental - currently uses enhanced rule-based)",
        )
        summary_api_base_url: str = Field(
            default="",
            description="Optional base URL override for AI summary self-calls (defaults to current Open WebUI request base URL)",
        )
        summary_api_key: str = Field(
            default="",
            description="Optional API key override for AI summary self-calls to /api/chat/completions",
        )
        summary_api_timeout_seconds: int = Field(
            default=60,
            description="Timeout in seconds for AI summary HTTP self-calls",
        )

        # Content filtering and enhancement
        min_message_length: int = Field(
            default=20,
            description="Minimum characters per message to count for summarization",
        )
        preserve_important_details: bool = Field(
            default=True,
            description="Extract and preserve numbers, dates, and key facts",
        )
        include_context_hints: bool = Field(
            default=True, description="Add helpful context hints to summaries"
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
        self.conversation_count = 0
        self.summary_cache = {}  # Cache summaries to avoid regeneration
        self.conversation_states = {}  # Track conversation analysis
        self.last_summary_turn_counts = {}  # Track when we last summarized
        self.performance_stats = {"cache_hits": 0, "summaries_created": 0}

    def _debug_log(self, message: str):
        """Debug logging that's always visible"""
        if self.valves.enable_debug:
            print(f"\n=== CONV_SUMMARIZER DEBUG ===")
            print(f"[{time.strftime('%H:%M:%S')}] {message}")
            print("=============================\n")

    def _safe_get_text_content(self, msg: Dict[str, Any]) -> str:
        """Safely extract text content from a message, handling list-based formats."""
        raw_content = msg.get("content", "")
        
        # If it's already a string, just return it
        if isinstance(raw_content, str):
            return raw_content
        
        # If it's a list, flatten only text-type items
        if isinstance(raw_content, list):
            parts = []
            for item in raw_content:
                if isinstance(item, dict):
                    # Only extract text from text-type items
                    if item.get("type") == "text":
                        parts.append(item.get("text", ""))
                    # Skip other types like image_url, audio, etc.
                else:
                    # Fallback for any unexpected non-dict item
                    parts.append(str(item))
            return " ".join(parts)
        
        # Fallback for any other unexpected type
        return str(raw_content)

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

    def _analyze_conversation_state(self, messages: List[Dict]) -> Dict[str, Any]:
        """Enhanced conversation analysis with smart detection"""

        if not self.valves.smart_detection:
            # Fallback to simple counting
            conv_messages = [
                m for m in messages if m.get("role") in ["user", "assistant"]
            ]
            return {
                "total_turns": len(conv_messages),
                "valid_turns": len(conv_messages),
                "has_existing_summary": False,
                "summary_count": 0,
                "complexity_score": 1.0,
                "avg_message_length": 100,
                "recent_activity_score": 1.0,
            }

        system_msgs = [m for m in messages if m.get("role") == "system"]
        conv_messages = [m for m in messages if m.get("role") in ["user", "assistant"]]

        # Check for existing summaries
        existing_summaries = 0
        for msg in system_msgs:
            content = msg.get("content", "")
            if (
                "📋" in content
                or "Summary" in content
                or "Previous conversation" in content
            ):
                existing_summaries += 1

        # Calculate message complexity
        total_chars = 0
        complex_messages = 0
        question_count = 0
        code_messages = 0
        technical_messages = 0

        for msg in conv_messages:
            content = self._safe_get_text_content(msg)
            if len(content) >= self.valves.min_message_length:
                total_chars += len(content)

                # Complexity indicators
                content_lower = content.lower()
                if any(
                    word in content_lower
                    for word in [
                        "analyze",
                        "explain",
                        "complex",
                        "detailed",
                        "comprehensive",
                        "describe",
                        "elaborate",
                    ]
                ):
                    complex_messages += 1
                if "?" in content:
                    question_count += 1
                if any(
                    indicator in content
                    for indicator in [
                        "```",
                        "def ",
                        "function",
                        "import ",
                        "class ",
                        "SELECT",
                        "UPDATE",
                        "CREATE",
                    ]
                ):
                    code_messages += 1
                if any(
                    term in content_lower
                    for term in [
                        "algorithm",
                        "database",
                        "api",
                        "server",
                        "client",
                        "protocol",
                        "framework",
                    ]
                ):
                    technical_messages += 1

        valid_messages = [
            m
            for m in conv_messages
            if len(self._safe_get_text_content(m)) >= self.valves.min_message_length
        ]
        avg_length = total_chars / max(len(valid_messages), 1)

        # Calculate complexity score
        complexity_score = 1.0
        if len(valid_messages) > 0:
            complexity_score += (complex_messages / len(valid_messages)) * 0.5
            complexity_score += (question_count / len(valid_messages)) * 0.3
            complexity_score += (code_messages / len(valid_messages)) * 0.4
            complexity_score += (technical_messages / len(valid_messages)) * 0.3
            complexity_score += min(avg_length / 200, 0.5)  # Length factor

        # Recent activity score (more recent = higher score)
        recent_activity = 0
        for msg in conv_messages[-5:]:  # Last 5 messages
            if len(self._safe_get_text_content(msg)) >= self.valves.min_message_length:
                recent_activity += 1
        recent_activity_score = min(recent_activity / 3, 1.0)

        self._debug_log(
            f"Conversation analysis - Total: {len(conv_messages)}, Valid: {len(valid_messages)}, Summaries: {existing_summaries}, Complexity: {complexity_score:.2f}, Activity: {recent_activity_score:.2f}"
        )

        return {
            "total_turns": len(conv_messages),
            "valid_turns": len(valid_messages),
            "has_existing_summary": existing_summaries > 0,
            "summary_count": existing_summaries,
            "complexity_score": complexity_score,
            "avg_message_length": avg_length,
            "recent_activity_score": recent_activity_score,
            "question_count": question_count,
            "code_messages": code_messages,
            "technical_messages": technical_messages,
        }

    def _should_summarize_smart(
        self, conv_state: Dict[str, Any], conversation_id: str
    ) -> bool:
        """Smart decision on whether to summarize"""

        base_threshold = self.valves.summary_trigger_turns
        current_turns = conv_state["valid_turns"]

        # Don't summarize if we just did recently
        if conversation_id in self.last_summary_turn_counts:
            turns_since_last = (
                current_turns - self.last_summary_turn_counts[conversation_id]
            )
            if (
                turns_since_last < base_threshold * 0.6
            ):  # Wait at least 60% of threshold
                self._debug_log(
                    f"Too soon since last summary ({turns_since_last} turns ago)"
                )
                return False

        # Don't summarize if there are existing summaries and not much new content
        if conv_state["has_existing_summary"] and current_turns < base_threshold * 1.5:
            self._debug_log(f"Existing summary present, waiting for more content")
            return False

        # Apply adaptive threshold
        if self.valves.adaptive_threshold:
            # Adjust based on complexity and activity
            complexity_factor = (conv_state["complexity_score"] - 1.0) * 0.3
            activity_factor = conv_state["recent_activity_score"] * 0.2

            adjusted_threshold = base_threshold * (
                1 - complexity_factor - activity_factor
            )
            adjusted_threshold = max(
                adjusted_threshold, base_threshold * 0.5
            )  # Never go below 50%

            self._debug_log(
                f"Adaptive threshold: {adjusted_threshold:.1f} (base: {base_threshold}, complexity: {complexity_factor:.2f}, activity: {activity_factor:.2f})"
            )

            return current_turns >= adjusted_threshold
        else:
            return current_turns >= base_threshold

    def _extract_key_information(self, messages: List[Dict]) -> Dict[str, List[str]]:
        """Extract key information from messages for enhanced summarization"""

        questions = []
        technical_terms = []
        numbers_and_dates = []
        key_decisions = []
        topics = []

        for msg in messages:
            # Use safe content extraction to handle list-based formats
            content = self._safe_get_text_content(msg)
            role = msg.get("role", "")

            # Extract questions
            if "?" in content and role == "user":
                sentences = content.split("?")
                for sentence in sentences[:-1]:  # Exclude last empty part
                    question = sentence.strip()
                    if len(question) > 10:
                        questions.append(question[-150:])  # Last 150 chars

            # Extract technical terms and code
            if any(
                indicator in content
                for indicator in [
                    "```",
                    "def ",
                    "function",
                    "import ",
                    "class ",
                    "SELECT",
                    "CREATE",
                ]
            ):
                if "```" in content:
                    technical_terms.append("code blocks")
                if any(
                    lang in content.lower()
                    for lang in ["python", "javascript", "sql", "html", "css"]
                ):
                    technical_terms.append("programming")
                if any(
                    db in content.lower()
                    for db in ["database", "table", "query", "sql"]
                ):
                    technical_terms.append("database")

            # Extract numbers and dates (improved)
            words = content.split()
            for word in words:
                # Numbers
                if word.replace(",", "").replace(".", "").isdigit() and len(word) <= 6:
                    numbers_and_dates.append(word)
                # Dates
                elif any(
                    month in word.lower()
                    for month in [
                        "jan",
                        "feb",
                        "mar",
                        "apr",
                        "may",
                        "jun",
                        "jul",
                        "aug",
                        "sep",
                        "oct",
                        "nov",
                        "dec",
                    ]
                ):
                    numbers_and_dates.append(word)
                # Years
                elif word.isdigit() and 1900 <= int(word) <= 2030:
                    numbers_and_dates.append(word)

            # Look for decision indicators
            if any(
                decision_word in content.lower()
                for decision_word in [
                    "decided",
                    "conclusion",
                    "result",
                    "solution",
                    "answer",
                    "resolved",
                    "outcome",
                    "final",
                ]
            ):
                if role == "assistant" and len(content) > 50:
                    key_decisions.append(content[:200])

            # Extract topics (first few words of user messages)
            if role == "user" and len(content) > 20:
                first_words = " ".join(content.split()[:8])
                if not any(
                    first_words.lower().startswith(q)
                    for q in ["what", "how", "can", "could", "would", "please"]
                ):
                    topics.append(first_words)

        return {
            "questions": questions[:5],  # Top 5 questions
            "technical_terms": list(set(technical_terms))[:5],
            "numbers_and_dates": list(set(numbers_and_dates))[:8],
            "key_decisions": key_decisions[:3],
            "topics": topics[:4],
        }

    def _create_enhanced_summary(self, messages: List[Dict], quality: str) -> str:
        """Create enhanced summary based on quality setting"""

        # Extract key information
        key_info = self._extract_key_information(messages)

        self._debug_log(
            f"Extracted key info: {len(key_info['questions'])} questions, {len(key_info['technical_terms'])} tech terms, {len(key_info['key_decisions'])} decisions"
        )

        # Build summary based on quality
        if quality == "quick":
            summary_parts = []
            if key_info["questions"]:
                summary_parts.append(
                    f"Discussed {len(key_info['questions'])} main question(s)"
                )
            if key_info["technical_terms"]:
                summary_parts.append(
                    f"including {', '.join(key_info['technical_terms'][:2])}"
                )
            if key_info["key_decisions"]:
                summary_parts.append("with conclusions reached")

            summary = (
                ". ".join(summary_parts) if summary_parts else "General discussion"
            )
            summary += f". Context from {len(messages)} messages preserved."

        elif quality == "detailed":
            summary_parts = []

            # Add questions/topics
            if key_info["questions"]:
                summary_parts.append(
                    f"Key questions addressed: {'; '.join(key_info['questions'][:2])}"
                )

            if key_info["topics"]:
                summary_parts.append(
                    f"Topics covered: {'; '.join(key_info['topics'][:3])}"
                )

            # Add technical context
            if key_info["technical_terms"]:
                summary_parts.append(
                    f"Technical areas: {', '.join(key_info['technical_terms'][:4])}"
                )

            # Add important numbers/dates
            if key_info["numbers_and_dates"] and self.valves.preserve_important_details:
                summary_parts.append(
                    f"Key details mentioned: {', '.join(key_info['numbers_and_dates'][:5])}"
                )

            # Add decisions/conclusions
            if key_info["key_decisions"]:
                summary_parts.append(
                    f"Conclusions: {key_info['key_decisions'][0][:150]}..."
                )

            summary = ". ".join(summary_parts)
            if not summary:
                summary = f"Comprehensive discussion across {len(messages)} messages with detailed exchanges"

            if self.valves.include_context_hints:
                summary += f". Complete context and technical details preserved for seamless continuation."

        else:  # balanced
            summary_parts = []

            # Balanced approach
            if key_info["questions"]:
                summary_parts.append(
                    f"Main topics: {len(key_info['questions'])} key questions/discussions"
                )
                if len(key_info["questions"]) > 0:
                    summary_parts.append(
                        f"including '{key_info['questions'][0][:80]}...'"
                    )

            context_items = []
            if key_info["technical_terms"]:
                context_items.append(f"{', '.join(key_info['technical_terms'][:3])}")
            if key_info["key_decisions"]:
                context_items.append("solutions provided")

            if context_items:
                summary_parts.append(f"Covering {', '.join(context_items)}")

            # Add some key details if available
            if key_info["numbers_and_dates"] and self.valves.preserve_important_details:
                summary_parts.append(
                    f"Key details: {', '.join(key_info['numbers_and_dates'][:4])}"
                )

            summary = ". ".join(summary_parts)
            if not summary:
                summary = f"Ongoing conversation with {len(messages)} substantive message exchanges"

            if self.valves.include_context_hints:
                summary += f". Context preserved for natural continuation."

        # Ensure reasonable length
        max_length = {"quick": 250, "balanced": 500, "detailed": 800}[quality]
        if len(summary) > max_length:
            summary = summary[: max_length - 3] + "..."

        self._debug_log(
            f"Generated {quality} summary ({len(summary)} chars): {summary[:100]}..."
        )

        return summary

    def _get_cache_key(self, messages: List[Dict]) -> str:
        """Generate cache key for messages"""
        if not self.valves.enable_caching:
            return ""

        # Create a hash based on message content
        content_string = (
            f"mode:{'ai' if self.valves.enable_ai_summarization else 'rule'}|"
            f"model:{self.valves.summary_model}|"
            f"quality:{self.valves.summary_quality}|"
        )
        for msg in messages[-25:]:  # Use last 25 messages for key
            content = self._safe_get_text_content(msg)
            content_string += f"{msg.get('role', '')}:{content[:150]}"

        return hashlib.md5(content_string.encode()).hexdigest()[:16]

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
            return None

        message = choices[0].get("message") or {}
        content = message.get("content", "")

        if isinstance(content, str):
            return content.strip() or None

        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text", "").strip()
                    if text:
                        parts.append(text)
                elif isinstance(item, str):
                    text = item.strip()
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
        __user__: Optional[dict] = None,
    ) -> Optional[str]:
        """Generate a summary, preferring the public chat API and falling back to internal helpers."""
        if not __request__ or not __user__ or not __user__.get("id"):
            self._debug_log("AI summarization skipped: missing request or user context")
            return None

        summary = await self._create_ai_summary_via_http(
            messages, summary_model, quality, __request__
        )
        if summary:
            return summary

        self._debug_log("Falling back to internal AI summarization path")

        prompt_messages = self._build_ai_summary_messages(messages, quality)
        if len(prompt_messages) < 2:
            return None

        try:
            from open_webui.main import generate_chat_completion
            from open_webui.models.users import Users
        except Exception as exc:
            self._debug_log(f"AI summarization unavailable: {exc}")
            return None

        user = await Users.get_user_by_id(__user__["id"])
        if not user:
            self._debug_log(
                f"AI summarization skipped: could not load user {__user__['id']}"
            )
            return None

        payload = {
            "model": summary_model,
            "messages": prompt_messages,
            "stream": False,
            "chat_id": f"local:summarizer:{uuid4().hex}",
            "id": f"summarizer-{uuid4().hex}",
        }

        self._debug_log(
            f"Attempting AI summarization with model {summary_model} for {len(messages)} messages"
        )

        generate_signature = inspect.signature(generate_chat_completion)
        if "bypass_filter" in generate_signature.parameters:
            response = await generate_chat_completion(
                __request__, payload, user, bypass_filter=True
            )
        else:
            self._debug_log(
                "AI summarization using legacy chat completion signature without bypass_filter"
            )
            response = await generate_chat_completion(__request__, payload, user)

        summary = self._extract_ai_summary_text(response)

        if summary:
            self._debug_log(
                f"AI summarization succeeded ({len(summary)} chars)"
            )
        else:
            self._debug_log("AI summarization returned no usable text")

        return summary

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
            conv_state = self._analyze_conversation_state(messages)
            conversation_id = f"conv_{len(messages)}_{int(time.time()/100)}"  # Group by ~100 second windows

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
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"📝 Creating {self.valves.summary_quality} summary using {model_display} ({conv_state['valid_turns']} turns → summary + {self.valves.preserve_recent_turns} recent)",
                            "done": False,
                            "hidden": False,
                        },
                    }
                )

                # Get all message types
                system_msgs = [m for m in messages if m.get("role") == "system"]
                conversation_messages = [
                    m for m in messages if m.get("role") in ["user", "assistant"]
                ]

                if len(conversation_messages) > self.valves.preserve_recent_turns:

                    # Split conversation
                    messages_to_summarize = conversation_messages[
                        : -self.valves.preserve_recent_turns
                    ]
                    messages_to_preserve = conversation_messages[
                        -self.valves.preserve_recent_turns :
                    ]

                    self._debug_log(
                        f"Summarizing {len(messages_to_summarize)} messages"
                    )
                    self._debug_log(
                        f"Preserving {len(messages_to_preserve)} recent messages"
                    )

                    # Create enhanced summary
                    summary_source = "rule-based"
                    if cached_summary:
                        summary_text = cached_summary
                        summary_source = "cached"
                        self._debug_log(f"Using cached summary")
                    else:
                        # Try AI summarization if enabled
                        summary_text = None
                        if self.valves.enable_ai_summarization:
                            try:
                                summary_text = await self._create_ai_summary(
                                    messages_to_summarize,
                                    summary_model,
                                    self.valves.summary_quality,
                                    __request__,
                                    __user__,
                                )
                                if summary_text:
                                    summary_source = "ai"
                                    self._debug_log(
                                        f"Generated AI summary using {summary_model}"
                                    )
                            except Exception as e:
                                self._debug_log(
                                    f"AI summarization failed: {str(e)}, falling back to enhanced rule-based"
                                )

                        # Fall back to enhanced rule-based summarization
                        if not summary_text:
                            summary_text = self._create_enhanced_summary(
                                messages_to_summarize, self.valves.summary_quality
                            )
                            summary_source = "rule-based"
                            self._debug_log(
                                f"Generated enhanced rule-based {self.valves.summary_quality} summary"
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
                    model_note = (
                        f" via {summary_model}" if summary_model != model else ""
                    )
                    summary_message = {
                        "role": "system",
                        "content": f"📋 **Conversation Summary** ({len(messages_to_summarize)} messages, {self.valves.summary_quality} quality{model_note}):\n\n{summary_text}\n\n---\n*Recent messages continue below*",
                    }
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
                    self.last_summary_turn_counts[conversation_id] = conv_state[
                        "valid_turns"
                    ]

                    self._debug_log(
                        f"Final message count: {len(new_messages)} (was {len(messages)})"
                    )
                    self._debug_log(f"Performance stats: {self.performance_stats}")

                    # Create status message with model info
                    source_label = {
                        "ai": "AI",
                        "cached": "cached",
                        "rule-based": "rule-based",
                    }[summary_source]
                    status_msg = f"✅ {source_label.capitalize()} summary created using {model_display}! {len(messages)} → {len(new_messages)} messages"

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
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": f"✋ No summarization needed ({conv_state['valid_turns']}/{self.valves.summary_trigger_turns} turns, complexity: {conv_state['complexity_score']:.2f}) | Model: {summary_model}",
                                "done": True,
                                "hidden": False,
                            },
                        }
                    )

            self.conversation_count += 1
            self._debug_log(f"Filter processed conversation #{self.conversation_count}")

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
