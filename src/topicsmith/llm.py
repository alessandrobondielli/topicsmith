"""Structured LLM access against an OpenAI-compatible endpoint (vLLM).

Three things make this worth having as its own module:

*Structured output.*  vLLM has exposed JSON-schema constraints under different
names across versions. We probe once for the mode the server actually supports
and reuse it, falling back to prompt-only JSON if none work.

*Caching.*  Every response is written to disk keyed on model + prompt + schema
+ temperature.  Reruns issue zero requests, which is what makes the notebook
re-runnable and lets a taxonomy be rebuilt without re-reading every paper.

*Concurrency.*  vLLM batches well, so requests go out concurrently under a
semaphore, with retries and per-item error capture rather than an aborted batch.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence, TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError
from tqdm.auto import tqdm

from .config import Config, load_config

T = TypeVar("T", bound=BaseModel)

# Ordered by preference; the first that the server accepts wins.
STRUCTURED_MODES = ("json_schema", "guided_json", "prompt")

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_THINK = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN = re.compile(r"<(think|thinking|reasoning)>", re.IGNORECASE)


class LLMError(RuntimeError):
    pass


def _strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """JSON schema for *model*, self-contained and tightened for strict decoding.

    Three transformations, all of which matter in practice:

    * **Inline every ``$ref``.** Pydantic emits nested models as references into
      ``$defs``. Backends differ in whether they resolve those, and the
      prompt-only fallback shows the schema to the model verbatim, where a
      dangling reference is simply unreadable. Inlining removes the question.
    * **``additionalProperties: false``** on every object, and every property
      required -- what strict JSON-schema decoding expects.
    * **Drop annotation noise** (``title``, ``default``) that only enlarges the
      prompt.

    Recursive models would not terminate here; none of the schemas in this
    pipeline are recursive, and a depth guard makes that failure loud rather
    than a hang.
    """
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def resolve(node: Any, depth: int = 0) -> Any:
        if depth > 24:
            raise LLMError(f"schema for {model.__name__} nests too deeply to inline")
        if isinstance(node, list):
            return [resolve(v, depth + 1) for v in node]
        if not isinstance(node, dict):
            return node

        if "$ref" in node:
            name = node["$ref"].rsplit("/", 1)[-1]
            if name not in defs:
                raise LLMError(f"unresolvable $ref {node['$ref']} in {model.__name__}")
            merged = {**defs[name], **{k: v for k, v in node.items() if k != "$ref"}}
            return resolve(merged, depth + 1)

        out = {k: resolve(v, depth + 1) for k, v in node.items() if k not in ("title", "default")}
        if out.get("type") == "object" and "properties" in out:
            out["additionalProperties"] = False
            out["required"] = list(out["properties"])
        return out

    return resolve(schema)


def strip_reasoning(text: str) -> str:
    """Remove a reasoning model's think block from *text*.

    Needed only when vLLM runs without a ``--reasoning-parser``: the whole
    generation, ``<think>`` block included, then arrives in ``message.content``
    instead of being split into ``reasoning_content``. Left in place it defeats
    the brace-matching fallback below, because a model reasoning about a JSON
    schema writes braces while it thinks -- so the failure is content-dependent
    and shows up on some items and not others.

    An unclosed opener means the generation stopped mid-thought; everything from
    it onwards is thinking, and there is no answer to recover.
    """
    text = _THINK.sub("", text)
    opener = _THINK_OPEN.search(text)
    return (text[: opener.start()] if opener else text).strip()


def extract_json(text: str) -> Any:
    """Pull a JSON value out of a model response that may be chatty or fenced."""
    raw = (text or "").strip()
    text = strip_reasoning(raw)
    if not text:
        raise LLMError(
            "no content outside the reasoning block -- the model spent its whole "
            "token budget thinking. Raise llm.max_tokens."
            if raw else "empty response"
        )
    for candidate in (text, *(m.group(1) for m in _FENCE.finditer(text))):
        candidate = candidate.strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    # Last resort: the outermost balanced {...} or [...] in the response.
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if 0 <= start < end:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise LLMError(f"could not parse JSON from response: {text[:400]!r}")


@dataclass
class Usage:
    """Running totals for a session, reported at the end of each command."""

    requests: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)

    def summary(self) -> str:
        # completion_tokens already includes the reasoning tokens; the split is
        # shown because it is the number that decides whether max_tokens is sane.
        thinking = f" (of which {self.reasoning_tokens:,} reasoning)" if self.reasoning_tokens else ""
        return (
            f"{self.requests} request(s), {self.cache_hits} cache hit(s), "
            f"{self.prompt_tokens:,} prompt + {self.completion_tokens:,} completion tokens"
            f"{thinking}, {len(self.failures)} failure(s)"
        )


class _Probe(BaseModel):
    ok: bool


class LLMClient:
    """Cached, concurrent, schema-constrained access to the configured model."""

    def __init__(self, cfg: Config | None = None, *, cache: bool = True):
        self.cfg = cfg or load_config()
        settings = self.cfg.llm
        if not settings.get("model"):
            raise LLMError(
                "No model configured. Set TOPICSMITH_MODEL (and OPENAI_BASE_URL) in "
                "the environment, or fill in llm.model in config.yaml."
            )
        self.model: str = settings["model"]
        self.temperature: float = settings.get("temperature", 0.0)
        self.max_tokens: int = settings.get("max_tokens", 4096)
        self.max_retries: int = settings.get("max_retries", 4)
        self.concurrency: int = settings.get("concurrency", 16)
        self.cache_dir: Path | None = self.cfg.paths.llm_cache if cache else None
        self.usage = Usage()
        self._mode: str | None = None
        self._client = AsyncOpenAI(
            base_url=settings["base_url"],
            api_key=settings.get("api_key") or "EMPTY",
            timeout=settings.get("timeout", 300),
            max_retries=0,  # retries are handled here, so backoff is visible
        )

    # ---------------------------------------------------------------- caching

    def _cache_path(self, key: dict[str, Any]) -> Path | None:
        if self.cache_dir is None:
            return None
        blob = json.dumps(key, sort_keys=True, ensure_ascii=False, default=str)
        return self.cache_dir / f"{hashlib.sha256(blob.encode()).hexdigest()}.json"

    # ------------------------------------------------------------- the call

    def _request_kwargs(self, mode: str, schema: dict[str, Any] | None) -> dict[str, Any]:
        if schema is None or mode == "prompt":
            return {}
        if mode == "json_schema":
            return {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "response", "schema": schema, "strict": True},
                }
            }
        if mode == "guided_json":
            return {"extra_body": {"guided_json": schema}}
        raise ValueError(f"unknown structured-output mode: {mode}")

    async def _raw_call(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any] | None,
        mode: str,
        temperature: float,
    ) -> str:
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=self.max_tokens,
            **self._request_kwargs(mode, schema),
        )
        if response.usage:
            self.usage.prompt_tokens += response.usage.prompt_tokens or 0
            self.usage.completion_tokens += response.usage.completion_tokens or 0
            details = getattr(response.usage, "completion_tokens_details", None)
            self.usage.reasoning_tokens += getattr(details, "reasoning_tokens", 0) or 0
        self.usage.requests += 1

        choice = response.choices[0]
        if choice.finish_reason == "length":
            # A truncated generation can never satisfy the schema, and retrying it
            # at a fixed temperature reproduces it exactly. Say what happened, so
            # the budget gets raised instead of the endpoint getting blamed.
            spent = response.usage.completion_tokens if response.usage else "?"
            raise LLMError(
                f"response hit the {self.max_tokens}-token completion limit "
                f"({spent} generated). Raise llm.max_tokens in config.yaml; a "
                "reasoning model charges its thinking against this same budget."
            )
        return choice.message.content or ""

    async def detect_mode(self) -> str:
        """Find the structured-output mode this server supports. Probes once."""
        if self._mode is not None:
            return self._mode
        schema = _strict_schema(_Probe)
        messages = [{"role": "user", "content": 'Reply with the JSON object {"ok": true}.'}]
        reasons: list[str] = []
        for mode in STRUCTURED_MODES:
            try:
                text = await self._raw_call(messages, schema, mode, 0.0)
                _Probe.model_validate(extract_json(text))
            except Exception as exc:  # noqa: BLE001 - any failure means "try the next mode"
                reasons.append(f"  {mode}: {type(exc).__name__}: {str(exc)[:200]}")
                continue
            self._mode = mode
            return mode
        raise LLMError(
            "The endpoint answered none of the structured-output modes.\n"
            + "\n".join(reasons)
            + "\n\nCheck base_url and model. If the model reasons, vLLM must be "
            "started with a matching --reasoning-parser (deepseek_r1, qwen3, ...), "
            "otherwise the think block arrives inside message.content."
        )

    async def complete(
        self,
        prompt: str,
        schema: type[T],
        *,
        system: str | None = None,
        temperature: float | None = None,
        cache_salt: str = "",
    ) -> T:
        """One schema-constrained completion, served from cache when possible.

        ``cache_salt`` lets a caller deliberately break cache reuse -- the
        taxonomy stability check passes the run index so repeated inductions at
        the same temperature do not collapse into one cached answer.
        """
        temperature = self.temperature if temperature is None else temperature
        json_schema = _strict_schema(schema)
        key = {
            "model": self.model,
            "system": system,
            "prompt": prompt,
            "schema": json_schema,
            "temperature": temperature,
            "salt": cache_salt,
        }
        path = self._cache_path(key)
        if path is not None and path.is_file():
            self.usage.cache_hits += 1
            try:
                return schema.model_validate(json.loads(path.read_text(encoding="utf-8")))
            except (ValidationError, json.JSONDecodeError):
                path.unlink(missing_ok=True)  # stale cache from an older schema

        mode = await self.detect_mode()
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        content = prompt
        if mode == "prompt":
            content += (
                "\n\nRespond with a single JSON object matching this schema, and nothing else:\n"
                + json.dumps(json_schema, ensure_ascii=False)
            )
        messages.append({"role": "user", "content": content})

        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                text = await self._raw_call(messages, json_schema, mode, temperature)
                result = schema.model_validate(extract_json(text))
            except Exception as exc:  # noqa: BLE001
                last = exc
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(min(2 ** attempt, 30))
                continue
            if path is not None:
                path.write_text(
                    result.model_dump_json(indent=2), encoding="utf-8"
                )
            return result
        raise LLMError(f"failed after {self.max_retries} attempts: {last}") from last

    # ------------------------------------------------------------- batching

    async def map_async(
        self,
        items: Sequence[Any],
        build_prompt: Callable[[Any], str],
        schema: type[T],
        *,
        system: str | None = None,
        temperature: float | None = None,
        label: str = "llm",
        describe: Callable[[Any], str] | None = None,
    ) -> list[T | None]:
        """Run one completion per item concurrently, preserving input order.

        A failing item yields ``None`` and is recorded in ``self.usage.failures``
        rather than taking the whole batch down.
        """
        if not items:
            return []
        await self.detect_mode()  # probe once, before the fan-out
        semaphore = asyncio.Semaphore(self.concurrency)
        bar = tqdm(total=len(items), desc=label, unit="item")

        async def one(item: Any) -> T | None:
            async with semaphore:
                try:
                    return await self.complete(
                        build_prompt(item), schema, system=system, temperature=temperature
                    )
                except Exception as exc:  # noqa: BLE001
                    name = describe(item) if describe else str(item)[:80]
                    self.usage.failures.append((name, str(exc)))
                    return None
                finally:
                    bar.update(1)

        try:
            return await asyncio.gather(*(one(i) for i in items))
        finally:
            bar.close()

    def map(self, *args: Any, **kwargs: Any) -> list[Any]:
        """Synchronous wrapper around :meth:`map_async`, for scripts and notebooks."""
        return run_sync(self.map_async(*args, **kwargs))

    def one(self, *args: Any, **kwargs: Any) -> Any:
        """Synchronous wrapper around :meth:`complete`."""
        return run_sync(self.complete(*args, **kwargs))

    def run_mode(self) -> str:
        """Synchronous wrapper around :meth:`detect_mode`."""
        return run_sync(self.detect_mode())


def run_sync(coro):
    """Run *coro*, tolerating an already-running loop (Jupyter).

    Under IPython the kernel owns the event loop, so ``asyncio.run`` would fail;
    nest_asyncio is applied only in that case, and only if available.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    try:
        import nest_asyncio  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise LLMError(
            "An event loop is already running. Install nest_asyncio, or await the "
            "*_async variant directly."
        ) from exc
    nest_asyncio.apply()
    return asyncio.get_event_loop().run_until_complete(coro)


def cache_stats(cfg: Config | None = None) -> dict[str, Any]:
    """Size and entry count of the on-disk response cache."""
    cfg = cfg or load_config()
    files = list(cfg.paths.llm_cache.glob("*.json"))
    return {
        "entries": len(files),
        "megabytes": round(sum(f.stat().st_size for f in files) / 1e6, 2),
        "path": str(cfg.paths.llm_cache),
    }
