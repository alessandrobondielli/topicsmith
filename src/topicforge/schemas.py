"""The structured outputs the model is constrained to produce.

Two of these are dynamic, and both for the same reason -- the constraint has to
match the project, not the code:

* :func:`paper_topics_model` adds whatever fields ``topics.extra_fields``
  declares, so asking every paper for, say, the languages it studies or the
  clinical population it recruits costs one line of config.
* :func:`assignment_model` makes ``main_area`` a ``Literal`` over the taxonomy's
  own area names. That is what makes the final counts trustworthy: an
  off-taxonomy answer fails *validation*, not merely a schema hint, so even a
  backend that ignores JSON-schema constraints cannot silently invent a
  category.

Note that a model's JSON schema is part of the response cache key. Adding an
extra field therefore re-pays pass 1 -- deliberately, since the old answers do
not contain it.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, create_model, field_validator

# What `topics.extra_fields` may ask for. Anything richer belongs in a prompt.
FIELD_TYPES: dict[str, Any] = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list[str]": list[str],
}

CORE_TOPIC_FIELDS = (
    "main_topic", "subtopics", "task", "method_family",
    "contribution_type", "one_line_summary",
)


class SchemaError(RuntimeError):
    pass


# --------------------------------------------------------------------- pass 1

class PaperTopics(BaseModel):
    """Free-form description of one paper. No fixed vocabulary anywhere."""

    main_topic: str
    subtopics: list[str] = Field(min_length=1, max_length=10)
    task: str
    method_family: str
    contribution_type: str
    one_line_summary: str


def extra_fields(cfg) -> list[dict[str, Any]]:
    """Validated ``topics.extra_fields`` entries."""
    declared = cfg.topics.get("extra_fields") or []
    out = []
    for entry in declared:
        if not isinstance(entry, dict) or "name" not in entry:
            raise SchemaError(
                f"topics.extra_fields entry {entry!r} needs at least a 'name'."
            )
        kind = entry.get("type", "str")
        if kind not in FIELD_TYPES:
            raise SchemaError(
                f"topics.extra_fields['{entry['name']}'].type is {kind!r}; "
                f"expected one of {', '.join(FIELD_TYPES)}."
            )
        if entry["name"] in CORE_TOPIC_FIELDS:
            raise SchemaError(
                f"topics.extra_fields cannot redefine the core field '{entry['name']}'."
            )
        out.append({"name": entry["name"], "type": kind,
                    "description": entry.get("description", "")})
    return out


def paper_topics_model(cfg) -> type[BaseModel]:
    """:class:`PaperTopics` plus the project's declared extra fields."""
    fields = extra_fields(cfg)
    if not fields:
        return PaperTopics
    definitions: dict[str, Any] = {
        f["name"]: (FIELD_TYPES[f["type"]], ...) for f in fields
    }
    return create_model("PaperTopicsExtended", __base__=PaperTopics, **definitions)


# --------------------------------------------------------------------- pass 2

def _merge_duplicate_names(areas: list) -> list:
    """Collapse areas that came back under the same name.

    Nothing in a JSON schema can require the entries of an array to be distinct,
    so a model can return two areas both called "Prompting". Left alone that
    produces a duplicated enum value in pass 3 and two rows in every count that
    should be one. Merge them instead of failing: the content is usually
    complementary, and a hard error here would throw away an otherwise good
    taxonomy.
    """
    merged: dict[str, Any] = {}
    for area in areas:
        key = area.name.strip().casefold()
        if key not in merged:
            area.name = area.name.strip()
            merged[key] = area
            continue
        kept = merged[key]
        kept.subtopics = list(dict.fromkeys(kept.subtopics + area.subtopics))[:10]
        kept.absorbs = list(dict.fromkeys(kept.absorbs + area.absorbs))
    return list(merged.values())


class InducedArea(BaseModel):
    """One area, as the model returns it."""

    name: str
    definition: str
    subtopics: list[str] = Field(min_length=1, max_length=10)
    absorbs: list[str] = Field(default_factory=list)


class InducedTaxonomy(BaseModel):
    areas: list[InducedArea] = Field(min_length=2, max_length=40)

    @field_validator("areas")
    @classmethod
    def _unique_names(cls, areas: list[InducedArea]) -> list[InducedArea]:
        return _merge_duplicate_names(areas)


class Area(InducedArea):
    """One area as stored: the induced fields plus provenance."""

    seeded: bool = False


class Taxonomy(BaseModel):
    """The canonical taxonomy, as written to (and read back from) YAML."""

    areas: list[Area] = Field(min_length=1)

    @field_validator("areas")
    @classmethod
    def _unique_names(cls, areas: list[Area]) -> list[Area]:
        return _merge_duplicate_names(areas)

    @classmethod
    def from_induced(cls, induced: InducedTaxonomy, seed_names: set[str] | None = None) -> "Taxonomy":
        seeds = {n.casefold() for n in (seed_names or set())}
        return cls(areas=[
            Area(**area.model_dump(), seeded=area.name.casefold() in seeds)
            for area in induced.areas
        ])

    def names(self) -> list[str]:
        return [a.name for a in self.areas]

    def all_subtopics(self) -> list[str]:
        seen, out = set(), []
        for area in self.areas:
            for sub in area.subtopics:
                if sub.casefold() not in seen:
                    seen.add(sub.casefold())
                    out.append(sub)
        return out

    def render(self) -> str:
        """The taxonomy as the assignment prompt shows it to the model."""
        return "\n\n".join(
            f"## {a.name}\n{a.definition}\nSubtopics: {', '.join(a.subtopics)}"
            for a in self.areas
        )


# --------------------------------------------------------------------- pass 3

def assignment_model(taxonomy: Taxonomy) -> type[BaseModel]:
    """A schema whose area fields are enums over the taxonomy's own names."""
    names = tuple(taxonomy.names())
    if not names:
        raise SchemaError("The taxonomy has no areas.")
    area_type = Literal[names]  # type: ignore[valid-type]
    secondary_type = Literal[names + ("none",)]  # type: ignore[valid-type]
    return create_model(
        "Assignment",
        main_area=(area_type, ...),
        secondary_area=(secondary_type, "none"),
        subtopics=(list[str], Field(min_length=1, max_length=6)),
        confidence=(float, Field(ge=0.0, le=1.0)),
        why=(str, ""),
    )
