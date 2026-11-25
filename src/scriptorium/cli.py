"""Command line interface.

Three commands, each corresponding to something a reviewer of this project would
want to run: check that the configuration is honest, extract a document, and
produce the metric table.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich.console import Console
from rich.table import Table

from scriptorium.adapters import EpubAdapter, Provenance
from scriptorium.config import ModelsConfig, Settings, load_glossary
from scriptorium.evaluation import (
    Harness,
    load_test_set,
    render_markdown_table,
    sloppy_recordings,
)
from scriptorium.orchestrator import TranslationLoop
from scriptorium.providers import EchoProvider, ScriptedProvider
from scriptorium.segmentation.placeholders import protect
from scriptorium.terminology import TerminologyStore

app = typer.Typer(add_completion=False, help="Terminology-controlled machine translation.")
console = Console()


def _models_config(settings: Settings) -> ModelsConfig:
    return ModelsConfig.load(settings.models_config)


@app.command("check-config")
def check_config() -> None:
    """Report whether model ids and prices are really pinned.

    Exits non-zero while placeholders remain, so a billed run cannot be started
    by accident with a cost column that would be fiction.
    """
    settings = Settings()
    config = _models_config(settings)

    table = Table(title=f"Model pins ({config.provider})")
    table.add_column("Role")
    table.add_column("Model")
    table.add_column("Effort")
    table.add_column("Pinned")
    for role, spec in config.roles.items():
        table.add_row(
            role,
            spec.model,
            spec.reasoning_effort or "-",
            "[green]yes[/]" if spec.is_pinned else "[red]no[/]",
        )
    console.print(table)
    console.print(f"pinned from: {config.pinned_from or '[red]unset[/]'}")
    console.print(f"pinned on:   {config.pinned_on or '[red]unset[/]'}")

    if not config.is_pinned:
        console.print(
            "\n[red]Not pinned.[/] Fill in real model ids and prices from the provider's "
            "official pricing page, and record the date you read it."
        )
        raise typer.Exit(code=1)
    console.print("\n[green]Pinned.[/] A billed run is allowed.")


@app.command()
def extract(
    path: Annotated[Path, typer.Argument(help="Path to the document")],
    manifest: Annotated[Path, typer.Option(help="YAML file with licence and source")],
    out: Annotated[Path, typer.Option(help="Where to write segments as JSONL")] = Path(
        "data/interim/segments.jsonl"
    ),
) -> None:
    """Extract segments from a document.

    The manifest is mandatory: a document that cannot state its source and licence
    is refused at ingestion (docs/data-policy.md).
    """
    provenance = Provenance.model_validate(yaml.safe_load(manifest.read_text(encoding="utf-8")))
    extracted = EpubAdapter().extract(path, provenance)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for segment in extracted.segments:
            handle.write(segment.model_dump_json() + "\n")

    console.print(
        f"[green]{len(extracted.segments)}[/] segments in "
        f"[green]{len(extracted.blocks)}[/] blocks -> {out}"
    )


@app.command("eval")
def run_eval(
    config_path: Annotated[Path, typer.Option("--config")] = Path("config/eval.yaml"),
    glossary_path: Annotated[Path, typer.Option("--glossary")] = Path("config/glossary.yaml"),
    report: Annotated[Path, typer.Option(help="Where to write the Markdown report")] = Path(
        "reports/evaluation-run.md"
    ),
) -> None:
    """Run the evaluation arms and print the metric table.

    With no billed provider configured, this runs the two checks that need no
    model at all:

    1. **The instrument.** An echo floor and a reference-replaying ceiling. If
       the ceiling does not top out on every metric, the harness is broken and
       no comparison between real systems would mean anything.
    2. **The loop.** A provider that makes terminology errors, run with and
       without the orchestrator. If terminology compliance does not rise, the
       agentic layer does not do what this project says it does.

    Neither check says anything about translation quality, and the report says so
    where it would otherwise be easy to misread.
    """
    settings = Settings()
    spec = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    first_set = next(iter(spec["test_sets"].values()))
    cases = load_test_set(Path(first_set["path"]))
    glossary = load_glossary(glossary_path) if glossary_path.exists() else []

    harness = Harness(
        cases,
        source_lang=spec["source_lang"],
        target_lang=spec["target_lang"],
        glossary=glossary,
    )

    oracle = {protect(c.source).text: protect(c.reference).text for c in cases}
    sloppy = sloppy_recordings(cases, glossary)

    def scripted() -> ScriptedProvider:
        return ScriptedProvider(sloppy.translations, revisions=sloppy.revisions)

    loop = TranslationLoop(
        scripted(),
        source_lang=spec["source_lang"],
        target_lang=spec["target_lang"],
        terminology=TerminologyStore(glossary),
        max_revisions=2,
    )

    results = [
        harness.run_arm("floor (echo)", EchoProvider()),
        harness.run_arm("ceiling (oracle)", ScriptedProvider(oracle), with_context=True),
        harness.run_arm("sloppy (no loop)", scripted(), with_context=True),
        harness.run_arm("sloppy + loop", loop, with_context=True),
    ]

    config = _models_config(settings)
    if not config.is_pinned:
        console.print(
            "[yellow]Arms B1/B2 are not running:[/] config/models.yaml still holds "
            "placeholder model ids. Pin them from the provider's official pricing "
            "page, then re-run.\n"
        )

    table = render_markdown_table(results)
    console.print(table)

    instrument = Harness.compare(results[0], results[1])
    console.print(
        f"\ninstrument — ceiling vs floor, chrF++: {instrument.mean_difference:+.2f} "
        f"(95% CI {instrument.ci_low:+.2f}..{instrument.ci_high:+.2f}, "
        f"p={instrument.p_value:.3f}, n={len(cases)})"
    )

    before, after = results[2], results[3]
    gain = after.terminology_compliance - before.terminology_compliance
    console.print(
        f"loop — terminology compliance: {before.terminology_compliance:.1%} -> "
        f"{after.terminology_compliance:.1%} ({gain:+.1%}) over "
        f"{sloppy.spoiled}/{len(cases)} deliberately broken segments"
    )
    if sloppy.spoiled == 0:
        console.print(
            "[yellow]Nothing was broken:[/] no reference in this test set contains a "
            "term with a forbidden variant, so the two loop arms are the same arm."
        )

    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# Evaluation run\n\n"
        f"Test set: `{first_set['path']}` ({len(cases)} segments, "
        f"{spec['source_lang']} -> {spec['target_lang']})\n\n"
        "No billed provider was used. The arms below are offline self-checks: the\n"
        "string metrics are a floor and a ceiling, not a result, because the\n"
        "scripted providers replay the references themselves.\n\n"
        f"{table}\n\n"
        "## Is the instrument working?\n\n"
        "Paired bootstrap on per-segment chrF++, ceiling vs floor: "
        f"{instrument.mean_difference:+.2f} "
        f"(95% CI {instrument.ci_low:+.2f}..{instrument.ci_high:+.2f}, "
        f"p={instrument.p_value:.3f})\n\n"
        "## Does the loop recover terminology errors?\n\n"
        f"A provider was derived from the references with one accepted term form\n"
        f"swapped for a forbidden variant in {sloppy.spoiled} of {len(cases)} segments. "
        "Run without the\n"
        "orchestrator, that defect reaches the output; run with it, the "
        "deterministic\ncheck catches it and one revision fixes it.\n\n"
        f"- terminology compliance without the loop: {before.terminology_compliance:.1%}\n"
        f"- terminology compliance with the loop: {after.terminology_compliance:.1%}\n"
        f"- cost of the loop on this run: ${after.cost_usd:.4f} "
        "(the scripted provider is free; the figure is here so that it stops being "
        "zero the moment a real provider is pinned)\n",
        encoding="utf-8",
    )
    console.print(f"\nwritten to {report}")


if __name__ == "__main__":
    app()
