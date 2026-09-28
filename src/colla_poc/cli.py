from __future__ import annotations
from pathlib import Path
from datetime import datetime
import json
import logging
import typer
from rich.console import Console
from .pipeline import analyze as run_analyze
from .evaluation import evaluate as run_evaluate

app = typer.Typer(help='Colla 로컬 사진 자동 큐레이션 PoC')
console = Console()

@app.command()
def analyze(photos: Path, output: Path | None = typer.Option(None, help='결과 저장 폴더'), config: Path | None = typer.Option(None, help='YAML 설정'), source_directory_as_user: bool = typer.Option(False, help='1차 하위 폴더를 촬영자 ID로 사용')) -> None:
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    target = output or Path('runs')/datetime.now().strftime('%Y%m%d-%H%M%S')
    try:
        metrics = run_analyze(photos, target, config, source_directory_as_user)
    except Exception as exc:
        console.print(f'[red]분석 실패: {exc}[/red]')
        raise typer.Exit(1)
    console.print(f'[green]완료[/green] {target.resolve()}')
    console.print_json(data=metrics)

@app.command()
def ui(run: Path) -> None:
    from .ui import build_ui
    build_ui(run.resolve()).launch(inbrowser=True, allowed_paths=[str(run.resolve())])

@app.command()
def evaluate(run: Path, labels: Path = typer.Option(..., help='Ground truth CSV')) -> None:
    try:
        result = run_evaluate(run.resolve(), labels.resolve())
    except Exception as exc:
        console.print(f'[red]평가 실패: {exc}[/red]')
        raise typer.Exit(1)
    (run/'evaluation.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
    console.print_json(data=result)

if __name__ == '__main__':
    app()
