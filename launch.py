import json
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional
from loguru import logger
from rich.logging import RichHandler
import typer
from modules.device import alloc1
from modules.sym import sym_tbl
from modules.ui import create_ui
from modules.proto import update_proto
import torch


app = typer.Typer()


@app.command()
def main(
    cfg: str = typer.Argument(..., help="config path"),
    device: Optional[List[int]] = typer.Option(None, help="Visible device list. If set as -1, will only use cpu."),
    listen: bool = typer.Option(True, help="Used in gradio.launch()"),
    port: int = typer.Option(7860, help="Used in gradio.launch()"),
    share: bool = typer.Option(False, help="Used in gradio.launch()"),
    debug: bool = typer.Option(False, help="Debug mode. Will change log level."),
    load_model: bool = typer.Option(True, help="Load model or not. Useful in debugging ui."),
    history: bool = typer.Option(True, help="Save history to history dir or not. If no, will use a tmpdir."),
):
    # config logger
    logger.remove()  # remove default stdout logger
    logger.add(
        RichHandler(markup=True, console=sym_tbl().console),
        level="DEBUG" if debug else "INFO",
        # rich handler已经自带了时间、level和代码所在行
        format="{message}",
    )

    # setup dir if not exists
    Path(sym_tbl().cfg["history_dir"]).mkdir(parents=True, exist_ok=True)
    Path(sym_tbl().cfg["tmp_dir"]).mkdir(parents=True, exist_ok=True)

    console_args = {
        "cfg": cfg,
        "device": device,
        "listen": listen,
        "port": port,
        "share": share,
        "debug": debug,
        "load_model": load_model,
        "history": history,
    }

    with tempfile.TemporaryDirectory(dir=sym_tbl().cfg["tmp_dir"]) as f:
        sym_tbl().tmp_dir = Path(f)

        with Path(cfg).open('r', encoding="utf8") as rf:
            file_args: Dict[str, Any] = json.load(rf)
        sym_tbl().cfg.update(file_args)
        sym_tbl().cfg.update(console_args)

        update_proto()
        if sym_tbl().cfg["load_model"]:
            sym_tbl().device_info = alloc1([] if sym_tbl().cfg["device"] is None else sym_tbl().cfg["device"])
            sym_tbl().device = torch.device(sym_tbl().device_info["device"])
            sym_tbl().proto.model.load()

        ui = create_ui()
        ui.queue().launch(
            server_name="0.0.0.0" if listen else None,
            server_port=port,
            share=share,
            # prevent_thread_lock=True,
        )
        ui.close()


if __name__ == "__main__":
    app()
