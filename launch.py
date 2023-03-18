from pathlib import Path
from typing import List, Optional
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
    model_path: str = "models/lm/chatglm-6b",
    prec: str = "fp16",
    device: Optional[List[int]] = None,
    listen: bool = True,
    port: int = 7860,
    share: bool = False,
    debug: bool = False,
):
    # config logger
    logger.remove()  # remove default stdout logger
    logger.add(
        RichHandler(markup=True, console=sym_tbl().console),
        level="DEBUG" if debug else "INFO",
    )

    # setup dir if not exists
    Path(sym_tbl().cfg["history_dir"]).mkdir(parents=True, exist_ok=True)

    sym_tbl().cfg.update({
        "model_path": model_path,
        "prec": prec,
        "device": device,
        "listen": listen,
        "port": port,
        "share": share,
        "debug": debug,
    })

    sym_tbl().device_info = alloc1([] if device is None else device)
    sym_tbl().device = torch.device(sym_tbl().device_info["device"])

    update_proto()
    sym_tbl().proto.history.new()
    if not debug:
        sym_tbl().proto.model.load()

    with torch.no_grad():
        ui = create_ui()
        ui.queue().launch(
            server_name="0.0.0.0" if listen else None,
            server_port=port,
            share=share
        )


if __name__ == "__main__":
    app()
