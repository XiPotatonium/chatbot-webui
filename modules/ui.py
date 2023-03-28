from pathlib import Path
from typing import List, Optional
import gradio as gr
from loguru import logger
from .sym import sym_tbl


_css = """
#icon-btn {
    max-width: 2.5em;
    min-width: 2.5em !important;
    height: 2.5em;
    margin: 1.5em 0;
}
"""

# #chatbot {
#     height: 100%;
#     overflow: auto !important;
# }


def send(instruction: str, msg: str, mm_ty, img, audio, video):
    if not sym_tbl().history.folder.exists():
        # lazy create dir
        sym_tbl().history.save()

    new_storage = sym_tbl().history.storage_meta()
    new_storage["query"]["text"] = msg if msg is not None else ""
    new_storage["query"]["instruction"] = instruction if instruction is not None else ""
    if (
        (mm_ty == "Image" and img is not None) or
        (mm_ty == "Audio" and audio is not None) or
        (mm_ty == "Video" and video is not None)
    ):
        new_storage["query"]["mm_type"] = mm_ty
        new_storage["query"]["mm_path"] = sym_tbl().history.save_media(img).name
    sym_tbl().history.storage.append(new_storage)
    sym_tbl().history.append_last_query_binding()

    # logger.debug(new_storage)

    return sym_tbl().history.binding


def predict(*args, **kwargs):
    if sym_tbl().cfg.get("support_stream", False):
        for _ in sym_tbl().model.stream_generate(*args, **kwargs):
            yield sym_tbl().history.binding
    else:
        sym_tbl().model.generate(*args, **kwargs)
        yield sym_tbl().history.binding


def lst_chats() -> List[str]:
    return [f.name for f in Path(sym_tbl().cfg["history_dir"]).iterdir() if f.is_dir()]


def refresh_chats():
    return gr.update(choices=lst_chats())


def create_ui():
    with gr.Blocks(css=_css + sym_tbl().proto.ui.css) as ui:
        with gr.Row():
            with gr.Column(scale=7, elem_id="chatpanel"):
                chatbot = gr.Chatbot(elem_id="chatbot", show_label=False).style(height=700)

            with gr.Column(scale=3):
                with gr.Row():
                    with gr.Column(variant="panel"):
                        with gr.Row():
                            dp_chats = gr.Dropdown(show_label=False, choices=lst_chats())
                            btn_refresh_chats = gr.Button("♻", elem_id="icon-btn")
                        with gr.Row():
                            btn_new_chat = gr.Button("New Chat")

                with gr.Row():
                    with gr.Column(variant="panel"):
                        with gr.Accordion("Instruction", open=False):
                            instruction = gr.Textbox(lines=2, placeholder="Your instruction here...", show_label=False).style(container=False)
                        with gr.Row():
                            msg = gr.Textbox(label="Input", placeholder="Type your text...", show_label=False, lines=2).style(container=False)
                        with gr.Row():
                            with gr.Accordion("Media", open=False):
                                with gr.Row():
                                    dp_mm = gr.Dropdown(["Image", "Audio", "Video"], label="Use", show_label=True)
                                with gr.Row():
                                    with gr.Tab("Image"):
                                        img = gr.Image(label=False, type="pil")
                                    with gr.Tab("Audio"):
                                        audio = gr.Audio(label=False, interactive=False)
                                    with gr.Tab("Video"):
                                        video = gr.Video(label=False, interactive=False)
                        with gr.Row():
                            submit = gr.Button("Send")
                with gr.Row():
                    # model specific configuration ui
                    cfgs = sym_tbl().proto.ui.builder()

        dp_chats.select(
            # load history
            fn=lambda x: sym_tbl().proto.history.load(Path(sym_tbl().cfg["history_dir"]) / x),
            inputs=[dp_chats]
        ).then(
            # update chatbot
            fn=lambda: sym_tbl().history.binding, outputs=[chatbot]
        ).then(
            # update instruction
            fn=lambda: "" if len(sym_tbl().history.storage) == 0 else sym_tbl().history.storage[-1]["query"]["instruction"],
            outputs=[instruction]
        )
        btn_refresh_chats.click(
            # refresh chatlist
            fn=refresh_chats, outputs=[dp_chats]
        ).then(
            # select current
            fn=lambda: sym_tbl().history.folder.name if sym_tbl().history.folder.exists() else "",
            outputs=[dp_chats]
        )

        btn_new_chat.click(
            # new chat
            fn=lambda : sym_tbl().proto.history.new(), outputs=[chatbot]
        ).then(
            # select none
            fn=lambda: "", outputs=[dp_chats]
        )

        submit.click(
            # update chatbot with input text, mkdir if new chat
            fn=send, inputs=[instruction, msg, dp_mm, img, audio, video], outputs=[chatbot]
        ).then(
            # select current
            fn=lambda: sym_tbl().history.folder.name, outputs=[dp_chats]
        ).then(
            # clear textbox and mm
            fn=lambda: (None, None, None, None), outputs=[msg, img, audio, video]
        ).then(
            # disable button
            fn=lambda: gr.update(interactive=False), outputs=[submit]
        ).then(
            # predict
            # update chatbot with output text
            fn=predict, inputs=cfgs, outputs=[chatbot]
        ).then(
            # flush chats to dir
            fn=lambda: sym_tbl().history.flush_last_rounds()
        ).then(
            # enable button
            fn=lambda: gr.update(interactive=True), outputs=[submit]
        )

    return ui