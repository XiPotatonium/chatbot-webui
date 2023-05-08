from pathlib import Path
from typing import List, Optional
import gradio as gr
from loguru import logger
from .sym import sym_tbl
from .state import State, ROLE_SYSTEM, ROLE_BOT, ROLE_USER
from .history import load_history, save_history, append_last_message_binding, sync_last_history


_css = """
#emoji-btn {
    max-width: 2.5em;
    min-width: 2.5em !important;
    height: 2.5em;
    margin: 0.15em 0;
}
"""

# #chatbot {
#     height: 100%;
#     overflow: auto !important;
# }


def send(state: State, binding, msg: str, instruction: str, mm_ty: str, img, audio, video):
    msg = msg if msg is not None else ""
    instruction = instruction if instruction is not None else ""
    if len(instruction) != 0:
        state.append_message_history(role=ROLE_SYSTEM, content=instruction)
        append_last_message_binding(state, binding)

    mm_path = None
    if (
        (mm_ty == "Image" and img is not None) or
        (mm_ty == "Audio" and audio is not None) or
        (mm_ty == "Video" and video is not None)
    ):
        from PIL import Image
        import uuid
        if isinstance(img, Image.Image):
            mm_path = state.folder / f"{uuid.uuid4()}.png"
            if mm_path.exists():
                raise FileExistsError(f"File {mm_path} already exists. WTF?")
            img.save(mm_path, "PNG")
            mime = "image/png"
        else:
            raise NotImplementedError()

        mm_path = mm_path.name         # may be serialized to json, convert to string format
    if len(msg) != 0 or mm_path is not None:
        state.append_message_history(role=ROLE_USER, content=msg)
        if mm_path is not None:
            state.history[-1]["media"] = [[mm_path, mime]]          # currently only 1 image in media
        append_last_message_binding(state, binding)

    return binding


def predict(*args, **kwargs):
    if sym_tbl().cfg.get("support_stream", False):
        for binding in sym_tbl().model.stream_generate(*args, **kwargs):
            yield binding
    else:
        binding = sym_tbl().model.generate(*args, **kwargs)
        yield binding


def lst_chats() -> List[str]:
    return [f.name for f in Path(sym_tbl().cfg["history_dir"]).iterdir() if f.is_dir()]


def refresh_chats(current: str):
    return gr.update(choices=lst_chats(), value=current)


def create_ui():
    with gr.Blocks(css=_css + sym_tbl().proto.ui.css, title="chatbot-webui") as ui:
        state = gr.State(value=State(folder=sym_tbl().tmp_dir))
        with gr.Row():
            with gr.Column(scale=7, elem_id="chatpanel"):
                chatbot = gr.Chatbot(elem_id="chatbot", show_label=False).style(height=700)

            with gr.Column(scale=3):
                with gr.Row():
                    with gr.Column(variant="panel"):
                        with gr.Row():
                            dp_chats = gr.Dropdown(
                                show_label=False, choices=lst_chats(), interactive=sym_tbl().cfg["history"],
                            ).style(container=False)
                            btn_refresh_chats = gr.Button("♻", elem_id="emoji-btn")
                        with gr.Row():
                            btn_new_chat = gr.Button("New")
                            btn_save_chat = gr.Button("Save", interactive=sym_tbl().cfg["history"])

                with gr.Row():
                    with gr.Column(variant="panel"):
                        with gr.Accordion("Instruction", open=False):
                            instruction = gr.Textbox(lines=2, placeholder="Your instruction here...", show_label=False).style(container=False)
                        with gr.Row():
                            with gr.Accordion(label="Media", open=False):
                                with gr.Row():
                                    radio_mm = gr.Radio(["Image", "Audio", "Video"], show_label=False).style(item_container=False)
                                with gr.Row():
                                    with gr.Tab("Image"):
                                        img = gr.Image(label=False, type="pil")
                                    with gr.Tab("Audio"):
                                        audio = gr.Audio(label=False, interactive=False)
                                    with gr.Tab("Video"):
                                        video = gr.Video(label=False, interactive=False)
                        with gr.Row():
                            msg = gr.Textbox(
                                label="Input",
                                placeholder="Type your text... (Press Enter for new line, Shift+Enter for sending)",
                                show_label=False, lines=3
                            ).style(container=False)
                with gr.Row():
                    # model specific configuration ui
                    cfgs = sym_tbl().proto.ui.builder()

        dp_chats.select(
            # load history
            # update chatbot
            fn=load_history,
            inputs=[state, dp_chats], outputs=[chatbot]
        )

        btn_refresh_chats.click(
            # refresh chatlist
            fn=refresh_chats, inputs=[dp_chats], outputs=[dp_chats]
        )

        btn_new_chat.click(
            # new chat
            # select none
            fn=lambda: (State(folder=sym_tbl().tmp_dir), [], ""), outputs=[state, chatbot, dp_chats]
        )

        btn_save_chat.click(
            # save history
            fn=lambda s: save_history(s), inputs=[state], outputs=[dp_chats],
        ).then(
            # refresh chatlist
            fn=refresh_chats, inputs=[dp_chats], outputs=[dp_chats]
        )

        msg.submit(
            # update chatbot with input text, mkdir if new chat
            fn=send, inputs=[state, chatbot, msg, instruction, radio_mm, img, audio, video], outputs=[chatbot],
            queue=False,
        ).then(
            # clear textbox and mm
            fn=lambda: (None, None, None), outputs=[instruction, msg, radio_mm]
        ).then(
            # predict
            # update chatbot with output text
            fn=predict, inputs=[state, chatbot] + cfgs, outputs=[chatbot]
        ).then(
            # sync last chat to history
            fn=lambda s, f: sync_last_history(s, f) if f is not None and len(f) != 0 else None, inputs=[state, dp_chats]
        )

    return ui