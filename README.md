# chatbot-webui

Now support:

* [llama](https://huggingface.co/decapoda-research/llama-7b-hf) with [lora](https://huggingface.co/tloen/alpaca-lora-7b)
* [chatglm](https://huggingface.co/THUDM/chatglm-6b)
* [BELLE-7B-2M](https://huggingface.co/BelleGroup/BELLE-7B-2M)
* [blip2chatglm](https://github.com/XiPotatonium/LAVIS). Currently only training code is provided, we will release pretrained model soon.
* ChatGPT. You should first acquire your openai API and set your api key at `cfgs/chatgpt.json`


## Usage

1. launch webui

```bash
python launch.py cfgs/chatglm-6b.json
```

You should first download the huggingface model and then save the model in the location set in the config.

![](doc/img/chat-overview.png)

2. MultiModal chats

blip2chatglm supports multimodal chats. You can use the following command to launch the webui.
The model imperfect at this stage and we will continue improving.

```bash
python launch.py cfgs/blip2zh-chatglm-6b.json
```

![](doc/img/mm-chat-overview.png)
