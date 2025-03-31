# 友情链接

对 ChatGLM 进行加速或者重新实现的开源项目：
* [lyraChatGLM](https://huggingface.co/TMElyralab/lyraChatGLM): 对 ChatGLM-6B 进行推理加速，最高可以实现 9000+ tokens/s 的推理速度
* [SwissArmyTransformer](https://github.com/THUDM/SwissArmyTransformer): 一个Transformer统一编程框架，ChatGLM-6B已经在SAT中进行实现并可以进行P-tuning微调。
* [ChatGLM-MNN](https://github.com/wangzhaode/ChatGLM-MNN): 一个基于 MNN 的 ChatGLM-6B C++ 推理实现，支持根据显存大小自动分配计算任务给 GPU 和 CPU
* [JittorLLMs](https://github.com/Jittor/JittorLLMs)：最低3G显存或者没有显卡都可运行 ChatGLM-6B FP16， 支持Linux、windows、Mac部署
* [InferLLM](https://github.com/MegEngine/InferLLM)：轻量级 C++ 推理，可以实现本地 x86，Arm 处理器上实时聊天，手机上也同样可以实时运行，运行内存只需要 4G



基于或使用了 ChatGLM-6B 的开源项目：
* [chatgpt_academic](https://github.com/binary-husky/chatgpt_academic): 支持ChatGLM-6B的学术写作与编程工具箱，具有模块化和多线程调用LLM的特点，可并行调用多种LLM。
* [闻达](https://github.com/l15y/wenda)：大型语言模型调用平台，基于 ChatGLM-6B 实现了类 ChatPDF 功能
* [glm-bot](https://github.com/initialencounter/glm-bot)：将ChatGLM接入Koishi可在各大聊天平台上调用ChatGLM
* [Chinese-LangChain](https://github.com/yanqiangmiffy/Chinese-LangChain):中文langchain项目，基于ChatGLM-6b+langchain实现本地化知识库检索与智能答案生成，增加web search功能、知识库选择功能和支持知识增量更新
* [bibliothecarius](https://github.com/coderabbit214/bibliothecarius)：快速构建服务以集成您的本地数据和AI模型，支持ChatGLM等本地化模型接入。
* [langchain-ChatGLM](https://github.com/imClumsyPanda/langchain-ChatGLM)：基于 langchain 的 ChatGLM 应用，实现基于可扩展知识库的问答
* [Casibase](https://github.com/casibase/casibase): 一个开源的AI知识库和对话系统，结合了RAG技术和SSO，并支持如chatGLM的各种主流AI模型。旨在为企业和开发者提供一个功能强大、灵活易用的知识管理和智能对话平台。
* [ChatGLM-web](https://github.com/NCZkevin/chatglm-web)：基于FastAPI和Vue3搭建的ChatGLM演示网站(支持chatglm流式输出、前端调整模型参数、上下文选择、保存图片、知识库问答等功能)
* [Chuanhu Chat](https://github.com/GaiZhenbiao/ChuanhuChatGPT): 为各个大语言模型和在线模型API提供美观易用、功能丰富、快速部署的用户界面，支持ChatGLM-6B。
* [ChatGLM-6B-Engineering](https://github.com/LemonQu-GIT/ChatGLM-6B-Engineering)：基于 ChatGLM-6B 后期调教，网络爬虫及 [Stable Diffusion](https://github.com/AUTOMATIC1111/stable-diffusion-webui) 实现的网络搜索及图片生成
* [ChatGLM-OpenAI-API](https://github.com/ninehills/chatglm-openai-api): 将 ChatGLM-6B 封装为 OpenAI API 风格，并通过 ngrok/cloudflare 对外提供服务，从而将 ChatGLM 快速集成到 OpenAI 的各种生态中。
* [ChatSQL](https://github.com/cubenlp/ChatSQL): 基于ChatGLM+SBERT实现NL2SQL本地化，并直接连接数据库查询数据返回结果，使得生成的SQL语句更具有实用性。

对 ChatGLM-6B 进行微调的开源项目：
* [InstructGLM](https://github.com/yanqiangmiffy/InstructGLM)：基于ChatGLM-6B进行指令学习，汇总开源中英文指令数据，基于Lora进行指令数据微调，开放了Alpaca、Belle微调后的Lora权重，修复web_demo重复问题
* [ChatGLM-Efficient-Tuning](https://github.com/hiyouga/ChatGLM-Efficient-Tuning)：实现了ChatGLM-6B模型的监督微调和完整RLHF训练，汇总10余种指令数据集和3种微调方案，实现了4/8比特量化和模型权重融合，提供微调模型快速部署方法。
* [ChatGLM-Finetuning](https://github.com/liucongg/ChatGLM-Finetuning)：基于ChatGLM-6B模型，进行下游具体任务微调，涉及Freeze、Lora、P-tuning等，并进行实验效果对比。
* [ChatGLM-Tuning](https://github.com/mymusise/ChatGLM-Tuning): 基于 LoRA 对 ChatGLM-6B 进行微调。类似的项目还包括 [Humanable ChatGLM/GPT Fine-tuning | ChatGLM 微调](https://github.com/hscspring/hcgf)


针对 ChatGLM-6B 的教程/文档：
* [Windows部署文档](https://github.com/ZhangErling/ChatGLM-6B/blob/main/deployment_windows.md)
* [搭建深度学习docker容器以运行 ChatGLM-6B - Luck_zy](https://www.luckzym.com/tags/ChatGLM-6B/)

如果你有其他好的项目/教程的话，欢迎参照上述格式添加到 README 中并提出 [Pull Request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork)。

