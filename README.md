## arXiv 文献检索（Flask）

一个支持 AI 辅助检索的 Flask 前后端应用，集成 LangChain/OpenAI 接口与 arXiv API，前端可流式查看多阶段进度。

### 快速开始

- 准备环境并安装依赖：
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements.txt`

- 开发运行：
  - `python app.py`
  - 或使用 Flask：
    - `export FLASK_APP=app.py`
    - `flask run`

提示：默认 `FLASK_DEBUG=1` 开启调试。生产部署请设置 `FLASK_DEBUG=0` 并使用 gunicorn/uwsgi 等 WSGI 服务器。

### API

- `GET /search?q=...&sort=relevance|submitted_date|last_updated&max_results=1..100`
  - 基础检索接口，直接将查询词交给 arXiv API，返回字段：`title, authors[], summary, published, arxiv_id, link, pdf_url, primary_category`
- `POST /chat`
  - 请求体：`{message: str, sort?: str, max_results?: int}`
  - 行为：调用 LLM 生成英文检索关键词，自动回退到用户原始输入，返回 `reasoning/query/results`
- `GET /ai-search-stream?q=...&sort=...&max_results=...`
  - 使用 SSE（Server-Sent Events）流式返回进度。AI 会生成 3 个不同的英文检索式，分路查询 arXiv，再由 LLM 阅读摘要筛选，返回精选结果与 summary。

### AI 工作流

1. 前端将自然语言需求发送到 `/ai-search-stream`，实时展示进度条与事件。
2. 后端 LLM 生成 3 条不同角度的检索式，逐条查询 arXiv 并去重。
3. 若某轮检索未命中文献，会触发“反思”提示重新生成检索式，直到命中或达到设定次数。
4. LLM 阅读候选摘要，挑选最相关的若干篇并给出筛选理由+概要。
4. 前端显示 AI 概览、筛选理由、PDF 链接及摘要折叠按钮。

### 安全与健壮性

- 服务端校验排序、数量等参数，异常统一返回可读提示。
- SSE 流式响应中仅输出必要信息，遇到错误时会发送 `failure` 事件以便前端提示。
- arXiv 客户端配置轻量重试；前端 DOM 渲染使用 `textContent` 防止 XSS。

### 目录结构

- `app.py` 后端入口
- `templates/index.html` 前端页面
- `requirements.txt` 依赖
- `README.md` 文档
