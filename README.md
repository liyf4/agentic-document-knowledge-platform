# Agentic RAG Chatbot

一个面向文档与表格知识工作的本地 Agentic RAG 应用。项目采用 FastAPI + React 架构，支持会话管理、文件上传与解析、混合检索、引用溯源、Skill 工作流、执行轨迹，以及可选的隔离工作区。

## 主要功能

- 上传并解析 PDF、Word、Markdown、文本、CSV 和 Excel 文件
- 结合向量检索、BM25、RRF 与可选重排器进行混合检索
- 在回答中保留来源引用，并记录工具调用和检索轨迹
- 管理会话、知识文件、异步索引任务和生成产物
- 通过 Skill 执行中文摘要、深度文档分析和表格分析等工作流
- 可选接入 OpenSandbox，为每个会话提供隔离、持久的工作区
- React + Ant Design 前端，FastAPI 后端，支持流式响应

## 技术栈

- 后端：Python、FastAPI、LangChain、ChromaDB、SQLite
- 检索：Hugging Face Embeddings、BM25、RRF、CrossEncoder（可选）
- 前端：React、TypeScript、Vite、Ant Design
- 数据处理：pandas、PyArrow、DuckDB、PyPDF、python-docx、openpyxl

## 环境要求

- Python 3.10 或更高版本
- Node.js 18 或更高版本
- npm
- 可用的大模型 API 密钥

首次安装 `sentence-transformers` 等依赖时可能需要下载模型，请确保网络可用。

## 快速开始

### 1. 获取代码并创建 Python 环境

```powershell
git clone https://github.com/liyf4/chatbot.git
cd chatbot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux/macOS 激活虚拟环境：

```bash
source .venv/bin/activate
```

### 2. 配置环境变量

复制示例文件：

```powershell
Copy-Item .env.example .env
```

然后在 `.env` 中填写所需配置：

```dotenv
ZAI_API_KEY=your_api_key
GOOGLE_API_KEY=
GOOGLE_CSE_ID=
SERPAPI_API_KEY=
PROXY_PORT=
OPEN_SANDBOX_API_KEY=
```

至少配置当前模型提供商所需的 API Key。`.env` 已被 Git 忽略，请勿把真实密钥写入 `.env.example`、`config.yaml` 或源码。

### 3. 安装前端依赖

```powershell
cd frontend
npm install
cd ..
```

### 4. 启动项目

Windows 可以直接运行：

```powershell
.\start.cmd
```

也可以跨平台启动：

```powershell
python scripts/dev.py
```

默认地址：

- 前端：http://127.0.0.1:5173
- 后端健康检查：http://127.0.0.1:8000/api/health
- FastAPI 接口文档：http://127.0.0.1:8000/docs

如需分别启动：

```powershell
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

```powershell
cd frontend
npm run dev
```

## 配置说明

通用的非敏感配置位于 `config.yaml`，包括模型、Embedding、检索参数、数据目录、日志和 Sandbox 设置。敏感值应只通过 `.env` 注入。

隔离工作区默认关闭。如需启用，请准备 OpenSandbox 服务，安装项目依赖，并在 `config.yaml` 中设置 `sandbox.enabled: true`。认证密钥放在 `.env` 的 `OPEN_SANDBOX_API_KEY` 中。

## 项目结构

```text
backend/           FastAPI 接口与流式聊天服务
core/              Agent、检索、路由、文档处理与执行轨迹
db/                会话、文件、Skill 和 Sandbox 状态管理
frontend/          React + TypeScript 前端
prompts/           Prompt 模板
sandbox_runtime/   隔离工作区运行时适配
skills/            可执行 Skill 定义
tools/             Agent 工具
utils/             网络与兼容性工具
config.yaml        非敏感运行配置
```

## 数据与隐私

本项目会在本地生成会话、上传文件、向量索引、数据库、日志和执行结果。这些内容以及 `.env`、测试目录、评估结果、依赖目录和普通项目文档均已加入 `.gitignore`。

提交前建议始终检查：

```powershell
git status --short
git diff --cached
```

如果密钥曾经进入 Git 提交，仅删除本地文件或加入 `.gitignore` 并不能清除历史；应立即轮换密钥，并使用 `git filter-repo` 等工具清理仓库历史。

## 前端构建

```powershell
cd frontend
npm run build
```

构建产物位于 `frontend/dist/`，不会提交到仓库。
