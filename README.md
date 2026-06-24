# Learn LangChain v1.0

从官方文档中学习 LangChain v1.0 的笔记与示例代码。

## 关于这个仓库

2026 年 1 月，LangChain 发布了 v1.0，架构和 API 相比 v0.x 有较大变化。当时中文社区的资料还不全，所以我在读官方文档时顺手记了这份笔记，按模块组织了示例代码和关键概念，方便快速上手。

## 设计思路

读官方文档时的一个感受是：LangChain v1.0 封装了非常多功能，但并非每个都对初学者友好。

以 **Memory（记忆管理）** 为例，官方提供了大量的内置封装：`ConversationBufferMemory`、`ConversationSummaryMemory`、`VectorStoreRetrieverMemory`…… 命名繁多、职责重叠，初学者很容易陷入"该用哪个"的纠结里，反而忽略了核心原理——所谓记忆，本质上就是 **把对话历史传给 LLM**，以及 **当上下文太长时做总结或裁剪**。

这个仓库的原则是：**了解原理为主，封装为辅。**

- 每个模块优先讲清楚"这件事本质上是什么"，再展示 LangChain 提供哪些方式来做。
- 不会把官方所有 API 都罗列一遍——文档本身就是做这件事的。这里只记那些真正有用、能让你理解系统怎么运转的内容。
- 示例代码贴近实际使用场景，同时也标注了哪些是拓展功能、哪些是核心必学。

## 目录结构

```
├── 快速入门.ipynb         # 一个完整的端到端示例：提示词 → 工具 → LLM → 输出
├── 测试代码.py             # 零散的测试与验证代码
│
├── model/                 # 模型调用
│   ├── 基础用法.ipynb     # invoke / stream / batch 三种调用方式
│   └── 进阶用法.ipynb     # 高阶用法概览（官方文档后续展开）
│
├── messages/              # 消息系统
│   └── 消息的基本用法.ipynb  # HumanMessage / AIMessage / ToolMessage 以及多模态输入
│
├── tools/                 # 工具
│   ├── 创建工具.ipynb      # @tool 装饰器、自定义名称和描述
│   └── 访问上下文.ipynb     # ToolRuntime（State / Context / Store / Stream Writer）
│
├── agent/                 # Agent（代理）
│   ├── 工具的基本使用.ipynb    # create_agent、工具注册、错误处理
│   └── 静态模型和动态模型.ipynb # 固定 LLM vs 条件选择 LLM
│
├── middleware/             # 中间件（v1.0 新核心概念）
│   ├── 内置中间件.ipynb       # SummarizationMiddleware、HumanInTheLoop、Anthropic 缓存
│   ├── 自定义中间件.ipynb      # 装饰器式（@before_model / @wrap_model_call / @dynamic_prompt）和基于类的中间件
│   └── 其余解析和用法.ipynb    # 自定义状态、执行顺序、jump_to 跳转
│
├── short-term momery/     # 短期记忆
│   └── 用法.ipynb          # InMemorySaver / PostgresSaver / 自定义 AgentState
│
├── streaming/             # 流式输出
│   └── 常见用法.ipynb      # stream_mode="updates" / "messages" / "custom" / 混合模式
│
└── structuredOutput/      # 结构化输出
    ├── 结果化输出.ipynb    # ProviderStrategy（原生） vs ToolStrategy（通用）
    └── 错误处理.ipynb      # 解析失败的处理方式
```

## 模块速览

### 模型调用（model/）

LangChain v1.0 的模型调用统一为三种方式：

- **invoke** — 等待完整响应后一次性返回
- **stream** — 逐块返回 `AIMessageChunk`，适合流式展示
- **batch** — 并行处理一组请求，提高吞吐、降低成本

### 消息系统（messages/）

消息类型与之前版本一致，核心是三种：

- `HumanMessage` / `AIMessage` / `ToolMessage` — 最常用的三元组
- 新增了 `content_blocks` 字段，支持原生多模态（文本 + 图片）输入
- `ToolMessage.tool_call_id` 必须与 `AIMessage.tool_calls` 中的 ID 对应

### 工具（tools/）

- 使用 `@tool` 装饰器即可将一个函数注册为工具，文档字符串自动作为工具描述
- 可自定义工具名称和描述
- 通过 `ToolRuntime` 参数可以访问代理的 State、Context、Store、Stream Writer、Config 等运行时信息
- 使用 `Command` 可以修改代理状态或控制执行流程（例如清除对话历史）

### Agent（agent/）

- `create_agent()` 是创建代理的统一入口
- 可以传入 tools、middleware、response_format 等参数
- 支持静态模型（全程用同一个 LLM）和动态模型（通过中间件按条件选择不同 LLM）

### 中间件（middleware/）

**这是 v1.0 最核心的新概念**，替代了 v0.x 中很多零散的 chain / callback / memory 机制。

中间件有两种写法：

1. **装饰器式** — 快速简洁，适合单一功能
   - `@before_model` / `@after_model` / `@before_agent` / `@after_agent` — 节点式钩子
   - `@wrap_model_call` / `@wrap_tool_call` — 包裹式钩子（可实现重试、短路等）
   - `@dynamic_prompt` — 动态系统提示的快捷方式

2. **基于类** — 功能更强大，适合多个钩子组合
   - 继承 `AgentMiddleware`，实现 `before_model` / `after_model` / `wrap_model_call` 等方法

内置中间件包括：
- `SummarizationMiddleware` — 接近 Token 限制时自动总结历史，替代了 v0.x 的各类 Memory 封装
- `HumanInTheLoopMiddleware` — 工具执行前请求人工审批，提升安全性
- Anthropic Prompt Caching — 使用 Anthropic 模型时降低重复计算的成本

### 短期记忆（short-term momery/）

v1.0 的记忆机制比之前简洁很多：

- **本质是 Checkpoint**：通过 `checkpointer` 参数保存线程状态，同一个 `thread_id` 自动恢复上下文
- 开发环境用 `InMemorySaver`，生产环境用 `PostgresSaver` 等持久化方案
- 可通过自定义 `AgentState` 扩展记忆字段（如 `user_id`、`preferences`）
- 消息修剪通过 `@before_model` 中间件实现

### 流式输出（streaming/）

- `stream_mode="updates"` — 每一步代理状态变更都通知你（工具调用、模型响应等）
- `stream_mode="messages"` — LLM 逐 Token 输出，适合打字机效果
- `stream_mode="custom"` — 工具内部通过 `get_stream_writer()` 自定义输出
- 多种模式可以组合传入

### 结构化输出（structuredOutput/）

两种策略：

- **ProviderStrategy** — 使用模型厂商原生支持的结构化输出（OpenAI、Grok、Gemini 等），最可靠
- **ToolStrategy** — 通过工具调用模拟结构化输出，适用于所有支持工具调用的模型

支持自定义 `tool_message_content`，用于控制工具调用在对话历史中的记录方式——既可以保护隐私数据，也能减少 Token 消耗。

## 备注

- 本仓库的示例代码主要来自 [LangChain 官方文档](https://python.langchain.com/docs/introduction/)，部分代码做了简化或注释补充
- 撰写时间：**2026 年 1 月**
- LangChain 版本：**v1.0**（刚发布时的版本，后续 API 可能有所变化）
- 运行示例需要配置 `OPENAI_API_KEY` 和 `OPENAI_BASE_URL` 环境变量，参考 `.env` 文件
