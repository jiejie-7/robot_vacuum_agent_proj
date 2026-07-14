可以，我们用一个最小例子把三件事串起来：

- `schema`：工具入参格式
- `message`：对话消息
- `checkpointer`：短期记忆保存

```python
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver


# 1. 手写 schema：定义工具需要什么参数
class WeatherInput(BaseModel):
    city: str = Field(..., description="要查询天气的城市名")


# 2. 普通 Python 函数：真正的工具逻辑
def get_weather(city: str) -> str:
    return f"{city} 今天晴，适合扫地机器人工作。"


# 3. 把函数包装成 LangChain 工具
weather_tool = StructuredTool.from_function(
    func=get_weather,
    name="get_weather",
    description="查询指定城市的天气情况",
    args_schema=WeatherInput,
)


# 4. checkpointer：保存短期记忆
memory = InMemorySaver()

checkpointer 可以理解成：用来存储 LangGraph / LangChain Agent 的 state 的组件。

但这里容易绕，我们一点点拆。

1. state 是什么
在 Agent 运行时，框架内部会维护一个状态，叫 state。

最核心的内容就是：

{
    "messages": [
        用户消息,
        AI 消息,
        工具调用消息,
        工具返回消息,
        AI 最终回答
    ]
}
所以你可以先简单理解：

state ≈ 当前会话到目前为止的全部消息 + 一些运行时状态

比如一次 Agent 执行中可能是这样：

state = {
    "messages": [
        {"role": "user", "content": "我在合肥，查一下天气"},
        {"role": "assistant", "content": "", "tool_calls": [...]},
        {"role": "tool", "content": "合肥今天晴"},
        {"role": "assistant", "content": "合肥今天晴，适合拖地"}
    ]
}
2. checkpointer 是什么
checkpointer 就是把这个 state 存起来的东西。

没有 checkpointer：

第一次 invoke 结束后，state 就没了
第二次 invoke 时，Agent 不知道上一轮聊了什么
有 checkpointer：

第一次 invoke 结束后，state 会按 thread_id 保存
第二次 invoke 使用同一个 thread_id 时，框架会把之前的 state 取出来
然后把新消息追加进去
所以它的作用是：

保存 state
恢复 state
让多轮对话能跨 invoke 延续
3. thread_id 是什么
thread_id 就是这段会话的 ID。

比如：

config={
    "configurable": {
        "thread_id": "user-001"
    }
}
意思是：

把这次对话状态保存到 user-001 这个会话下面
下一次你还传：

"thread_id": "user-001"
框架就知道：

哦，这还是同一个用户/同一段会话，把之前的 state 拿出来继续用
如果你换成：

"thread_id": "user-002"
那就是另一段新会话。

4. 一个最简单的比喻
可以这样记：

messages = 聊天记录
state = 聊天记录 + Agent 运行状态
checkpointer = 保存 state 的存档器
thread_id = 存档名称
所以：

checkpointer 不是记忆本身
它是保存记忆的地方
真正的记忆内容在 state["messages"] 里。

5. 和你当前项目的区别
你现在项目是手动传历史：

agent.execute(query, history=history)
也就是：

前端 Streamlit 自己保存 history
每次调用 Agent 时手动传进去
如果用 checkpointer，就变成：

agent.invoke(
    {"messages": [{"role": "user", "content": query}]},
    config={"configurable": {"thread_id": user_id}}
)
这时候你不用每次手动拼完整 history，框架会根据 thread_id 自动恢复之前的 state。

一句话总结
checkpointer 是 LangGraph 用来保存和恢复 state 的组件；在 Agent 场景里，state 里最重要的就是 messages，所以它经常被用来实现短期记忆和多轮对话。


# 5. 创建 Agent
agent = create_agent(
    model="qwen:qwen-plus",  # 这里换成你项目实际模型
    tools=[weather_tool],
    checkpointer=memory,
)


# 6. 第一次对话
result1 = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "我在合肥，帮我查一下天气"}
        ]
    },
    config={
        "configurable": {
            "thread_id": "user_001"
        }
    },
)

print(result1["messages"][-1].content)


# 7. 第二次对话：不再说城市，但 Agent 能通过 thread_id 找到上一轮历史
result2 = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "那适合让扫地机器人拖地吗？"}
        ]
    },
    config={
        "configurable": {
            "thread_id": "user_001"
        }
    },
)

print(result2["messages"][-1].content)
```

这里最关键的是这几个点。

`WeatherInput` 就是 schema：

```python
class WeatherInput(BaseModel):
    city: str = Field(..., description="要查询天气的城市名")
```

它告诉模型：  
这个工具必须传一个 `city` 参数，而且是字符串。

`StructuredTool.from_function(...)` 是手写工具注册方式：

```python
weather_tool = StructuredTool.from_function(
    func=get_weather,
    name="get_weather",
    description="查询指定城市的天气情况",
    args_schema=WeatherInput,
)
```

这等价于你用 `@tool` 装饰器，但更显式。

`messages` 是本轮传给 Agent 的消息：

```python
{"role": "user", "content": "我在合肥，帮我查一下天气"}
```

它不是记忆本身，只是“这一轮输入”。

`checkpointer` 才是短期记忆的关键：

```python
memory = InMemorySaver()
```

然后通过同一个 `thread_id` 把多轮对话串起来：

```python
"thread_id": "user_001"
```

所以第二轮你只问：

```python
"那适合让扫地机器人拖地吗？"
```

Agent 仍然能知道上一轮你说过“我在合肥”。

简单总结：

- `schema`：工具参数说明书
- `tool`：模型可以调用的函数能力
- `messages`：当前轮输入输出消息
- `checkpointer`：把历史 messages 存起来
- `thread_id`：告诉系统这是同一个会话

你项目里现在是手动传 `history` 实现多轮；如果用 `checkpointer`，就可以让 LangGraph/LangChain runtime 帮你按 `thread_id` 管理短期记忆。