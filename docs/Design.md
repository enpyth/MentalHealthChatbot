## 技术设计

### 产品图

![image-20241110111856122](../imgs/ProductDiagram.png)

### 流程图

![image-20241110111856122](../imgs/FlowDiagram.svg)


### 功能需求

1. 用户管理

   注册、登录 

2. 问答功能

   TODO临时对话 [short-term memory](https://langchain-ai.github.io/langgraph/concepts/memory/#short-term-memory)  [thread-level persistence](https://langchain-ai.github.io/langgraph/how-tos/persistence/)

   TODO登录后多轮对话 [long-term memory](https://langchain-ai.github.io/langgraph/concepts/memory/#long-term-memory)  [cross-thread persistence](https://langchain-ai.github.io/langgraph/how-tos/cross-thread-persistence/)

   恢复历史会话 [conversation history](https://langchain-ai.github.io/langgraph/how-tos/memory/manage-conversation-history/)

   会话持久化 [PG](https://langchain-ai.github.io/langgraph/how-tos/persistence_postgres/)  [MongoDB](https://langchain-ai.github.io/langgraph/how-tos/persistence_mongodb/) [Redis](https://langchain-ai.github.io/langgraph/how-tos/persistence_redis/)

   TODO外部工具 [tool calling](https://langchain-ai.github.io/langgraph/how-tos/#tool-calling)  [structured output](https://langchain-ai.github.io/langgraph/how-tos/react-agent-structured-output/)

   收集用户数据的对话

   TODO生成问题询问病人，判断继续心理咨询师开导、心理医生进一步诊断和治疗。心理咨询师，继续聊；心理医生时，反馈一句话推荐找心理医生（借鉴西湖大学）；

3. 反馈机制

   每隔n轮询问用户是否产生健康报告 [Breakpoints + Approval](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/breakpoints/)

   健康预警 [Breakpoints + Input](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/wait-user-input/)

4. 安全检测

   敏感词过滤、内容审查 Filter Agent

5. 系统设置




### 非功能需求

1. 性能

   [How to run a graph asynchronously](https://langchain-ai.github.io/langgraph/how-tos/async/#how-to-run-a-graph-asynchronously)

2. 安全

3. 维护性

   日志管理 [streaming](https://langchain-ai.github.io/langgraph/how-tos/#streaming)

   架构展示 [visualize graph](https://langchain-ai.github.io/langgraph/how-tos/visualization/)
   
   复现调试 [node-retry](https://langchain-ai.github.io/langgraph/how-tos/node-retries/)

4. 扩展性

   前后端部署分离，前端streamlit，后端[FastAPI](https://fastapi.tiangolo.com/) + AgentServer



### 部署

[deploy hosted](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/)