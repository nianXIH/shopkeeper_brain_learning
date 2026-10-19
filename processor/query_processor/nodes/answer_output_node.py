from typing import List, Dict, Any, Tuple
from langchain_openai import ChatOpenAI
from sympy import content

from knowledge.processor.query_processor.base import BaseNode, T
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.mongo_history_util import save_chat_message
from knowledge.utils.task_util import set_task_result
from knowledge.utils.sse_util import push_sse_event, SSEEvent
from knowledge.prompts.query_prompt import ANSWER_PROMPT


class AnswerOutPutNode(BaseNode):

    name = "answer_output_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        核心逻辑：
        1. 从state中获取answer
        1.1 如果获取到了answer:--->没有进行三路检索[不用在生成答案，直接返回]-----【答案如何推送给前端：1.流式（直接将已经生成的内容都给前端）2.非流式（直接将已经生成的内容都给前端）】
        1.2 如果没有获取answer---->进行了三路检索[需要调用LLM生成答案，在返回]----【答案如何推送给前端：1.流式推送（SSE）  2.非流式的(传统)】明显变化
        Args:
            state:

        Returns:

        """
        # 1. 判断是否是流式
        is_stream = state.get("is_stream")

        # 2. 获取任务id
        task_id = state.get("task_id")

        # 3. 判断state中是否有answer
        if state.get("answer"):
            # 将答案推送出去
            self._push_exist_answer(task_id,is_stream,state)
            is_stream = False
        else:
            # 3.2 组装提示词
            prompt = self._build_prompt(state)
            state["prompt"] = prompt

            # 3.3 调用LLM(流式和非流式)
            self._generate_answer(prompt,task_id,state)
            is_streamed = is_stream

        # 4. 保留历史会话
        # 4.1 只要你问了问题 有答案(LLM生成 你生成)
        self.save_history(state)

        # 5. 关闭sse通道(修改事件类型为FINAL)
        if is_stream:
            # 5.1 已经流过（LLM生成的答案）
            if is_streamed:
                push_sse_event(task_id=task_id, event=SSEEvent.FINAL, data={})
            # 5.2 没有流过(自己生成的答案)
            else:
                push_sse_event(task_id=task_id, event=SSEEvent.FINAL, data={"answer": state.get('answer')})

        # 6. 返回
        return state


    def _push_exist_answer(self, task_id: str, is_stream: bool, state: QueryGraphState):
        """


        Args:
            self:
            task_id:
            is_stream:
            state:

        Returns:

        """
        # 1. 判断是非流式【普通的任务队列：任务结果队列_tasks_result】
        if not is_stream:
            set_task_result(task_id=task_id, key="answer", value=state.get('answer'))  # 在该处放


    def _build_prompt(self, state: QueryGraphState) -> str:
        """
        Args:
            state:

        Returns:

        """

        max_context_chars = self.config.max_context_chars

        # 1. 获取用户的原始问题
        user_query = state.get("rewritten_query")

        # 2. 获取商品名列表
        item_names = state.get('item_names') or []

        # 3. 构建检索的上下文
        retrieval_context = state.get("reranked_docs") or []
        formatted_context,usage_chars = self._format_retrieval_context(retrieval_context,max_context_chars)

        # 4. 构建历史上下文
        chat_history_context = state.get('history') or []  # 从内存中获取历史对话（获取不到）
        formatted_history = self._format_chat_history(chat_history_context, usage_chars)

        # 5. 格式化提示词模版
        return ANSWER_PROMPT.format(
            context=formatted_context or "暂无检索到上下文",
            history=formatted_history or "暂无历史上下文",
            item_names=",".join(item_names),
            question=user_query
        )


    def _format_retrieval_context(self, retrieval_context: List[Dict[str, Any]], max_context_chars: int) -> Tuple[str, int]:
        """
        格式化检索到的上下文
        【自己拼接一些元数据：供LLM学习，回答答案更准确】
        Args:
            retrieval_context: 检索到的上下文
            max_context_chars: 最大上下文的长度

        Returns:
            格式后的上下文

        """

        # 1. 遍历
        formatted_lines = []
        usage = 0
        for index, context in enumerate(retrieval_context,1):

            # 1.1 获取内容
            content = context.get("content","")

            # 1.2 判断内容
            if not content:
                continue

            # 1.3 获取元数据
            metadata_content = [f"[文档:{index}]"]

            # 1.4 定义其它元数据
            for meta_field, template in [("chunk_id", "[chunk_id={}]"),
                                         ("title", "[title={}]"),
                                         ("source", "[source={}]"),
                                         ("url", "[url={}]")]:
                # a. 获取各个元数据字段的值
                filed_value = str(context.get(meta_field, "")).strip()

                # b.格式化模版中的占位符
                if filed_value:
                    metadata_content.append(template.format(filed_value))

            # 1.5 获取得分
            doc_score = context.get('score')

            # 1.6 判断得分
            if doc_score is not None:
                metadata_content.append(f"[score={float(doc_score):.6f}]")

            # 1.7 构建完整行的数据（元数据+获取的内容）
            formatted_line = " ".join(metadata_content) + "\n" + content

            # 1.8 计算行与行之间的字符数(\n\n)
            sep_chars = 2 if formatted_lines else 0

            total_length = sep_chars + len(formatted_line)

            # 1.9 计算总长度
            if usage + total_length > max_context_chars:
                break
            else:
                formatted_lines.append(formatted_line)
                usage += total_length

        return "\n\n".join(formatted_lines), max_context_chars - usage


    def _format_chat_history(self, chat_history_context: List[Dict[str, Any]], usage_chars: int) -> str:
        """
        格式化历史上下文
        Args:
            chat_history_context: 历史上下文
            usage_chars: 可用字符串长度

        Returns:

        """

        formatted_lines = []
        used_chars = 0
        # 1. 遍历格式化后的文档
        role_map = {"user": "用户", "assistant": "助手"}
        for msg in chat_history_context:

            # 1.1 获取消息角色
            role = msg.get('role', '')

            # 1.2 获取消息内容
            text = msg.get('text', '')

            # 1.3 获取格式化后的行
            if not text or role not in role_map:
                continue

            formatted_line = f"{role_map[role]}: {text}"

            # 1.4 计算分割符长度
            seperator_usage = 1 if formatted_lines else 0

            # 1.5 计算总长度
            total_usage = seperator_usage + len(formatted_line)

            if used_chars + total_usage > usage_chars:
                break

            formatted_lines.append(formatted_line)
            used_chars += total_usage

        return "\n".join(formatted_lines)


    def _generate_answer(self, prompt: str, task_id: str, state: QueryGraphState):
        """
        调用LLM  生成答案 更新到state
        Args:
            prompt:  提示词
            task_id: 任务id
            state:

        Returns:

        """

        # 1. 获取LLM客户端
        try:
            llm_client = AIClients.get_llm_client(response_format=False)
        except ConnectionError as e:
            self.logger.error(f"获取LLM客户端失败 原因:{str(e)}")
            state['answer'] = "LLM暂无法回答"
            return

        # 2. 判断流式开关
        if state.get('is_stream'):
            # 2.1 流式调用
            # 获取llm的结果(stream)
            # 写入到sse队列
            state['answer'] = self._stream_llm(task_id, prompt, llm_client)

        else:
            # 2.2 非流式调用
            state['answer'] = self._invoke_llm(prompt, llm_client)
            # 写入到任务结果队列中(非流式调用)
            set_task_result(task_id=task_id, key="answer", value=state['answer'])


    def _invoke_llm(self, prompt: str, llm_client: ChatOpenAI) -> str:
        """
        非流式LLM调用
        Args:
            prompt:
            llm_client:

        Returns:

        """
        try:
            # 1. 同步调用
            llm_res = llm_client.invoke(prompt)

            # 2. 获取内容
            llm_content = getattr(llm_res, 'content', "") or ""

            # 3. 判断
            if not llm_content:
                return "LLM暂无法回答"

            return llm_content
        except Exception as e:
            self.logger.error(f"LLM调用失败 原因:{str(e)}")
            return "LLM暂无法回答"


    def _stream_llm(self, task_id: str, prompt: str, llm_client: ChatOpenAI) -> str:
        """
        流式调用
        Args:
            prompt:
            llm_client:

        Returns:

        """
        accelerate_delta = ""  # 获取全量数据
        try:
            # 1. 流式调用
            for chunk in llm_client.stream(prompt):

                # 1.1 获取片段的内容
                delta_text = getattr(chunk, 'content', "") or ""
                if delta_text:  # 增量放到sse队列
                    push_sse_event(task_id=task_id,
                                   event=SSEEvent.DELTA,
                                   data={"delta": delta_text}
                                   )
                    accelerate_delta += delta_text
        except Exception as e:
            self.logger.error(f"LLM流式调用失败 原因:{str(e)}")
            return "LLM暂无法回答"

        return accelerate_delta


    def save_history(self, state: QueryGraphState):
        """
        保存历史对话（Q--->A）
        存储位置：mongodb对应kb001库下的chat_message表中
        Args:
            state:

        Returns:

        """
        # 1. 获取session_id
        session_id = state.get('session_id')

        # 2. 获取用户的查询问题
        user_query = state.get('original_query')

        # 3. 获取改写后的查询问题
        rewritten_query = state.get('rewritten_query')

        # 4. 获取商品名列表
        item_names = state.get('item_names') or []

        # 5.1 保存用户角色的消息
        try:
            save_chat_message(
                session_id=session_id,
                role="user",
                text=user_query,
                rewritten_query=rewritten_query,
                item_names=item_names
            )

            # 5.2 保存AI角色的消息
            save_chat_message(session_id=session_id,
                              role="assistant",
                              text=state.get('answer'),
                              rewritten_query=rewritten_query,
                              item_names=item_names
                              )
        except Exception as e:
            self.logger.error(f"保存历史对话到MongDB中失败 原因:{str(e)}")
