"""
LangChain多智能体系统

本模块基于LangChain框架实现多智能体系统，使用自定义智能体协作解决问题，替代原CAMEL框架。
"""

import os
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import contextlib
import asyncio
from dotenv import load_dotenv
from typing import TypedDict

from openai import AsyncOpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain.callbacks.openai_info import OpenAICallbackHandler
from mas_arena.agents.base import AgentSystem, AgentSystemRegistry

# 加载环境变量
load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class AgentResponse(TypedDict):
    """Structured output for agent responses"""
    analysis: str  # Problem analysis
    solution: str  # Solution
    confidence: int  # Confidence level in the solution, range 1-5

@dataclass
class Agent:
    """Represents an LLM agent"""
    agent_id: str
    name: str
    model_name: str
    system_prompt: str
    chat_history: List[Dict[str, str]] = None
    
    def __post_init__(self):
        self.chat_history = []
        self.llm = ChatOpenAI(
            model=self.model_name,
            api_key=os.getenv("OPENAI_API_KEY", "sk-nhgJRZ0BQsKljo9ZOOcS4qlKejpptHpStgXjnpCLmmbb63pS"),
            base_url=os.getenv("OPENAI_API_BASE", "https://35.aigcbest.top/v1"),
            request_timeout=60,
            max_retries=2
        )

    async def generate_response(self, context: str) -> Dict[str, Any]:
        """Generate agent response"""
        messages = [
            SystemMessage(content=self.system_prompt),
            *[HumanMessage(content=msg["human"]) if msg.get("role") == "human" 
              else AIMessage(content=msg["ai"]) 
              for msg in self.chat_history],
            HumanMessage(content=context)
        ]
        
        try:
            callback_handler = OpenAICallbackHandler()
            config = {"callbacks": [callback_handler]}
            llm_with_schema = self.llm.with_structured_output(schema=AgentResponse, include_raw=True)
            response = await llm_with_schema.ainvoke(messages, config=config)
            
            structured_data = response["parsed"]
            raw_response = response["raw"]
            
            if isinstance(raw_response, AIMessage):
                raw_response.usage_metadata = {
                    "input_tokens": callback_handler.prompt_tokens,
                    "output_tokens": callback_handler.completion_tokens,
                    "total_tokens": callback_handler.total_tokens,
                    "input_token_details": {},
                    "output_token_details": {"reasoning": callback_handler.completion_tokens}
                }
            
            if hasattr(structured_data, "dict"):
                structured_data = structured_data.dict()
            elif hasattr(structured_data, "model_dump"):
                structured_data = structured_data.model_dump()
            
            raw_response.name = self.name
            
            self.chat_history.append({
                "role": "human",
                "human": context
            })
            self.chat_history.append({
                "role": "ai",
                "ai": raw_response.content
            })
            
            return {
                "message": raw_response,
                "structured_solution": structured_data,
                "solution": raw_response.content
            }
            
        except Exception as e:
            logger.error(f"Structured output failed: {str(e)}, falling back to standard output")
            
            callback_handler = OpenAICallbackHandler()
            config = {"callbacks": [callback_handler]}
            response = await self.llm.ainvoke(messages, config=config)
            response.name = self.name
            
            if isinstance(response, AIMessage):
                response.usage_metadata = {
                    "input_tokens": callback_handler.prompt_tokens,
                    "output_tokens": callback_handler.completion_tokens,
                    "total_tokens": callback_handler.total_tokens,
                    "input_token_details": {},
                    "output_token_details": {"reasoning": callback_handler.completion_tokens}
                }
            
            self.chat_history.append({
                "role": "human",
                "human": context
            })
            self.chat_history.append({
                "role": "ai",
                "ai": response.content
            })
            
            return {
                "message": response,
                "solution": response.content
            }

class ResultExtractor:
    """Extract final results from conversation history"""
    def __init__(self, model_name: str = None, format_prompt: str = ""):
        self.model_name = model_name or os.getenv("MODEL_NAME", "gpt-4o")
        self.format_prompt = format_prompt or """
- Ensure the final answer is a single line with no extra whitespace or formatting.
- Match the answer format to the problem type, such as:
   - Boolean problems: 'True' or 'False'
   - date_understanding: '(A)', '(B)', '(C)', etc.
   - Multiple-choice problems: '(A)', '(B)', '(C)', etc.
   - Sequence completion problems: A sequence of closing brackets like `)`, `]`, `}`, or `>`
   - Word sorting problems: Space-separated words in alphabetical order
   - Causal judgment or web of lies problems: 'Yes' or 'No'
   - Sports understanding problems: 'Yes' or 'No'
   - Formal fallacies: 'valid' or 'invalid'
<answer>
[Your final answer here]
</answer>
"""
        self.llm = ChatOpenAI(
            model=self.model_name,
            api_key=os.getenv("OPENAI_API_KEY", "sk-nhgJRZ0BQsKljo9ZOOcS4qlKejpptHpStgXjnpCLmmbb63pS"),
            base_url=os.getenv("OPENAI_API_BASE", "https://35.aigcbest.top/v1"),
            request_timeout=60,
            max_retries=2
        )
        self.name = "result_extractor"
        
    async def extract(self, all_histories: List[List[Dict[str, str]]], problem: str) -> Dict[str, Any]:
        """Extract final answer from all agents' conversation histories"""
        prompt = f"""Original problem: {problem}

Below are the discussion histories of multiple AI agents:

{self._format_histories(all_histories)}

Please analyze the above discussions and provide a final answer. Requirements:
- Synthesize all agents' viewpoints.
- Choose the most reasonable solution/option.
- Ensure the final answer is a single line with no extra whitespace or formatting.
- For multiple-choice problems, return only the option letter in the format '(A)', '(B)', '(C)', etc., without additional text.
- For web of lies problems, the answer must be 'Yes' or 'No' wrapped in <answer> tags.
{self.format_prompt}
"""
        messages = [
            SystemMessage(content="You are a professional result analyzer, responsible for extracting the final answer from discussions of multiple AI agents."),
            HumanMessage(content=prompt)
        ]
        
        try:
            callback_handler = OpenAICallbackHandler()
            config = {"callbacks": [callback_handler]}
            response = await self.llm.ainvoke(messages, config=config)
            response.name = "evaluator"
            
            if isinstance(response, AIMessage):
                response.usage_metadata = {
                    "input_tokens": callback_handler.prompt_tokens,
                    "output_tokens": callback_handler.completion_tokens,
                    "total_tokens": callback_handler.total_tokens,
                    "input_token_details": {},
                    "output_token_details": {"reasoning": callback_handler.completion_tokens}
                }
                
            return {
                "message": response
            }
        except Exception as e:
            logger.error(f"LLM call failed: {str(e)}")
            return {
                "message": None
            }

    def _format_histories(self, all_histories: List[List[Dict[str, str]]]) -> str:
        """Format all conversation histories"""
        formatted = []
        agent_names = [f"Agent_{i+1}" for i in range(len(all_histories))]
        for i, history in enumerate(all_histories):
            formatted.append(f"\n{agent_names[i]}'s discussion:")
            for msg in history:
                if msg.get("role") == "human":
                    formatted.append(f"Question: {msg['human']}")
                else:
                    formatted.append(f"Answer: {msg['ai']}")
        return "\n".join(formatted)

class Camel_Mas(AgentSystem):
    """
    LangChain多智能体系统

    该智能体系统使用LangChain框架中的多个智能体协作解决问题，包括任务指定、任务规划、执行和评估。
    """

    def __init__(self, name: str = "camel_mas", config: Dict[str, Any] = None):
        """初始化LangChain多智能体系统"""
        super().__init__(name, config)
        self.config = config or {}
        
        # 提取配置参数
        self.assistant_role_name = self.config.get("assistant_role_name", "助手")
        self.user_role_name = self.config.get("user_role_name", "用户")
        self.critic_role_name = self.config.get("critic_role_name", "批评者")
        self.task_type = self.config.get("task_type", "AI_SOCIETY")
        self.output_language = self.config.get("output_language", "中文")
        self.model_name = self.config.get("model_name") or os.getenv("MODEL_NAME", "gpt-4o")
        
        # API配置
        self.api_key = os.getenv("OPENAI_API_KEY", "sk-nhgJRZ0BQsKljo9ZOOcS4qlKejpptHpStgXjnpCLmmbb63pS")
        self.base_url = os.getenv("OPENAI_API_BASE", "https://35.aigcbest.top/v1")
        logger.info(f"使用API密钥: {self.api_key[:8]}... 端点: {self.base_url}")

        # 初始化OpenAI客户端（异步）
        try:
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            logger.info("AsyncOpenAI客户端初始化成功")
        except Exception as e:
            logger.error(f"初始化AsyncOpenAI客户端失败: {str(e)}")
            raise

        # 初始化智能体
        self._initialize_agents()

        # 为工具集成暴露LLM实例
        self.llm = ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url
        )
        self.workers = [
            {"name": self.assistant_role_name, "llm": self.llm},
            {"name": self.user_role_name, "llm": self.llm}
        ]

    def _initialize_agents(self):
        """初始化系统中使用的所有智能体"""
        self.task_specify_agent = Agent(
            agent_id="task_specify",
            name="TaskSpecifier",
            model_name=self.model_name,
            system_prompt=f"你是一个任务指定智能体。根据以下问题描述，明确具体任务，使用{self.output_language}。任务应清晰、具体、可执行。"
        )
        self.task_planner_agent = Agent(
            agent_id="task_planner",
            name="TaskPlanner",
            model_name=self.model_name,
            system_prompt=f"你是一个任务规划智能体。为以下任务制定详细的子任务计划，使用{self.output_language}。列出具体步骤。"
        )
        self.assistant_agent = Agent(
            agent_id="assistant",
            name=self.assistant_role_name,
            model_name=self.model_name,
            system_prompt=f"你是一个{self.assistant_role_name}，协助{self.user_role_name}完成任务。你的目标是提供准确、清晰的回答，使用{self.output_language}。"
        )
        self.user_agent = Agent(
            agent_id="user",
            name=self.user_role_name,
            model_name=self.model_name,
            system_prompt=f"你是{self.user_role_name}，与{self.assistant_role_name}协作完成任务。提出问题或响应助手，使用{self.output_language}。"
        )
        self.critic_agent = Agent(
            agent_id="critic",
            name=self.critic_role_name,
            model_name=self.model_name,
            system_prompt=f"你是{self.critic_role_name}，负责评估任务结果，提供建设性反馈。基于执行结果进行逻辑推理，生成四个具体的多选选项（A、B、C、D），每个选项必须严格匹配题目提供的选项（例如‘Yes’或‘No’），并选择最合理的答案。使用{self.output_language}。格式为：\nA. [选项内容]\nB. [选项内容]\nC. [选项内容]\nD. [选项内容]\n答案：[选中的选项]"
        )
        self.result_extractor = ResultExtractor(
            model_name=self.model_name,
            format_prompt=self._get_format_prompt()
        )
        logger.info("所有智能体初始化成功")

    def _get_format_prompt(self) -> str:
        """Retrieve the BBH format prompt for web of lies problems"""
        return """
- Ensure the final answer is a single line with no extra whitespace or formatting.
- Match the answer format to the problem type, such as:
   - Boolean problems: 'True' or 'False'
   - date_understanding: '(A)', '(B)', '(C)', etc.
   - Multiple-choice problems: '(A)', '(B)', '(C)', etc.
   - Sequence completion problems: A sequence of closing brackets like `)`, `]`, `}`, or `>`
   - Word sorting problems: Space-separated words in alphabetical order
   - Causal judgment or web of lies problems: 'Yes' or 'No'
   - Sports understanding problems: 'Yes' or 'No'
   - Formal fallacies: 'valid' or 'invalid'
<answer>
[Your final answer here]
</answer>
"""

    async def run_agent(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        在给定问题上运行多智能体系统。
        
        Args:
            problem: 包含问题数据的字典
            
        Returns:
            包含消息和使用元数据的运行结果字典
        """
        problem_text = problem["problem"]
        problem_id = problem.get("id", "unknown")
        logger.info(f"处理问题ID: {problem_id}, 问题: {problem_text}")

        # 步骤1: 任务指定
        specified_task = await self._specify_task(problem_text)

        # 步骤2: 任务规划
        planned_tasks = await self._plan_tasks(specified_task)

        # 步骤3: 通过聊天智能体执行任务
        execution_result = await self._execute_tasks(planned_tasks)

        # 步骤4: 使用批评者智能体评估结果
        critic_result = await self._evaluate_result(execution_result)

        # 步骤5: 提取最终答案
        all_histories = [
            self.task_specify_agent.chat_history,
            self.task_planner_agent.chat_history,
            self.assistant_agent.chat_history,
            self.user_agent.chat_history,
            self.critic_agent.chat_history
        ]
        final_result = await self.result_extractor.extract(all_histories, problem_text)

        # 格式化最终消息
        with contextlib.suppress(UnicodeDecodeError):
            final_answer = final_result["message"].content.encode('utf-8').decode('utf-8-sig')

        ai_message = {
            'content': final_answer,
            'name': self.assistant_role_name,
            'role': 'assistant',
            'message_type': 'ai_response',
            'usage_metadata': final_result["message"].usage_metadata
        }

        # 记录智能体响应
        self._record_agent_responses(problem_id, [ai_message])

        return {
            "messages": [ai_message],
            "final_answer": final_answer
        }

    async def _specify_task(self, problem_text: str) -> str:
        """使用TaskSpecifyAgent根据问题指定任务"""
        try:
            result = await self.task_specify_agent.generate_response(problem_text)
            logger.info("任务指定成功")
            return result["solution"]
        except Exception as e:
            logger.error(f"任务指定失败: {str(e)}")
            raise

    async def _plan_tasks(self, specified_task: str) -> List[str]:
        """使用TaskPlannerAgent规划子任务"""
        try:
            result = await self.task_planner_agent.generate_response(specified_task)
            logger.info("任务规划成功")
            return result["solution"].split('\n')
        except Exception as e:
            logger.error(f"任务规划失败: {str(e)}")
            raise

    async def _execute_tasks(self, tasks: List[str]) -> str:
        """使用ChatAgents执行规划任务"""
        execution_result = ""
        for task in tasks:
            init_msg = f"让我们开始这个任务: {task}"
            user_response = await self.user_agent.generate_response(init_msg)
            assistant_response = await self.assistant_agent.generate_response(user_response["solution"])
            execution_result += assistant_response["solution"] + "\n"
        return execution_result.strip()

    async def _evaluate_result(self, execution_result: str) -> str:
        """使用CriticAgent评估执行结果并生成多选选项"""
        result = await self.critic_agent.generate_response(execution_result)
        return result["solution"]

    def _record_agent_responses(self, problem_id: str, messages: List[Dict[str, Any]]):
        """记录智能体响应以便日志记录或进一步处理"""
        pass

# 注册智能体系统
AgentSystemRegistry.register("camel_mas", Camel_Mas)

