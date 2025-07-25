import json
import re
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union

from langchain.base_language import BaseLanguageModel
from langchain.chains import LLMChain
from langchain_core.callbacks.manager import Callbacks
from langchain_core.prompts.chat import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel

DEMONSTRATIONS = [
    {
        "role": "user",
        "content": "解决这个数学问题：如果x + 2 = 5，那么x等于多少？并解释解题步骤",
    },
    {
        "role": "assistant", 
        "content": '[{{"task": "math_solver", "id": 0, "dep": [-1], "args": {{"equation": "x + 2 = 5", "show_steps": true}}}}, {{"task": "text_explainer", "id": 1, "dep": [0], "args": {{"content": "<resource-0>", "task_type": "math_explanation"}}}}]',
    },
    {
        "role": "user",
        "content": "给我总结这段文本的主要内容，然后回答：作者的主要观点是什么？",
    },
    {
        "role": "assistant",
        "content": '[{{"task": "text_summarizer", "id": 0, "dep": [-1], "args": {{"text": "input_text"}}}}, {{"task": "text_qa", "id": 1, "dep": [0], "args": {{"context": "<resource-0>", "question": "作者的主要观点是什么？"}}}}]',
    },
    {
        "role": "user", 
        "content": "编写一个Python函数来计算斐波那契数列的第n项，并测试这个函数",
    },
    {
        "role": "assistant",
        "content": '[{{"task": "code_generator", "id": 0, "dep": [-1], "args": {{"description": "编写Python函数计算斐波那契数列第n项", "language": "python"}}}}, {{"task": "code_tester", "id": 1, "dep": [0], "args": {{"code": "<resource-0>", "test_cases": "fibonacci sequence test cases"}}}}]',
    },
]


class TaskPlaningChain(LLMChain):
    """Chain to execute tasks."""

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        demos: List[Dict] = DEMONSTRATIONS,
        verbose: bool = False,  # 改为False减少输出
    ) -> LLMChain:
        """Get the response parser."""
        system_template = """#1 Task Planning Stage: The AI assistant can parse user input to several tasks: [{{"task": task, "id": task_id, "dep": dependency_task_id, "args": {{"input name": text may contain <resource-dep_id>}}}}]. The special tag "dep_id" refer to the one generated text/image/audio in the dependency task (Please consider whether the dependency task generates resources of this type.) and "dep_id" must be in "dep" list. The "dep" field denotes the ids of the previous prerequisite tasks which generate a new resource that the current task relies on. The task MUST be selected from the following tools (along with tool description, input name and output type): {tools}. There may be multiple tasks of the same type. Think step by step about all the tasks needed to resolve the user's request. Parse out as few tasks as possible while ensuring that the user request can be resolved. Pay attention to the dependencies and order among tasks. If the user input can't be parsed, you need to reply empty JSON []."""  # noqa: E501
        human_template = """Now I input: {input}."""
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            system_template
        )
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

        demo_messages: List[
            Union[HumanMessagePromptTemplate, AIMessagePromptTemplate]
        ] = []
        for demo in demos:
            if demo["role"] == "user":
                demo_messages.append(
                    HumanMessagePromptTemplate.from_template(demo["content"])
                )
            else:
                demo_messages.append(
                    AIMessagePromptTemplate.from_template(demo["content"])
                )
            # demo_messages.append(message)

        prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, *demo_messages, human_message_prompt]
        )

        return cls(prompt=prompt, llm=llm, verbose=verbose)


class Step:
    """A step in the plan."""

    def __init__(
        self, task: str, id: int, dep: List[int], args: Dict[str, str], tool: BaseTool
    ):
        self.task = task
        self.id = id
        self.dep = dep
        self.args = args
        self.tool = tool


class Plan:
    """A plan to execute."""

    def __init__(self, steps: List[Step]):
        self.steps = steps

    def __str__(self) -> str:
        return str([str(step) for step in self.steps])

    def __repr__(self) -> str:
        return str(self)


class BasePlanner(BaseModel):
    """Base class for a planner."""

    @abstractmethod
    def plan(self, inputs: dict, callbacks: Callbacks = None, **kwargs: Any) -> Plan:
        """Given input, decide what to do."""

    @abstractmethod
    async def aplan(
        self, inputs: dict, callbacks: Callbacks = None, **kwargs: Any
    ) -> Plan:
        """Asynchronous Given input, decide what to do."""


class PlanningOutputParser(BaseModel):
    """Parses the output of the planning stage."""
    
    class Config:
        arbitrary_types_allowed = True

    def parse(self, text: str, hf_tools: List[BaseTool]) -> Plan:
        """Parse the output of the planning stage.

        Args:
            text: The output of the planning stage.
            hf_tools: The tools available.

        Returns:
            The plan.
        """
        steps = []
        try:
            # 尝试找到JSON数组
            json_match = re.findall(r"\[.*\]", text)
            if not json_match:
                # 如果没有找到JSON数组，返回空计划
                return Plan(steps=[])
            
            # 尝试解析JSON
            try:
                task_list = json.loads(json_match[0])
            except json.JSONDecodeError:
                # JSON解析失败，返回空计划
                return Plan(steps=[])
            
            # 处理任务列表
            for v in task_list:
                if not isinstance(v, dict) or "task" not in v:
                    continue
                    
                choose_tool = None
                for tool in hf_tools:
                    if tool.name == v["task"]:
                        choose_tool = tool
                        break
                        
                if choose_tool:
                    steps.append(Step(v["task"], v["id"], v["dep"], v["args"], tool))
                    
        except Exception as e:
            # 任何其他错误，返回空计划
            pass
            
        return Plan(steps=steps)


class TaskPlanner(BasePlanner):
    """Planner for tasks."""

    llm_chain: LLMChain
    output_parser: PlanningOutputParser
    stop: Optional[List] = None
    
    class Config:
        arbitrary_types_allowed = True

    def __init__(self, llm: BaseLanguageModel, verbose: bool = False):
        """
        初始化任务规划器。
        
        Args:
            llm: 基础语言模型。
            verbose: 是否启用详细日志。
        """
        llm_chain = TaskPlaningChain.from_llm(llm, verbose=verbose)
        output_parser = PlanningOutputParser()
        stop = None
        
        # 使用pydantic的正确初始化方式
        super().__init__(
            llm_chain=llm_chain,
            output_parser=output_parser,
            stop=stop
        )

    def plan(self, inputs: dict, callbacks: Callbacks = None, **kwargs: Any) -> Plan:
        """Given input, decided what to do."""
        inputs["tools"] = [
            f"{tool.name}: {tool.description}" for tool in inputs["hf_tools"]
        ]
        llm_response = self.llm_chain.run(**inputs, stop=self.stop, callbacks=callbacks)
        return self.output_parser.parse(llm_response, inputs["hf_tools"])

    async def aplan(
        self, inputs: dict, callbacks: Callbacks = None, **kwargs: Any
    ) -> Plan:
        """Asynchronous Given input, decided what to do."""
        inputs["hf_tools"] = [
            f"{tool.name}: {tool.description}" for tool in inputs["hf_tools"]
        ]
        llm_response = await self.llm_chain.arun(
            **inputs, stop=self.stop, callbacks=callbacks
        )
        return self.output_parser.parse(llm_response, inputs["hf_tools"])


def load_chat_planner(llm: BaseLanguageModel) -> TaskPlanner:
    """Load the chat planner."""

    return TaskPlanner(llm = llm)
