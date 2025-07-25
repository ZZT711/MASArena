from typing import Any, List, Optional

from langchain.base_language import BaseLanguageModel
from langchain.chains import LLMChain
from langchain_core.callbacks.manager import Callbacks
from langchain_core.prompts import PromptTemplate


class ResponseGenerationChain(LLMChain):
    """Chain to execute tasks."""

    @classmethod
    def from_llm(cls, llm: BaseLanguageModel, verbose: bool = False) -> LLMChain:
        execution_template = (
            "The AI assistant has parsed the user input into several tasks"
            "and executed them. The results are as follows:\n"
            "{task_execution}"
            "\nPlease summarize the results and generate a response."
        )
        prompt = PromptTemplate(
            template=execution_template,
            input_variables=["task_execution"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)


class ResponseGenerator:
    """Generates a response based on the input."""

    def __init__(self, llm_chain: LLMChain, stop: Optional[List] = None):
        self.llm_chain = llm_chain
        self.stop = stop

    def generate(self, inputs: dict, callbacks: Callbacks = None, **kwargs: Any) -> str:
        """Given input, decided what to do."""
        llm_response = self.llm_chain.run(**inputs, stop=self.stop, callbacks=callbacks)
        return llm_response

    def run(self, problem: str, task_list: str, executed_task_list: str, format_prompt: str = None, **kwargs) -> str:
        """
        运行响应生成，兼容evaluation_framework中的调用方式
        
        Args:
            problem: 原始问题
            task_list: 任务列表
            executed_task_list: 执行结果列表
            format_prompt: 格式化提示，用于指导输出格式
        """
        # 构建基础输入
        task_execution = f"Problem: {problem}\nTasks: {task_list}\nResults: {executed_task_list}"
        
        # 如果提供了格式化提示，将其添加到输入中
        if format_prompt:
            task_execution += f"\n\nFormat Requirements:\n{format_prompt}"
        
        inputs = {
            "task_execution": task_execution
        }
        return self.generate(inputs, **kwargs)


def load_response_generator(llm: BaseLanguageModel) -> ResponseGenerator:
    """Load the ResponseGenerator."""

    llm_chain = ResponseGenerationChain.from_llm(llm, verbose=False)
    return ResponseGenerator(
        llm_chain=llm_chain,
    ) 