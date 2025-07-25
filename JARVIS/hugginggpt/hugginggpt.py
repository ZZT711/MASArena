from typing import List, Dict, Any
import json

from langchain.base_language import BaseLanguageModel
from langchain_core.tools import BaseTool

from response_generator import load_response_generator

from task_executor import TaskExecutor
from task_planner import load_chat_planner


class HuggingGPT:
    """Agent for interacting with HuggingGPT - Text Processing Version."""

    def __init__(self, llm: BaseLanguageModel, tools: List[BaseTool]):
        self.llm = llm
        self.tools = tools
        self.task_planner = load_chat_planner(llm)
        self.response_generator = load_response_generator(llm)
        self.task_executor = None

    def run(self, input: str) -> str:
        """Process text input through planning, execution, and response generation."""
        # Plan tasks based on input
        plan = self.task_planner.plan(inputs={"input": input, "hf_tools": self.tools})
        
        # Execute planned tasks
        self.task_executor = TaskExecutor(plan)
        execution_status = self.task_executor.run()
        
        # Generate response based on execution results
        response = self.response_generator.generate(
            {"task_execution": self.task_executor}
        )
        return response
    
    def get_plan(self, input: str):
        """Get the execution plan for debugging purposes."""
        return self.task_planner.plan(inputs={"input": input, "hf_tools": self.tools})
    
    def get_execution_details(self):
        """Get detailed execution information for analysis."""
        if hasattr(self, 'task_executor') and self.task_executor:
            return self.task_executor.describe()
        return "No execution performed yet."

    def run_with_trace(self, problem: str, format_prompt: str = None, **kwargs) -> Dict[str, Any]:
        """
        Run HuggingGPT and return results with full execution trace.

        Args:
            problem: input problem
            format_prompt: format prompt for guiding output
            **kwargs: other parameters
        """
        # Temporarily disable detailed langchain logging
        import logging
        import warnings
        
        # Silence various loggers
        loggers_to_silence = [
            "langchain",
            "langchain.chains", 
            "langchain.schema",
            "httpx",
        ]
        
        old_levels = {}
        for logger_name in loggers_to_silence:
            logger = logging.getLogger(logger_name)
            old_levels[logger_name] = logger.level
            logger.setLevel(logging.ERROR)
        
        # Suppress warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        try:
            # If a format prompt is provided, append it to the problem
            formatted_problem = problem
            if format_prompt:
                formatted_problem = f"{problem}\n\n{format_prompt}"
            
            # Use unified task_planner
            task_list = self.task_planner.plan(inputs={"input": formatted_problem, "hf_tools": self.tools}, **kwargs)
        finally:
            # Restore original log levels
            for logger_name, old_level in old_levels.items():
                logging.getLogger(logger_name).setLevel(old_level)
            warnings.filterwarnings("default", category=DeprecationWarning)
        
        if not task_list:
            # Similarly disable output for response_generator
            old_levels = {}
            for logger_name in loggers_to_silence:
                logger = logging.getLogger(logger_name)
                old_levels[logger_name] = logger.level
                logger.setLevel(logging.ERROR)
            
            try:
                # Pass format_prompt to response_generator as well
                response_input = f"Problem: {problem}\nTasks: []\nResults: []"
                if format_prompt:
                    response_input += f"\n\nFormat Requirements: {format_prompt}"
                
                response = self.response_generator.run(
                    problem=problem,
                    task_list="[]",
                    executed_task_list="[]",
                    format_prompt=format_prompt
                )
            finally:
                for logger_name, old_level in old_levels.items():
                    logging.getLogger(logger_name).setLevel(old_level)
            return {
                "tasks": [],
                "executed_tasks": [],
                "response": response,
                "messages": [("ai", response)] # Simulated message history
            }

        # Execute tasks
        self.task_executor = TaskExecutor(task_list)
        execution_status = self.task_executor.run()
        
        # Collect execution results
        executed_task_list = []
        for task in self.task_executor.tasks:
            executed_task_list.append({
                "task": task.task,
                "id": task.id,
                "status": task.status,
                "result": task.result,
                "message": task.message
            })
        
        # Convert Plan object to a serializable format
        task_list_serializable = []
        for step in task_list.steps:
            task_list_serializable.append({
                "task": step.task,
                "id": step.id,
                "dep": step.dep,
                "args": step.args
            })
        
        # Disable response_generator output again
        old_levels = {}
        for logger_name in loggers_to_silence:
            logger = logging.getLogger(logger_name)
            old_levels[logger_name] = logger.level
            logger.setLevel(logging.ERROR)
        
        try:
            response = self.response_generator.run(
                problem=problem, 
                task_list=json.dumps(task_list_serializable), 
                executed_task_list=json.dumps(executed_task_list),
                format_prompt=format_prompt,
                **kwargs
            )
        finally:
            for logger_name, old_level in old_levels.items():
                logging.getLogger(logger_name).setLevel(old_level)
        
        # Collect all messages for evaluation
        messages = []
        if executed_task_list and isinstance(executed_task_list, list) and len(executed_task_list) > 0:
            last_result = executed_task_list[-1].get("result", "")
            if isinstance(last_result, list):
                messages = last_result
            elif isinstance(last_result, str):
                messages = [("ai", last_result)]
        
        if not messages:
            messages = [("ai", response)]

        return {
            "tasks": task_list,
            "executed_tasks": executed_task_list,
            "response": response,
            "messages": messages
        }
