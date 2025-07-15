"""
LangChain Multi-Agent System

This module implements a multi-agent system based on the LangChain framework, using custom agents to collaboratively solve problems, replacing the original CAMEL framework.
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

# Load environment variables
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
    LangChain Multi-Agent System

    This agent system uses multiple agents from the LangChain framework to collaboratively solve problems, including task specification, planning, execution, and evaluation.
    """

    def __init__(self, name: str = "camel_mas", config: Dict[str, Any] = None):
        """Initialize the LangChain Multi-Agent System"""
        super().__init__(name, config)
        self.config = config or {}
        
        # Extract configuration parameters
        self.assistant_role_name = self.config.get("assistant_role_name", "Assistant")
        self.user_role_name = self.config.get("user_role_name", "User")
        self.critic_role_name = self.config.get("critic_role_name", "Critic")
        self.task_type = self.config.get("task_type", "AI_SOCIETY")
        self.output_language = self.config.get("output_language", "English")
        self.model_name = self.config.get("model_name") or os.getenv("MODEL_NAME", "gpt-4o")
        
        # API configuration
        self.api_key = os.getenv("OPENAI_API_KEY", "sk-nhgJRZ0BQsKljo9ZOOcS4qlKejpptHpStgXjnpCLmmbb63pS")
        self.base_url = os.getenv("OPENAI_API_BASE", "https://35.aigcbest.top/v1")
        logger.info(f"Using API key: {self.api_key[:8]}... Endpoint: {self.base_url}")

        # Initialize OpenAI client (asynchronous)
        try:
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            logger.info("AsyncOpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AsyncOpenAI client: {str(e)}")
            raise

        # Initialize agents
        self._initialize_agents()

        # Expose LLM instance for tool integration
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
        """Initialize all agents used in the system"""
        self.task_specify_agent = Agent(
            agent_id="task_specify",
            name="TaskSpecifier",
            model_name=self.model_name,
            system_prompt=f"You are a task specification agent. Based on the following problem description, specify a clear, specific, and executable task, using {self.output_language}."
        )
        self.task_planner_agent = Agent(
            agent_id="task_planner",
            name="TaskPlanner",
            model_name=self.model_name,
            system_prompt=f"You are a task planning agent. Create a detailed sub-task plan for the following task, using {self.output_language}. List specific steps."
        )
        self.assistant_agent = Agent(
            agent_id="assistant",
            name=self.assistant_role_name,
            model_name=self.model_name,
            system_prompt=f"You are an {self.assistant_role_name}, assisting the {self.user_role_name} to complete tasks. Your goal is to provide accurate and clear responses, using {self.output_language}."
        )
        self.user_agent = Agent(
            agent_id="user",
            name=self.user_role_name,
            model_name=self.model_name,
            system_prompt=f"You are the {self.user_role_name}, collaborating with the {self.assistant_role_name} to complete tasks. Ask questions or respond to the assistant, using {self.output_language}."
        )
        self.critic_agent = Agent(
            agent_id="critic",
            name=self.critic_role_name,
            model_name=self.model_name,
            system_prompt=f"You are the {self.critic_role_name}, responsible for evaluating task results and providing constructive feedback. Perform logical reasoning based on the execution results, generate four specific multiple-choice options (A, B, C, D), each option must strictly match the options provided in the problem (e.g., 'Yes' or 'No'), and select the most reasonable answer. Use {self.output_language}. Format: \nA. [Option content]\nB. [Option content]\nC. [Option content]\nD. [Option content]\nAnswer: [Selected option]"
        )
        self.result_extractor = ResultExtractor(
            model_name=self.model_name,
            format_prompt=self._get_format_prompt()
        )
        logger.info("All agents initialized successfully")

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
        Run the multi-agent system on the given problem.
        
        Args:
            problem: Dictionary containing problem data
            
        Returns:
            Dictionary containing the run results with messages and usage metadata
        """
        problem_text = problem["problem"]
        problem_id = problem.get("id", "unknown")
        logger.info(f"Processing problem ID: {problem_id}, Problem: {problem_text}")

        # Step 1: Task Specification
        specified_task = await self._specify_task(problem_text)

        # Step 2: Task Planning
        planned_tasks = await self._plan_tasks(specified_task)

        # Step 3: Execute tasks using chat agents
        execution_result = await self._execute_tasks(planned_tasks)

        # Step 4: Evaluate results using critic agent
        critic_result = await self._evaluate_result(execution_result)

        # Step 5: Extract final answer
        all_histories = [
            self.task_specify_agent.chat_history,
            self.task_planner_agent.chat_history,
            self.assistant_agent.chat_history,
            self.user_agent.chat_history,
            self.critic_agent.chat_history
        ]
        final_result = await self.result_extractor.extract(all_histories, problem_text)

        # Format final message
        with contextlib.suppress(UnicodeDecodeError):
            final_answer = final_result["message"].content.encode('utf-8').decode('utf-8-sig')

        ai_message = {
            'content': final_answer,
            'name': self.assistant_role_name,
            'role': 'assistant',
            'message_type': 'ai_response',
            'usage_metadata': final_result["message"].usage_metadata
        }

        # Record agent responses
        self._record_agent_responses(problem_id, [ai_message])

        return {
            "messages": [ai_message],
            "final_answer": final_answer
        }

    async def _specify_task(self, problem_text: str) -> str:
        """Use TaskSpecifyAgent to specify tasks based on the problem"""
        try:
            result = await self.task_specify_agent.generate_response(problem_text)
            logger.info("Task specification successful")
            return result["solution"]
        except Exception as e:
            logger.error(f"Task specification failed: {str(e)}")
            raise

    async def _plan_tasks(self, specified_task: str) -> List[str]:
        """Use TaskPlannerAgent to plan subtasks"""
        try:
            result = await self.task_planner_agent.generate_response(specified_task)
            logger.info("Task planning successful")
            return result["solution"].split('\n')
        except Exception as e:
            logger.error(f"Task planning failed: {str(e)}")
            raise

    async def _execute_tasks(self, tasks: List[str]) -> str:
        """Execute planned tasks using ChatAgents"""
        execution_result = ""
        for task in tasks:
            init_msg = f"Let's start this task: {task}"
            user_response = await self.user_agent.generate_response(init_msg)
            assistant_response = await self.assistant_agent.generate_response(user_response["solution"])
            execution_result += assistant_response["solution"] + "\n"
        return execution_result.strip()

    async def _evaluate_result(self, execution_result: str) -> str:
        """Use CriticAgent to evaluate execution results and generate multiple-choice options"""
        result = await self.critic_agent.generate_response(execution_result)
        return result["solution"]

    def _record_agent_responses(self, problem_id: str, messages: List[Dict[str, Any]]):
        """Record agent responses for logging or further processing"""
        pass

# Register the agent system
AgentSystemRegistry.register("camel_mas", Camel_Mas)
