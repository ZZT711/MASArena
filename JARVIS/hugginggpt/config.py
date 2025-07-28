"""
配置文件

管理 LLM、工具和评估设置的配置。
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import os
from langchain.llms.base import LLM
from langchain_core.tools import BaseTool
from dotenv import load_dotenv

load_dotenv(override=True)

@dataclass
class LLMConfig:
    """LLM配置"""
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout: int = 60
    api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    api_base: Optional[str] = os.getenv("OPENAI_API_BASE")


@dataclass
class ToolConfig:
    """工具配置"""
    enabled_tools: List[str] = None
    tool_timeout: int = 30
    max_tool_calls: int = 5


@dataclass
class EvaluationFrameworkConfig:
    """评估框架总配置"""
    llm_config: LLMConfig
    tool_config: ToolConfig
    data_dir: str = "hugginggpt/dataset"
    results_dir: str = "results"
    log_level: str = "INFO"
    
    def __post_init__(self):
        if self.tool_config.enabled_tools is None:
            self.tool_config.enabled_tools = []


def create_openai_llm(config: LLMConfig) -> LLM:
    """创建 OpenAI LLM"""
    try:
        from langchain_openai import ChatOpenAI
        
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            api_key=config.api_key,
            base_url=config.api_base,
            request_timeout=config.timeout
        )
    except ImportError:
        raise ImportError("需要安装 openai 包: pip install openai")


def create_azure_openai_llm(config: LLMConfig) -> LLM:
    """创建 Azure OpenAI LLM"""
    try:
        from langchain.llms import AzureOpenAI
        
        return AzureOpenAI(
            deployment_name=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            openai_api_key=config.api_key or os.getenv("AZURE_OPENAI_API_KEY"),
            openai_api_base=config.api_base or os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
            request_timeout=config.timeout
        )
    except ImportError:
        raise ImportError("需要安装 openai 包: pip install openai")


def create_huggingface_llm(config: LLMConfig) -> LLM:
    """创建 HuggingFace LLM"""
    try:
        from langchain.llms import HuggingFacePipeline
        from transformers import pipeline
        
        pipe = pipeline(
            "text-generation",
            model=config.model_name,
            max_new_tokens=config.max_tokens,
            temperature=config.temperature,
        )
        
        return HuggingFacePipeline(pipeline=pipe)
    except ImportError:
        raise ImportError("需要安装 transformers 包: pip install transformers torch")


def create_llm_from_config(config: LLMConfig, llm_type: str = "openai") -> LLM:
    """根据配置创建 LLM"""
    if llm_type.lower() == "openai":
        return create_openai_llm(config)
    elif llm_type.lower() == "azure":
        return create_azure_openai_llm(config)
    elif llm_type.lower() == "huggingface":
        return create_huggingface_llm(config)
    else:
        raise ValueError(f"不支持的 LLM 类型: {llm_type}")


def create_default_tools() -> List[BaseTool]:
    """创建默认工具列表"""
    tools = []
    
    try:
        # 数学工具
        from langchain.tools import WolframAlphaQueryRun
        from langchain.utilities import WolframAlphaAPIWrapper
        
        wolfram_alpha = WolframAlphaQueryRun(
            api_wrapper=WolframAlphaAPIWrapper(
                wolfram_alpha_appid=os.getenv("WOLFRAM_ALPHA_APPID")
            )
        )
        tools.append(wolfram_alpha)
    except (ImportError, Exception):
        pass  # 如果没有配置或导入失败，跳过
    
    try:
        # Python 代码执行工具
        from langchain.tools import PythonREPLTool
        python_repl = PythonREPLTool()
        tools.append(python_repl)
    except (ImportError, Exception):
        pass
    
    try:
        # 搜索工具
        from langchain.tools import DuckDuckGoSearchRun
        search = DuckDuckGoSearchRun()
        tools.append(search)
    except (ImportError, Exception):
        pass
    
    return tools


# 预定义配置
DEFAULT_CONFIG = EvaluationFrameworkConfig(
    llm_config=LLMConfig(
        model_name="gpt-4o-mini",
        temperature=0.0,
        max_tokens=2048,
        timeout=60
    ),
    tool_config=ToolConfig(
        enabled_tools=["python", "search"],
        tool_timeout=30,
        max_tool_calls=5
    ),
    data_dir="hugginggpt/dataset",
    results_dir="results",
    log_level="INFO"
)

MATH_OPTIMIZED_CONFIG = EvaluationFrameworkConfig(
    llm_config=LLMConfig(
        model_name="gpt-4o-mini",
        temperature=0.0,
        max_tokens=4096,
        timeout=120
    ),
    tool_config=ToolConfig(
        enabled_tools=["wolfram", "python"],
        tool_timeout=60,
        max_tool_calls=10
    ),
    data_dir="hugginggpt/dataset",
    results_dir="results/math",
    log_level="INFO"
)

CODING_OPTIMIZED_CONFIG = EvaluationFrameworkConfig(
    llm_config=LLMConfig(
        model_name="gpt-4o-mini",
        temperature=0.1,
        max_tokens=4096,
        timeout=180
    ),
    tool_config=ToolConfig(
        enabled_tools=["python"],
        tool_timeout=120,
        max_tool_calls=15
    ),
    data_dir="hugginggpt/dataset",
    results_dir="results/coding",
    log_level="INFO"
) 