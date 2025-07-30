import copy
from typing import Dict, List

from langchain_core.tools import BaseTool

from langchain_experimental.autonomous_agents.hugginggpt.task_planner import Plan


class Task:
    """Task to be executed."""

    def __init__(self, task: str, id: int, dep: List[int], args: Dict, tool: BaseTool):
        self.task = task
        self.id = id
        self.dep = dep
        self.args = args
        self.tool = tool
        self.status = "pending"
        self.message = ""
        self.result = ""

    def __str__(self) -> str:
        return f"{self.task}({self.args})"

    def save_product(self) -> None:
        """Save text-based products to result field."""
        # For text-based tasks, we directly store the result
        # No file saving needed for text outputs
        if hasattr(self, 'product'):
            self.result = str(self.product)

    def completed(self) -> bool:
        return self.status == "completed"

    def failed(self) -> bool:
        return self.status == "failed"

    def pending(self) -> bool:
        return self.status == "pending"

    def run(self) -> str:
        """Execute the task using the associated tool."""
        try:
            new_args = copy.deepcopy(self.args)
            # For text-based tasks, execute tool and get result
            result = self.tool(**new_args)
            
            # Store result directly for text-based outputs
            if isinstance(result, str):
                self.result = result
            else:
                # If tool returns complex object, store as product and convert to string
                self.product = result
                self.save_product()
                
        except Exception as e:
            self.status = "failed"
            self.message = str(e)
            return self.message

        self.status = "completed"
        return self.result


class TaskExecutor:
    """Load tools and execute tasks."""

    def __init__(self, plan: Plan):
        self.plan = plan
        self.tasks = []
        self.id_task_map = {}
        self.status = "pending"
        for step in self.plan.steps:
            task = Task(step.task, step.id, step.dep, step.args, step.tool)
            self.tasks.append(task)
            self.id_task_map[step.id] = task

    def completed(self) -> bool:
        return all(task.completed() for task in self.tasks)

    def failed(self) -> bool:
        return any(task.failed() for task in self.tasks)

    def pending(self) -> bool:
        return any(task.pending() for task in self.tasks)

    def check_dependency(self, task: Task) -> bool:
        for dep_id in task.dep:
            if dep_id == -1:
                continue
            dep_task = self.id_task_map[dep_id]
            if dep_task.failed() or dep_task.pending():
                return False
        return True

    def update_args(self, task: Task) -> None:
        for dep_id in task.dep:
            if dep_id == -1:
                continue
            dep_task = self.id_task_map[dep_id]
            for k, v in task.args.items():
                if f"<resource-{dep_id}>" in v:
                    task.args[k] = task.args[k].replace(
                        f"<resource-{dep_id}>", dep_task.result
                    )

    def run(self) -> str:
        for task in self.tasks:
            print(f"running {task}")  # noqa: T201
            if task.pending() and self.check_dependency(task):
                self.update_args(task)
                task.run()
        if self.completed():
            self.status = "completed"
        elif self.failed():
            self.status = "failed"
        else:
            self.status = "pending"
        return self.status

    def __str__(self) -> str:
        result = ""
        for task in self.tasks:
            result += f"{task}\n"
            result += f"status: {task.status}\n"
            if task.failed():
                result += f"message: {task.message}\n"
            if task.completed():
                result += f"result: {task.result}\n"
        return result

    def __repr__(self) -> str:
        return self.__str__()

    def describe(self) -> str:
        return self.__str__()
