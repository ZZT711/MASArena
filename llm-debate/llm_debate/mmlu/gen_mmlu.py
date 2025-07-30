from glob import glob
import pandas as pd
import json
import time
import random
import os
import asyncio
import aiohttp
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

# Load environment variables from .env file
load_dotenv(override=True)
num_questions = 100
# Initialize asynchronous OpenAI client
client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
    timeout=40
)

def construct_message(agents, question, idx):
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Put your final answer in the form (X) at the end of your response."}

    prefix_string = "These are the solutions to the problem from other agents: "

    for agent in agents:
        agent_response = agent[idx]["content"]
        response = "\n\n One agent solution: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + """\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that other agents step by step. Put your final answer letter inside <answer> tags. For example: <answer>X</answer>"""
    return {"role": "user", "content": prefix_string}


def construct_assistant_message(completion):
    # Check if completion is a string (error case)
    if isinstance(completion, str):
        print(f"Warning: Received string response instead of API object: {completion[:100]}...")
        return {"role": "assistant", "content": "API returned an incorrect response format"}
    
    # Check for choices attribute
    if hasattr(completion, 'choices') and len(completion.choices) > 0:
        content = completion.choices[0].message.content
        return {"role": "assistant", "content": content}
    else:
        # Fallback
        return {"role": "assistant", "content": "Sorry, I cannot get an answer."}


async def generate_answer(answer_context):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            completion = await client.chat.completions.create(
                      model="gpt-4o-mini",
                      messages=answer_context,
                      n=1)
            return completion
        except Exception as e:
            print(f"API call failed, retrying... (Attempt {attempt + 1}/{max_retries})")
            print(f"Error type: {type(e).__name__}")
            print(f"Error details: {str(e)}")
            if attempt < max_retries - 1:
                print("Waiting 20 seconds before retrying...")
                await asyncio.sleep(20)
            else:
                print("Maximum retries reached, skipping this request")
                # Return a mock completion object to avoid program crash
                class MockCompletion:
                    def __init__(self):
                        self.choices = [type('obj', (object,), {
                            'message': type('obj', (object,), {
                                'content': "Sorry, due to API error, I cannot generate an answer."
                            })()
                        })()]
                return MockCompletion()
    
    # This line will not be reached, but kept for completeness
    return None


def parse_question_answer_mmlu_pro(data_item):
    """Parse mmlu_pro format question and answer"""
    question = data_item["question"]
    options = data_item["options"]
    
    # Build option string, supporting up to 10 options (A-J)
    option_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    option_text = ""
    for i, option in enumerate(options[:10]):  # Process up to 10 options
        if i < len(option_letters):
            option_text += f"{option_letters[i]}) {option}, "
    
    # Remove trailing comma and space
    option_text = option_text.rstrip(", ")
    
    question = f"Can you answer the following question as accurately as possible? {question}: {option_text} Explain your answer, putting the final answer letter inside <answer> tags. For example: <answer>X</answer>"
    
    answer = data_item["answer"]
    
    return question, answer


def load_mmlu_pro_data(file_path):
    """Load mmlu_pro JSONL format data"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


async def process_single_question(data_item, agents, rounds):
    """Asynchronously process a single question"""
    question, answer = parse_question_answer_mmlu_pro(data_item)
    agent_contexts = [[{"role": "user", "content": question}] for agent in range(agents)]

    for round in range(rounds):
        # Prepare context for all agents
        tasks = []
        for i, agent_context in enumerate(agent_contexts):
            if round != 0:
                agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                message = construct_message(agent_contexts_other, question, 2 * round - 1)
                agent_context.append(message)
            
            tasks.append(generate_answer(agent_context))
        
        # Execute API calls in parallel for all agents
        completions = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and update context
        for i, completion in enumerate(completions):
            if isinstance(completion, Exception):
                print(f"Agent {i} encountered an error in round {round}: {completion}")
                # Create an error response
                assistant_message = {"role": "assistant", "content": "Due to API error, I cannot generate an answer."}
            else:
                assistant_message = construct_assistant_message(completion)
            
            agent_contexts[i].append(assistant_message)

    # Build test_data dictionary, containing all fields needed by the evaluator
    test_data = {
        "question": data_item.get("question", ""),
        "options": data_item.get("options", []),
        "answer": data_item.get("answer", ""),
        "answer_index": data_item.get("answer_index"),
        "question_id": data_item.get("question_id", data_item.get("task_id", f"q_{hash(question) % 10000}")),
        "category": data_item.get("category", ""),
        "src": data_item.get("src", "")
    }
    
    # If answer_index is None but there is an answer letter, calculate the corresponding index
    if test_data["answer"] and test_data["options"] and test_data["answer_index"] is None:
        try:
            answer_idx = ord(test_data["answer"].upper()) - ord('A')
            if 0 <= answer_idx < len(test_data["options"]):
                test_data["answer_index"] = answer_idx
        except Exception:
            pass
    
    # Get task_id, prioritize task_id field, then question_id
    task_id = data_item.get("task_id", data_item.get("question_id", test_data["question_id"]))
    
    # Return format expected by eval_mmlu.py: (responses, test_data, task_id)
    return question, (agent_contexts, test_data, task_id)


async def process_batch_questions(mmlu_pro_data, batch_size, agents, rounds):
    """Process questions in batches in parallel"""
    response_dict = {}
    
    # Randomly select 100 questions
    selected_questions = random.sample(mmlu_pro_data, min(num_questions, len(mmlu_pro_data)))
    
    # Process in batches
    for i in range(0, len(selected_questions), batch_size):
        batch = selected_questions[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(selected_questions) + batch_size - 1)//batch_size}")
        
        # Process all questions in this batch in parallel
        tasks = [process_single_question(data_item, agents, rounds) for data_item in batch]
        
        # Use tqdm to show progress
        results = await tqdm.gather(*tasks, desc=f"Batch {i//batch_size + 1}")
        
        # Collect results
        for question, result in results:
            response_dict[question] = result
        
        # Take a short break between batches to avoid API limits
        if i + batch_size < len(selected_questions):
            await asyncio.sleep(2)
    
    return response_dict


async def main():
    agents = 3
    rounds = 2
    batch_size = 5  # Process 5 questions per batch, adjust based on API limits

    # Load mmlu_pro data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    mmlu_pro_file = os.path.join(current_dir, "..", "..", "data", "mmlu_pro_test.jsonl")
    mmlu_pro_data = load_mmlu_pro_data(mmlu_pro_file)

    random.seed(0)
    
    print(f"Starting to process {num_questions} questions, using {agents} agents, {rounds} rounds of debate")
    print(f"Batch size: {batch_size}")
    
    start_time = time.time()
    response_dict = await process_batch_questions(mmlu_pro_data, batch_size, agents, rounds)
    end_time = time.time()
    
    # Save results
    output_file = f"mmlu_pro_{agents}_{rounds}.json"
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(response_dict, f, ensure_ascii=False, indent=2)
    
    print(f"Processing complete! Processed {len(response_dict)} questions")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
