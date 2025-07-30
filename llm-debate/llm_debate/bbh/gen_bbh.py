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
    """Construct multi-agent debate messages"""
    if len(agents) == 0:
        return {"role": "user", "content": "Please carefully check if your answer is correct. Put your final answer in <answer></answer> tags."}

    prefix_string = "Here are the solutions from other agents for this problem: "

    for agent in agents:
        agent_response = agent[idx]["content"]
        response = "\n\n One agent's solution: ```{}```".format(agent_response)
        prefix_string = prefix_string + response

    prefix_string = prefix_string + """\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? Please examine your solution and those of other agents step by step. Put your final answer in <answer></answer> tags."""
    return {"role": "user", "content": prefix_string}


def construct_assistant_message(completion):
    """Construct assistant message"""
    # Check if completion is a string (error case)
    if isinstance(completion, str):
        print(f"Warning: Received string response instead of API object: {completion[:100]}...")
        return {"role": "assistant", "content": "API returned incorrect response format"}
    
    # Check if there is a 'choices' attribute
    if hasattr(completion, 'choices') and len(completion.choices) > 0:
        content = completion.choices[0].message.content
        return {"role": "assistant", "content": content}
    else:
        # Fallback
        return {"role": "assistant", "content": "Sorry, unable to get response content."}


async def generate_answer(answer_context):
    """Async function to generate answers"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            completion = await client.chat.completions.create(
                      model="gpt-4o-mini",
                      messages=answer_context,
                      n=1)
            return completion
        except Exception as e:
            print(f"API call error, retrying... (attempt {attempt + 1}/{max_retries})")
            print(f"Error type: {type(e).__name__}")
            print(f"Error details: {str(e)}")
            if attempt < max_retries - 1:
                print("Waiting 20 seconds before retry...")
                await asyncio.sleep(20)
            else:
                print("Reached maximum retry attempts, skipping this request")
                # Return a mock completion object to avoid program crash
                class MockCompletion:
                    def __init__(self):
                        self.choices = [type('obj', (object,), {
                            'message': type('obj', (object,), {
                                'content': "Sorry, unable to generate response due to API error."
                            })()
                        })()]
                return MockCompletion()
    
    return None


def parse_question_answer_bbh(data_item):
    """Parse BBH format questions and answers"""
    question = data_item["input"]
    answer = data_item["target"]
    task_id = data_item["task_id"]
    
    # Build question format based on BBH task type
    if "boolean_expressions" in task_id:
        formatted_question = f"Please calculate the value of the following boolean expression: {question}\n\nPut your final answer (True or False) in <answer></answer> tags."
    elif "causal_judgement" in task_id:
        formatted_question = f"{question}\n\nPlease carefully analyze and choose the correct answer (Yes or No). Put your final answer in <answer></answer> tags."
    elif "date_understanding" in task_id:
        formatted_question = f"{question}\n\nPlease carefully calculate the date and choose the correct answer. Put your final answer (e.g., (A), (B), etc.) in <answer></answer> tags."
    elif "disambiguation_qa" in task_id:
        formatted_question = f"{question}\n\nPlease carefully analyze the pronoun reference and choose the correct answer. Put your final answer (e.g., (A), (B), etc.) in <answer></answer> tags."
    elif "dyck_languages" in task_id:
        formatted_question = f"Please complete the following sequence, ensuring proper bracket matching: {question}\n\nPut your final answer in <answer></answer> tags."
    elif "formal_fallacies" in task_id:
        formatted_question = f"{question}\n\nPlease determine whether this argument is valid or invalid. Put your final answer in <answer></answer> tags."
    elif "geometric_shapes" in task_id:
        formatted_question = f"{question}\n\nPlease analyze the SVG path and choose the correct shape. Put your final answer (e.g., (A), (B), etc.) in <answer></answer> tags."
    elif "hyperbaton" in task_id:
        formatted_question = f"{question}\n\nPlease choose the sentence with correct adjective order. Put your final answer (e.g., (A), (B), etc.) in <answer></answer> tags."
    elif "logical_deduction" in task_id:
        formatted_question = f"{question}\n\nPlease perform logical deduction based on the given conditions and choose the correct answer. Put your final answer (e.g., (A), (B), etc.) in <answer></answer> tags."
    else:
        # Default format
        formatted_question = f"Please answer the following question: {question}\n\nPlease carefully analyze and put your final answer(if have options, please put the true option in <answer></answer> tags) in <answer></answer> tags."
    
    return formatted_question, answer


def load_bbh_data(file_path):
    """Load BBH JSONL format data"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


async def process_single_question(data_item, agents, rounds):
    """Async process single question"""
    question, _ = parse_question_answer_bbh(data_item)
    agent_contexts = [[{"role": "user", "content": question}] for _ in range(agents)]

    for round_num in range(rounds):
        tasks = []
        for i, agent_context in enumerate(agent_contexts):
            if round_num != 0:
                other_agent_contexts = agent_contexts[:i] + agent_contexts[i+1:]
                # The index for the other agents' responses is 2 * round_num - 1
                message = construct_message(other_agent_contexts, question, 2 * round_num - 1)
                agent_context.append(message)
            
            tasks.append(generate_answer(agent_context))
        
        completions = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, completion in enumerate(completions):
            if isinstance(completion, Exception):
                print(f"Agent {i} encountered an error in round {round_num}: {completion}")
                assistant_message = {"role": "assistant", "content": "Unable to generate response due to API error."}
            else:
                assistant_message = construct_assistant_message(completion)
            
            agent_contexts[i].append(assistant_message)

    # Use original input as key, and save the entire data_item as test_data
    return data_item["input"], (agent_contexts, data_item, data_item["task_id"])


async def process_batch_questions(bbh_data, batch_size, agents, rounds):
    """Process questions in parallel batches"""
    response_dict = {}
    
    # Randomly select questions
    if len(bbh_data) < num_questions:
        print(f"Warning: Number of available questions ({len(bbh_data)}) is less than requested ({num_questions}). Using all available questions.")
        selected_questions = bbh_data
    else:
        selected_questions = random.sample(bbh_data, num_questions)
    
    # Process in batches
    for i in range(0, len(selected_questions), batch_size):
        batch = selected_questions[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(selected_questions) + batch_size - 1)//batch_size}")
        
        # Process all questions in this batch in parallel
        tasks = [process_single_question(data_item, agents, rounds) for data_item in batch]
        
        # Use tqdm to show progress
        results = await tqdm.gather(*tasks, desc=f"Batch {i//batch_size + 1}")
        
        # Collect results
        for key, result in results:
            response_dict[key] = result
        
        # Brief pause between batches to avoid API limits
        if i + batch_size < len(selected_questions):
            await asyncio.sleep(2)
    
    return response_dict


async def main():
    agents = 2
    rounds = 3
    batch_size = 5  # Process 5 questions per batch, can be adjusted based on API limits

    # Load BBH data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    bbh_file = os.path.join(current_dir, "..", "dataset", "bbh_test.jsonl")
    bbh_data = load_bbh_data(bbh_file)

    random.seed(0)
    
    print(f"Starting to process {num_questions} BBH questions with {agents} agents, {rounds} rounds of debate")
    print(f"Batch size: {batch_size}")
    
    start_time = time.time()
    response_dict = await process_batch_questions(bbh_data, batch_size, agents, rounds)
    end_time = time.time()
    
    # Save results
    output_file = f"bbh_{agents}_{rounds}.json"
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(response_dict, f, ensure_ascii=False, indent=2)
    
    print(f"\nTotal processed {len(response_dict)} questions")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main()) 