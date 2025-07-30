import json
import numpy as np
import time
import pickle
import random
from tqdm import tqdm
import os
from dotenv import load_dotenv
from openai import OpenAI
import asyncio
import aiohttp
from openai import AsyncOpenAI

# Load environment variables from .env file
load_dotenv(override=True)

num_questions = 100

# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=40,
    base_url=os.getenv("OPENAI_API_BASE")
)

# Initialize asynchronous OpenAI client
async_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=40,
    base_url=os.getenv("OPENAI_API_BASE")
)

def read_jsonl(path: str):
    """Read JSONL file"""
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]

def parse_bullets(sentence):
    bullets_preprocess = sentence.split("\n")
    bullets = []

    for bullet in bullets_preprocess:
        try:
            idx = bullet.find(next(filter(str.isalpha, bullet)))
        except:
            continue

        bullet = bullet[idx:]

        if len(bullet) != 0:
            bullets.append(bullet)

    return bullets


def generate_answer(answer_context):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                      model="gpt-4o-mini",
                      messages=answer_context,
                      n=1
                      )
            return completion
        except Exception as e:
            print(f"API call failed, retrying... (Attempt {attempt + 1}/{max_retries})")
            print(f"Error type: {type(e).__name__}")
            print(f"Error details: {str(e)}")
            if attempt < max_retries - 1:
                print("Waiting 20 seconds before retrying...")
                time.sleep(20)
            else:
                print("Maximum retries reached, skipping this request")
                # Return a mock completion object to avoid program crash
                class MockCompletion:
                    def __init__(self):
                        self.choices = [type('obj', (object,), {
                            'message': type('obj', (object,), {
                                'content': "Sorry, API error prevented generating an answer."
                            })()
                        })()]
                return MockCompletion()
    
    # This line will not be reached, but kept for completeness
    return None


async def generate_answer_async(answer_context):
    """Asynchronous API call function"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            completion = await async_client.chat.completions.create(
                      model="gpt-4o-mini",
                      messages=answer_context,
                      n=1
                      )
            return completion
        except Exception as e:
            print(f"API call failed, retrying... (Attempt {attempt + 1}/{max_retries})")
            print(f"Error type: {type(e).__name__}")
            print(f"Error details: {str(e)}")
            if attempt < max_retries - 1:
                print("Waiting 5 seconds before retrying...")  # Reduce waiting time
                await asyncio.sleep(5)
            else:
                print("Maximum retries reached, skipping this request")
                # Return a mock completion object to avoid program crash
                class MockCompletion:
                    def __init__(self):
                        self.choices = [type('obj', (object,), {
                            'message': type('obj', (object,), {
                                'content': "Sorry, API error prevented generating an answer."
                            })()
                        })()]
                return MockCompletion()
    
    return None


def construct_message(agents, question, idx):
    # Use introspection in the case in which there are no other agents.
    if len(agents) == 0:
        return {"role": "user", "content": "Can you verify that your answer is correct. Please reiterate your answer, making sure to state your answer at the end of the response."}

    prefix_string = "These are the recent/updated opinions from other agents: "

    for agent in agents:
        agent_response = agent[idx]["content"]
        response = "\n\n One agent response: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + "\n\n Use these opinions carefully as additional advice, can you provide an updated answer? Make sure to state your answer at the end of the response."
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
        return {"role": "assistant", "content": "Sorry, unable to get the answer content."}

def parse_answer(sentence):
    parts = sentence.split(" ")

    for part in parts[::-1]:
        try:
            answer = float(part)
            return answer
        except:
            continue


def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        current_frequency = List.count(i)
        if current_frequency > counter:
            counter = current_frequency
            num = i

    return num


async def process_single_question_async(question, answer, agents, rounds):
    """Asynchronous process for a single debate question"""
    # Create initial context for each agent
    agent_contexts = [[{"role": "user", "content": """Can you solve the following math problem? {} Please explain your reasoning step by step. Make sure to state your final answer clearly at the end of your response.""".format(question)}] for agent in range(agents)]

    # Multi-round debate
    for round in range(rounds):
        # Collect tasks for all agents
        tasks = []
        
        for i, agent_context in enumerate(agent_contexts):
            if round != 0:
                agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                message = construct_message(agent_contexts_other, question, 2*round - 1)
                agent_context.append(message)

            # Create asynchronous tasks
            task = generate_answer_async(agent_context)
            tasks.append((i, task))
        
        # Parallel execution of API calls for all agents
        print(f"Round {round + 1}: Parallel call to {len(tasks)} agents...")
        
        # Use asyncio.gather to truly parallelize all tasks
        task_results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
        
        # Process results
        for (i, _), result in zip(tasks, task_results):
            if isinstance(result, Exception):
                print(f"Agent {i} encountered an exception: {result}")
                # Create error response
                class MockCompletion:
                    def __init__(self):
                        self.choices = [type('obj', (object,), {
                            'message': type('obj', (object,), {
                                'content': f"Sorry, agent {i} encountered an error: {str(result)}"
                            })()
                        })()]
                result = MockCompletion()
            
            assistant_message = construct_assistant_message(result)
            agent_contexts[i].append(assistant_message)

    return agent_contexts


if __name__ == "__main__":
    agents = 2
    rounds = 3
    np.random.seed(0)
    random.seed(0)  # Set random seed for reproducibility

    generated_description = {}

    # Read math_test dataset
    # Construct correct file path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, "..", "dataset", "math_test.jsonl")
    math_questions = read_jsonl(dataset_path)
    
    # Asynchronously process questions
    async def main():
        # Randomly select the first 100 questions
        selected_questions = random.sample(math_questions, min(num_questions, len(math_questions)))
        for idx, data in enumerate(tqdm(selected_questions)):
            question = data['problem']
            answer = data['solution']
            
            print(f"\nProcessing question {idx + 1}: {question[:num_questions]}...")
            start_time = time.time()
            
            agent_contexts = await process_single_question_async(question, answer, agents, rounds)
            
            end_time = time.time()
            print(f"Question {idx + 1} processed, time taken: {end_time - start_time:.2f} seconds")
            
            generated_description[question] = (agent_contexts, answer)

        # Save results
        pickle.dump(generated_description, open("math_agents{}_rounds{}.p".format(agents, rounds), "wb"))
        print(f"Processed {len(generated_description)} questions with {agents} agents and {rounds} rounds")

    # Run the asynchronous main function
    asyncio.run(main())
