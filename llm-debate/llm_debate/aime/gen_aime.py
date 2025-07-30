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
num_questions = 30  # AIME test 30 questions (all questions in the dataset)
# Initialize asynchronous OpenAI client
client = AsyncOpenAI(
    timeout=40,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)

def construct_message(agents, question, idx):
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Put your final answer in the form (X) at the end of your response."}

    prefix_string = "These are the solutions to the problem from other agents: "

    for agent in agents:
        agent_response = agent[idx]["content"]
        response = "\n\n One agent solution: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + """\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that other agents step by step. Put your answer in the form (X) at the end of your response.""".format(question)
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


def parse_question_answer_aime(data_item):
    """Parse AIME format question and answer"""
    question = data_item["question"]
    answer = data_item["answer"]
    
    # Construct prompt for AIME mathematical problems
    question = f"Can you solve this AIME-level mathematics problem as accurately as possible? {question} Show your work step by step and put your final answer as a number in parentheses (X) at the end of your response."
    
    return question, answer


def load_aime_data(file_path):
    """Load AIME JSONL format data"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


async def process_single_question(data_item, agents, rounds):
    """Asynchronously process a single question"""
    question, answer = parse_question_answer_aime(data_item)
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

    return question, (agent_contexts, answer)


async def process_batch_questions(aime_data, batch_size, agents, rounds):
    """Process questions in batches in parallel"""
    response_dict = {}
    
    selected_questions = aime_data[:num_questions]  # Use num_questions questions
    
    # Process in batches
    for i in range(0, num_questions, batch_size):
        batch = selected_questions[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(num_questions + batch_size - 1)//batch_size}")
        
        # Process all questions in this batch in parallel
        tasks = [process_single_question(data_item, agents, rounds) for data_item in batch]
        
        # Use tqdm to show progress
        results = await tqdm.gather(*tasks, desc=f"Batch {i//batch_size + 1}")
        
        # Collect results
        for question, result in results:
            response_dict[question] = result
        
        # Take a short break between batches to avoid API limits
        if i + batch_size < num_questions:
            await asyncio.sleep(2)
    
    return response_dict


async def main():
    agents = 3
    rounds = 2
    batch_size = 3  # Process 3 questions per batch, AIME questions are more difficult and require more time

    # Load AIME data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    aime_file = os.path.join(current_dir, "..", "dataset", "aime_test.jsonl")
    aime_data = load_aime_data(aime_file)

    random.seed(0)
    
    print(f"Starting to process {num_questions} AIME questions, using {agents} agents, {rounds} rounds of debate")
    print(f"Batch size: {batch_size}")
    
    start_time = time.time()
    response_dict = await process_batch_questions(aime_data, batch_size, agents, rounds)
    end_time = time.time()
    
    # Save results
    output_file = f"aime_{agents}_{rounds}.json"
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(response_dict, f, ensure_ascii=False, indent=2)
    
    print(f"Processing complete! Processed {len(response_dict)} questions")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main()) 