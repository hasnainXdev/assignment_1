from agents import Agent, OpenAIChatCompletionsModel, RunConfig, Runner, function_tool
from openai import AsyncOpenAI
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

# Set up Gemini Client
gemini_api_key = os.getenv("GEMINI_API_KEY")
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client,
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True,
)

get_country_capital = Agent(
    name="Get Country Capital Agent",
    instructions="""You are an expert at finding the capital city of a given country. When provided with a country name, 
        return only the name of the capital city as a single string (e.g., 'Islamabad' for Pakistan).
        Do not include additional text, explanations.""",
)

get_country_language = Agent(
    name="Get Country Language Agent",
    instructions="""
        You are an expert at identifying the national language of a given country. When provided with a country name,
        return only the name of the primary national language as a single string (e.g., 'Urdu' for Pakistan). 
        Do not include additional text, explanations.
        """,
)

get_country_population = Agent(
    name="Get Country Population Agent",
    instructions="""
    You are an expert at retrieving the population of a given country. When provided with a country name,
        return only the population as a number (e.g., '220 million' for Pakistan). 
        Do not include additional text, explanations.""",
)


# Using `as_tool`

get_country_capital_as_tool = get_country_capital.as_tool(
    tool_name="get_country_capital",
    tool_description="Retrieves the capital city of the given country as a single string.",
)

get_country_language_as_tool = get_country_language.as_tool(
    tool_name="get_country_language",
    tool_description="Retrieves the primary national language of the given country as a single string.",
)


get_country_population_as_tool = get_country_population.as_tool(
    tool_name="get_country_population",
    tool_description="Retrieves the population of the given country as a number.",
)

triage_agent = Agent(
    name="Assistant",
    instructions="""You are a smart Country Info Assistant. When given a country name
        you will use the tools to get the capital, language, and population of that country.
        
        You will use the tools provided to you to answer the question.
        
        To get the capital of country, use the tool `get_country_capital`
        To get the language of country, use the tool `get_country_language`
        To get the population of country, use the tool `get_country_population`
        
        """,
    tools=[
        get_country_capital_as_tool,
        get_country_language_as_tool,
        get_country_population_as_tool,
    ],
)


async def main():
    result = await Runner.run(
        triage_agent,
        run_config=config,
        input="what is the capital of pakistan including language and population?",
    )

    print(result.final_output)


asyncio.run(main())
