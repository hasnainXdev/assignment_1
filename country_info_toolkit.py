from agents import Agent, OpenAIChatCompletionsModel, RunConfig, Runner, function_tool
from openai import AsyncOpenAI
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

get_contry_capital = Agent(
    name="Get Country Capital Agent",
    instructions="You are an expert country capital finder. Answer only with the capital of the given country.",
    run_config=config,
)

get_country_language = Agent(
    name="Get Country Language Agent",
    instructions="You are an expert country language finder. Answer only with the national language of the given country.",
    run_config=config,
)

get_country_population = Agent(
    name="Get Country Population Agent",
    instructions="You are an expert country population finder. Answer only with the population of the given country.",
    run_config=config,
)

triage_agent = Agent(
    name="Triage Agent",
    instructions=(
        "You are a smart assistant. When given a country name, use all tools to fetch its country capital, national language, and population. Summarize the info clearly. never guess only use tools"
    ),
    tools=[
        get_contry_capital.as_tool(
            tool_name="get_contry_capital",
            tool_description="get country capital",
        ),
        get_country_language.as_tool(
            tool_name="get_country_language",
            tool_description="get country language",
        ),
        get_country_population.as_tool(
            tool_name="get_country_population",
            tool_description="get country population",
        ),
    ],
)

result = Runner.run_sync(
    triage_agent,
    run_config=config,
    input="What is the capital of pakistan including population and language",
)

print(result.final_output)
