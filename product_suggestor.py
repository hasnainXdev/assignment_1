from agents import Agent, OpenAIChatCompletionsModel, RunConfig, Runner, function_tool
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

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


product_suggestor = Agent(
    name="Product Suggestor Agent",
    instructions="You are an expert product suggestion agent. Suggest useful products based on the user's feelings, needs, or problems. For example, if the user says 'I'm tired,' suggest an energy drink or pillow with a short reason. Always respond in short, clear sentences. If you're unsure, ask a follow-up question.",
)


result = Runner.run_sync(product_suggestor, run_config=config, input="I'm feeling fever")

print(result.final_output)
