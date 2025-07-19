import asyncio
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

# ------------------ Agent 1: Mood Detector ------------------


@function_tool
def detect_mood(message: str) -> str:
    """
    Detect the user's mood from their message.
    Possible moods: happy, sad, stressed, neutral
    """
    message = message.lower()
    if any(word in message for word in ["sad", "upset", "down"]):
        return "sad"
    elif "stressed" in message:
        return "stressed"
    elif any(word in message for word in ["happy", "excited", "great"]):
        return "happy"
    else:
        return "neutral"


mood_detector_agent = Agent(
    name="Mood Detector Agent",
    instructions="you are an expert mood detector. Never guess. Always use the tool detect_mood.",
    tools=[detect_mood],
)

# ------------------ Agent 2: Activity Suggestor ------------------


@function_tool
def suggest_activity(mood: str) -> str:
    """
    Suggest an activity based on the user's mood.
    Only for sad or stressed moods.
    """
    if mood == "sad":
        return "Take a walk, call a friend, or listen to your favorite music."
    elif mood == "stressed":
        return "Try meditation or take a 10-minute break from work."
    else:
        return "You seem fine! Keep doing what you love."


activity_suggestor_agent = Agent(
    name="Activity Suggestor Agent",
    instructions="you are an expert activity suggestor. Never guess. Always use the tool suggest_activity.",
    tools=[suggest_activity],
)

# ------------------ Run Both Agents ------------------


async def main():
    user_message = str(input("Enter your mood: "))

    # Mood Detection
    mood = await Runner.run(mood_detector_agent, run_config=config, input=user_message)
    print(f"Mood detected: {mood.final_output}")

    # Activity Suggestion
    if "sad" in mood.final_output or "stressed" in mood.final_output:
        suggestion = await Runner.run(
            activity_suggestor_agent, run_config=config, input=mood.final_output
        )
        print(f"Suggested activity: {suggestion.final_output}")
    else:
        print("No activity suggestion needed.")


asyncio.run(main())
