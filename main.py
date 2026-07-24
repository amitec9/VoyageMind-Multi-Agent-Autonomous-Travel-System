import uuid

from fastapi import FastAPI
from pydantic import BaseModel

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from graph import app

api = FastAPI(
    title="Real-World Multi-Agent Travel Planner",
    version="1.0.0"
)

# Store sessions (Use Redis/DB in production)
sessions = {}


class TravelRequest(BaseModel):
    user_id: str = "demo_user"
    query: str


class ApprovalRequest(BaseModel):
    thread_id: str
    approved: bool
    feedback: str = ""


@api.post("/travel/plan")
def create_plan(request: TravelRequest):

    thread_id = f"{request.user_id}_{uuid.uuid4().hex[:8]}"

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = app.invoke(
        {
            "messages": [
                HumanMessage(content=request.query)
            ],
            "user_id": request.user_id,
            "user_query": request.query,
            "flight_results": "",
            "hotel_results": "",
            "weather_results": "",
            "budget_results": "",
            "itinerary": "",
            "final_response": "",
            "llm_calls": 0,
        },
        config=config,
    )

    sessions[thread_id] = config

    if "__interrupt__" in result:
        draft = result["__interrupt__"][0].value.get(
            "draft_itinerary", ""
        )
        waiting = True
    else:
        draft = result.get("itinerary", "")
        waiting = False

    return {
        "thread_id": thread_id,
        "waiting_for_approval": waiting,
        "supervisor_reasoning": result.get(
            "supervisor_reasoning", ""
        ),
        "selected_agents": result.get(
            "selected_agents", []
        ),
        "flight_results": result.get(
            "flight_results", ""
        ),
        "hotel_results": result.get(
            "hotel_results", ""
        ),
        "weather_results": result.get(
            "weather_results", ""
        ),
        "budget_results": result.get(
            "budget_results", ""
        ),
        "draft_itinerary": draft,
    }


@api.post("/travel/approve")
def approve_plan(request: ApprovalRequest):

    config = sessions.get(request.thread_id)

    if config is None:
        return {
            "error": "Invalid thread_id"
        }

    result = app.invoke(
        Command(
            resume={
                "approved": request.approved,
                "feedback": request.feedback,
            }
        ),
        config=config,
    )

    return {
        "final_response": result.get("final_response", ""),
        "itinerary": result.get("itinerary", ""),
    }


@api.get("/")
def home():
    return {
        "message": "Travel Planner API is running."
    }