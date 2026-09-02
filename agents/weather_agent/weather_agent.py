# Weather Agent
# ORCA - Marine Ecosystem Reasoning with Collaborative Agents

import requests
from datetime import datetime, timezone


def get_weather(location):
    try:
        # Keep the wttr.in API implementation
        url = f"https://wttr.in/{location}?format=j1"

        response = requests.get(url, timeout=10)
        response.raise_for_status()

        weather_data = response.json()

        # Extract current weather
        current = weather_data["current_condition"][0]

        temperature = current.get("temp_C", "N/A")
        humidity = current.get("humidity", "N/A")
        wind_speed = current.get("windspeedKmph", "N/A")

        # Extract weather condition
        condition = "N/A"

        if "weatherDesc" in current and current["weatherDesc"]:
            condition = current["weatherDesc"][0].get("value", "N/A")

        # ORCA Common AgentResponse format
        result = {
            "agent": "weather_agent",

            "status": "success",

            "location": {
                "name": location,
                "latitude": None,
                "longitude": None
            },

            "timestamp": datetime.now(timezone.utc).isoformat(),

            "data": {
                "temperature": f"{temperature}°C",
                "condition": condition,
                "humidity": f"{humidity}%",
                "wind_speed": f"{wind_speed} km/h"
            },

            "assessment": {
                "risk_level": "low",
                "summary": "Current weather conditions retrieved successfully."
            },

            "sources": [
                "wttr.in"
            ],

            "confidence": 0.8,

            "errors": []
        }

        return result

    except requests.exceptions.RequestException as error:

        return {
            "agent": "weather_agent",

            "status": "error",

            "location": {
                "name": location,
                "latitude": None,
                "longitude": None
            },

            "timestamp": datetime.now(timezone.utc).isoformat(),

            "data": {},

            "assessment": {
                "risk_level": "unknown",
                "summary": "Unable to retrieve weather data."
            },

            "sources": [
                "wttr.in"
            ],

            "confidence": 0.0,

            "errors": [
                str(error)
            ]
        }


if __name__ == "__main__":

    # Ask the user which state/location to test
    location = input("Enter the state or location: ").strip()

    if not location:
        print("Please enter a valid state or location.")

    else:
        weather_result = get_weather(location)
        print(weather_result)