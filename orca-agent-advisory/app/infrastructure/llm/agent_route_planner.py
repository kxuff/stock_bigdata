import json
import re
from dataclasses import dataclass
from typing import Any

from app.config import AgentSettings
from app.schemas.agent import AgentQueryRequest, RoutedAgentQuery
from app.schemas.enums import AgentRoute


AgentRoutePlan = RoutedAgentQuery


@dataclass
class LiteLLMAgentRoutePlanner:
    settings: AgentSettings

    def plan(self, request: AgentQueryRequest) -> RoutedAgentQuery:
        import litellm

        response = litellm.completion(
            model=self.settings.llm_model,
            api_key=self.settings.llm_api_key.get_secret_value() if self.settings.llm_api_key else None,
            api_base=self.settings.llm_base_url,
            temperature=0,
            timeout=min(self.settings.agent_timeout_seconds, 30),
            messages=[
                {"role": "system", "content": "You are production stock advisory router. Classify into one allowed route. Use out_of_scope for non-market requests. Return JSON only matching schema."},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "message": request.message,
                            "conversation_id": request.conversation_id,
                            "history": [item.model_dump(mode="json") for item in request.history],
                            "context": request.context.model_dump(),
                            "allowed_routes": [r.value for r in AgentRoute],
                            "schema": RoutedAgentQuery.model_json_schema(),
                        },
                        default=str,
                    ),
                },
            ],
        )
        return RoutedAgentQuery.model_validate(_parse_json(response["choices"][0]["message"]["content"]))


def _parse_json(content: Any) -> dict[str, Any]:
    if isinstance(content, dict):
        return content
    text = str(content).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)
